#!/bin/bash
# Durable local supervisor for one flywheel run.
# Arguments: INSTANCE_ID HOST PORT RUN_NAME CONTROL_DIR LOCAL_SSH_SESSION LOCAL_TB_SESSION ARCH

set -o pipefail

if [ "$#" -ne 8 ]; then
  echo "ERROR: supervisor requires exactly 8 arguments, received $#" >&2
  exit 2
fi

INSTANCE_ID="$1"
HOST="$2"
PORT="$3"
RUN_NAME="$4"
CONTROL_DIR="$5"
LOCAL_SSH_SESSION="$6"
LOCAL_TB_SESSION="$7"
ARCH="$8"

case "$INSTANCE_ID" in
  ""|*[!0-9]*) echo "ERROR: INSTANCE_ID must be numeric" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(readlink -f "${SCRIPT_DIR}/../../../..")
EMERGENCY_CLEANUP_ARMED=true

emergency_validation_cleanup() {
  local exit_code=$?
  if [ "$EMERGENCY_CLEANUP_ARMED" = true ] && [ "$exit_code" -ne 0 ]; then
    echo "ERROR: supervisor validation failed; destroying instance $INSTANCE_ID" >&2
    "${SCRIPT_DIR}/destroy-instance.sh" "$INSTANCE_ID"
  fi
}
trap emergency_validation_cleanup EXIT

case "$PORT" in
  ""|*[!0-9]*) echo "ERROR: PORT must be numeric" >&2; exit 2 ;;
esac
case "$RUN_NAME" in
  ""|*[!A-Za-z0-9._-]*) echo "ERROR: invalid RUN_NAME" >&2; exit 2 ;;
esac
case "$ARCH" in
  ""|*[!A-Za-z0-9._-]*) echo "ERROR: invalid ARCH" >&2; exit 2 ;;
esac
test "$CONTROL_DIR" = "/workspace/toy-pickplace/.flywheel/${RUN_NAME}" || {
  echo "ERROR: CONTROL_DIR does not match RUN_NAME" >&2
  exit 2
}
case "$LOCAL_SSH_SESSION" in
  vast-ssh-[0-9]*) ;;
  *) echo "ERROR: invalid local SSH session name" >&2; exit 2 ;;
esac
case "$LOCAL_TB_SESSION" in
  tb-flywheel-[0-9]*) ;;
  *) echo "ERROR: invalid local TensorBoard session name" >&2; exit 2 ;;
esac

STATUS_FILE="/tmp/toy-pickplace-flywheel-${RUN_NAME}.status"
WORKFLOW_INDEX=${LOCAL_SSH_SESSION#vast-ssh-}
OWNER_FILE="/tmp/toy-pickplace-flywheel-${WORKFLOW_INDEX}.owner"
SSH_CMD=(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
  -o ConnectTimeout=15 -p "$PORT" "root@$HOST")

STATE="starting"
SSH_FAILURES=0
FINAL_BACKUP_TOKEN=""
FINAL_BACKUP_DEADLINE=0
EVALUATION_DEADLINE=0
CLEANUP_STARTED=false
SUPERVISOR_TERMINAL=false

write_status() {
  local outcome="$1"
  local now
  now=$(date -Iseconds)
  printf '{"outcome":"%s","timestamp":"%s","run_name":"%s","instance_id":"%s","last_state":"%s"}\n' \
    "$outcome" "$now" "$RUN_NAME" "$INSTANCE_ID" "$STATE" \
    > "${STATUS_FILE}.tmp"
  mv "${STATUS_FILE}.tmp" "$STATUS_FILE"
}

remote_value() {
  "${SSH_CMD[@]}" "$1" 2>/dev/null | tail -1
}

remote_probe() {
  test "$(remote_value "printf '%s\\n' SSH_OK")" = "SSH_OK"
}

remote_lifecycle_state() {
  local session_name="$1"
  local completed_marker="$2"
  local failed_marker="$3"

  remote_value \
    "if tmux has-session -t '${session_name}' 2>/dev/null; then
       printf '%s\\n' running
     elif test -f '${completed_marker}'; then
       cat '${completed_marker}'
     elif test -f '${failed_marker}'; then
       cat '${failed_marker}'
     else
       printf '%s\\n' disappeared
     fi"
}

capture_local_diagnostics() {
  local local_dir="/tmp/toy-pickplace-flywheel-${RUN_NAME}"
  mkdir -p "$local_dir"
  {
    echo "=== Capture $(date -Iseconds) ==="
    echo "=== Run: ${RUN_NAME} ==="
    echo "=== Instance: ${INSTANCE_ID} ==="
    echo "=== Host: ${HOST}:${PORT} ==="
    echo "=== State: ${STATE} ==="
    echo ""
  } > "$local_dir/meta.txt"

  "${SSH_CMD[@]}" \
    "cat '${CONTROL_DIR}/state/'* 2>/dev/null; echo '---'; ls -la '${CONTROL_DIR}/state/'" \
    > "$local_dir/state.txt" 2>/dev/null || true

  for session in flywheel-run ckpt-bkp tensorboard heldout-eval; do
    "${SSH_CMD[@]}" "tmux capture-pane -t '$session' -p -S -300 2>/dev/null" \
      > "$local_dir/${session}.log" 2>/dev/null || true
  done

  "${SSH_CMD[@]}" "tail -500 '${CONTROL_DIR}/logs/${RUN_NAME}.log' 2>/dev/null" \
    > "$local_dir/training-tail.log" 2>/dev/null || true
  "${SSH_CMD[@]}" "tail -500 '${CONTROL_DIR}/logs/backup-${RUN_NAME}.log' 2>/dev/null" \
    > "$local_dir/backup-tail.log" 2>/dev/null || true
  echo "Local diagnostics saved to $local_dir"
}

upload_diagnostics() {
  local local_dir="/tmp/toy-pickplace-flywheel-${RUN_NAME}"
  local deadline=$(( $(date +%s) + 300 ))
  capture_local_diagnostics || true

  set -a
  . "${PROJECT_ROOT}/.env" 2>/dev/null || return 1
  set +a
  test -n "${S3_BUCKET:-}" || return 1

  for file in "$local_dir"/*; do
    test -f "$file" || continue
    [ "$(date +%s)" -gt "$deadline" ] && break
    key=".flywheel/${RUN_NAME}/diagnostics/$(basename "$file")"
    uv run python -c "
import boto3, os
s = boto3.Session(profile_name=os.environ.get('AWS_PROFILE') or None)
c = s.client('s3', region_name=os.environ.get('AWS_REGION') or 'ap-south-1')
c.upload_file('$file', os.environ['S3_BUCKET'], '$key')
" 2>/dev/null || true
  done
}

cleanup() {
  local outcome="$1"
  if [ "$CLEANUP_STARTED" = true ]; then
    return
  fi
  CLEANUP_STARTED=true
  echo "=== Cleanup: $outcome ==="
  capture_local_diagnostics || true
  write_status "cleanup-pending-${outcome}"
  echo "Instance destruction in 60s. SSH: ssh -p $PORT root@$HOST"
  "${SSH_CMD[@]}" \
    "printf '%s\n' 'pending-destruction at $(date -Iseconds)' > '${CONTROL_DIR}/state/pending-destruction'" \
    2>/dev/null || true
  sleep 60

  until "${SCRIPT_DIR}/destroy-instance.sh" "$INSTANCE_ID"; do
    echo "WARNING: destruction helper exited; retrying in 30 seconds"
    sleep 30
  done
  write_status "$outcome"

  exec 8>/tmp/toy-pickplace-flywheel-local-wrapper.lock
  if flock 8; then
    if [ -f "$OWNER_FILE" ] && [ "$(cat "$OWNER_FILE" 2>/dev/null)" = "$RUN_NAME" ]; then
      tmux kill-session -t "$LOCAL_SSH_SESSION" 2>/dev/null || true
      tmux kill-session -t "$LOCAL_TB_SESSION" 2>/dev/null || true
      rm -f "$OWNER_FILE"
    else
      echo "WARNING: wrapper ownership changed; local sessions were not killed"
    fi
    flock -u 8
  else
    echo "WARNING: wrapper lock unavailable; local sessions were not killed"
  fi
  exec 8>&-
  SUPERVISOR_TERMINAL=true
  echo "Cleanup complete: $outcome"
}

on_supervisor_exit() {
  local exit_code=$?
  if [ "$SUPERVISOR_TERMINAL" != true ] && [ "$CLEANUP_STARTED" != true ]; then
    echo "ERROR: supervisor exited unexpectedly with status $exit_code"
    upload_diagnostics || true
    cleanup "supervisor-exited-${exit_code}"
  fi
}

handle_supervisor_signal() {
  local exit_code="$1"
  if [ "$CLEANUP_STARTED" = true ]; then
    echo "Cleanup already in progress; deferring signal $exit_code"
    return
  fi
  exit "$exit_code"
}

refresh_endpoint() {
  local new_url new_host new_port probe
  new_url=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null | tail -1)
  test -n "$new_url" || return 1
  new_host=$(printf '%s\n' "$new_url" | sed 's|ssh://root@||;s|:.*||')
  new_port=$(printf '%s\n' "$new_url" | sed 's|.*:||')
  test -n "$new_host" && test -n "$new_port" || return 1
  if [ "$new_host" = "$HOST" ] && [ "$new_port" = "$PORT" ]; then
    return 1
  fi

  echo "Endpoint changed: $HOST:$PORT -> $new_host:$new_port"
  probe=$(ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
    -o ConnectTimeout=15 -p "$new_port" "root@$new_host" \
    "printf '%s\n' SSH_OK" 2>/dev/null | tail -1)
  test "$probe" = "SSH_OK" || return 1
  probe=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
    -o ConnectTimeout=15 -p "$new_port" "root@$new_host" \
    "printf '%s\n' SSH_OK" 2>/dev/null | tail -1)
  test "$probe" = "SSH_OK" || return 1

  HOST=$new_host
  PORT=$new_port
  SSH_CMD=(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
    -o ConnectTimeout=15 -p "$PORT" "root@$HOST")
  return 0
}

launch_heldout() {
  local backup_alive heldout_state heldout_alive

  "${SSH_CMD[@]}" "tmux kill-session -t ckpt-bkp 2>/dev/null || true" 2>/dev/null || true
  for i in $(seq 1 15); do
    backup_alive=$(remote_value "tmux has-session -t ckpt-bkp 2>/dev/null && echo yes || echo no")
    [ "$backup_alive" = "no" ] && break
    sleep 1
  done
  test "$backup_alive" = "no" || return 1

  heldout_alive=$(remote_value "tmux has-session -t heldout-eval 2>/dev/null && echo yes || echo no")
  test "$heldout_alive" = "no" || return 1
  heldout_state=$(remote_value \
    "test ! -e '${CONTROL_DIR}/state/heldout-running' && test ! -e '${CONTROL_DIR}/state/heldout-completed' && test ! -e '${CONTROL_DIR}/state/heldout-failed' && echo clean || echo collision")
  test "$heldout_state" = "clean" || return 1

  "${SSH_CMD[@]}" \
    "tmux new-session -d -s heldout-eval 'cd /workspace/toy-pickplace && exec bash ${CONTROL_DIR}/heldout-eval.sh'" \
    2>/dev/null || return 1

  for i in $(seq 1 30); do
    heldout_state=$(remote_value \
      "test -f '${CONTROL_DIR}/state/heldout-running' && echo running || { test -f '${CONTROL_DIR}/state/heldout-failed' && cat '${CONTROL_DIR}/state/heldout-failed' || echo starting; }")
    [ "$heldout_state" = "running" ] && return 0
    case "$heldout_state" in failed*) return 1 ;; esac
    heldout_alive=$(remote_value "tmux has-session -t heldout-eval 2>/dev/null && echo yes || echo no")
    [ "$heldout_alive" = "yes" ] || return 1
    sleep 1
  done
  return 1
}

EMERGENCY_CLEANUP_ARMED=false
trap on_supervisor_exit EXIT
trap 'handle_supervisor_signal 129' HUP
trap 'handle_supervisor_signal 130' INT
trap 'handle_supervisor_signal 143' TERM

write_status "supervisor-starting"
set -a
. "${PROJECT_ROOT}/.env" || { echo "ERROR: could not load ${PROJECT_ROOT}/.env" >&2; exit 1; }
set +a

echo "Supervisor started: instance=$INSTANCE_ID host=$HOST port=$PORT run=$RUN_NAME"
remote_probe || {
  echo "ERROR: cannot connect to instance on startup"
  upload_diagnostics || true
  cleanup "supervisor-startup-failed"
  exit 1
}

initial_state=$(remote_value \
  "test -d '${CONTROL_DIR}/state' && { test -f '${CONTROL_DIR}/state/run-status' || test -f '${CONTROL_DIR}/state/completed'; } && tmux has-session -t ckpt-bkp 2>/dev/null && echo SUPERVISOR_REMOTE_OK || echo INVALID")
test "$initial_state" = "SUPERVISOR_REMOTE_OK" || {
  echo "ERROR: remote workflow state is not ready for supervision"
  upload_diagnostics || true
  cleanup "supervisor-remote-state-invalid"
  exit 1
}

STATE="running"
write_status "supervisor-ready"

while true; do
  if ! remote_probe; then
    SSH_FAILURES=$((SSH_FAILURES + 1))
    echo "SSH probe failed ($SSH_FAILURES consecutive)"
    if [ "$SSH_FAILURES" -ge 3 ] && refresh_endpoint; then
      SSH_FAILURES=0
    elif [ "$SSH_FAILURES" -ge 3 ]; then
      instance_status=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | \
        uv run python -c "import json,sys; print(json.load(sys.stdin).get('actual_status','unknown'))" \
        2>/dev/null | tail -1)
      echo "Vast status: $instance_status"
      if echo "$instance_status" | grep -qE 'offline|stopped|error|terminated'; then
        upload_diagnostics || true
        cleanup "instance-terminated-${instance_status}"
        break
      fi
    fi
    write_status "supervisor-degraded-ssh"
    sleep 30
    continue
  fi
  SSH_FAILURES=0

  case "$STATE" in
    running)
      run_state=$(remote_lifecycle_state \
        "flywheel-run" \
        "${CONTROL_DIR}/state/completed" \
        "${CONTROL_DIR}/state/failed")
      backup_failed=$(remote_value \
        "test -f '${CONTROL_DIR}/state/backup-failed' && echo yes || echo no")
      backup_alive=$(remote_value \
        "tmux has-session -t ckpt-bkp 2>/dev/null && echo yes || echo no")

      case "$run_state" in
        failed*)
          echo "Training failed: $run_state"
          upload_diagnostics || true
          cleanup "training-failed-${run_state#failed }"
          break
          ;;
      esac
      if [ "$run_state" = "succeeded 0" ]; then
        echo "Training completed: $run_state"
        FINAL_BACKUP_TOKEN=$(date +%s%N)
        FINAL_BACKUP_DEADLINE=$(( $(date +%s) + 10800 ))
        "${SSH_CMD[@]}" \
          "printf '%s\n' '$FINAL_BACKUP_TOKEN' > '${CONTROL_DIR}/state/backup-final-requested.tmp' && mv '${CONTROL_DIR}/state/backup-final-requested.tmp' '${CONTROL_DIR}/state/backup-final-requested'" \
          2>/dev/null || {
            upload_diagnostics || true
            cleanup "final-backup-request-failed"
            break
          }
        STATE="completed-wait-backup"
      elif [ "$backup_failed" = "yes" ] || [ "$backup_alive" != "yes" ]; then
        echo "ERROR: checkpoint backup became unhealthy"
        upload_diagnostics || true
        cleanup "backup-failed"
        break
      elif [ "$run_state" = "disappeared" ]; then
        echo "ERROR: flywheel-run disappeared without a terminal marker"
        upload_diagnostics || true
        cleanup "runner-disappeared"
        break
      fi
      ;;

    completed-wait-backup)
      final_backup_succeeded=$(remote_value \
        "test -f '${CONTROL_DIR}/state/backup-final-succeeded' && cat '${CONTROL_DIR}/state/backup-final-succeeded' || echo not_found")
      backup_failed=$(remote_value \
        "test -f '${CONTROL_DIR}/state/backup-failed' && echo yes || echo no")
      backup_alive=$(remote_value \
        "tmux has-session -t ckpt-bkp 2>/dev/null && echo yes || echo no")

      if [ "$final_backup_succeeded" = "$FINAL_BACKUP_TOKEN" ]; then
        echo "Post-completion backup acknowledged. Launching held-out evaluation."
        if launch_heldout; then
          STATE="evaluation-running"
          EVALUATION_DEADLINE=$(( $(date +%s) + 43200 ))
        else
          upload_diagnostics || true
          cleanup "heldout-startup-failed"
          break
        fi
      elif [ "$backup_failed" = "yes" ] || [ "$backup_alive" != "yes" ] || \
           [ "$(date +%s)" -ge "$FINAL_BACKUP_DEADLINE" ]; then
        echo "ERROR: final backup was not acknowledged"
        upload_diagnostics || true
        cleanup "final-backup-failed"
        break
      fi
      ;;

    evaluation-running)
      heldout_state=$(remote_lifecycle_state \
        "heldout-eval" \
        "${CONTROL_DIR}/state/heldout-completed" \
        "${CONTROL_DIR}/state/heldout-failed")

      if [ "$heldout_state" = "succeeded 0" ]; then
        echo "Held-out evaluation completed successfully"
        set -a
        . "${PROJECT_ROOT}/.env"
        set +a
        verify_ok=true
        for key in "results/flywheel/${ARCH}/${RUN_NAME}/resolved-config.yaml" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/experiment-manifest.json" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/environment-manifest.json" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/metrics.json" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/final_scores.json" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/final-placement-score.png" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/final-score-curve.png"; do
          uv run python -c "
import boto3, os
s = boto3.Session(profile_name=os.environ.get('AWS_PROFILE') or None)
c = s.client('s3', region_name=os.environ.get('AWS_REGION') or 'ap-south-1')
r = c.head_object(Bucket=os.environ['S3_BUCKET'], Key='$key')
print(f'Verified: s3://{os.environ[\"S3_BUCKET\"]}/$key ({r[\"ContentLength\"]} bytes)')
" 2>/dev/null || { verify_ok=false; echo "FAILED to verify: $key"; }
        done
        if [ "$verify_ok" = true ]; then
          cleanup "evaluation-verified"
        else
          cleanup "evaluation-verification-failed"
        fi
        break
      fi

      case "$heldout_state" in
        failed*)
          echo "Held-out evaluation failed: $heldout_state"
          upload_diagnostics || true
          cleanup "heldout-failed"
          break
          ;;
      esac
      if [ "$heldout_state" = "disappeared" ]; then
        echo "ERROR: heldout-eval disappeared without a terminal marker"
        upload_diagnostics || true
        cleanup "heldout-disappeared"
        break
      fi
      if [ "$(date +%s)" -ge "$EVALUATION_DEADLINE" ]; then
        echo "ERROR: held-out evaluation exceeded its 12-hour deadline"
        upload_diagnostics || true
        cleanup "heldout-timeout"
        break
      fi
      ;;
  esac

  for local_session in "$LOCAL_SSH_SESSION" "$LOCAL_TB_SESSION"; do
    tmux has-session -t "$local_session" 2>/dev/null || \
      echo "WARNING: local wrapper $local_session is not running"
  done
  write_status "supervisor-running"
  sleep 30
done
