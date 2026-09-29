#!/bin/bash
# Detached local watcher for a vast-train run.
#
# Usage:
#   local-watcher.sh [RUN_DIR]
#
# RUN_DIR is a `.vast-train-local/toy-act-<INSTANCE_ID>/` directory containing
# `setup.env`. When omitted it defaults to the directory holding this script, so
# the intended use is to copy this file to `$RUN_DIR/watcher.sh` and launch:
#
#   setsid nohup bash "$RUN_DIR/watcher.sh" >"$RUN_DIR/watcher.out" 2>&1 </dev/null &
#
# The watcher owns the cleanup decision for exactly one instance. It reads the
# remote run state over SSH and may only destroy the instance after the cleanup
# gate is open: before `run-status=running` (setup phase), or after a directly
# observed remote `state/completed` / `state/failed` marker. Connectivity
# failures never open the gate; they keep the watcher in its retry loop.
#
# Environment overrides (all optional):
#   REPO, UV, POLL_SECONDS, WATCHER_DRY_RUN, S3_BUCKET,
#   S3_CHECKPOINT_BASE, S3_RUNS_BASE, AWS_PROFILE, AWS_REGION, VAST_API_KEY

set -o pipefail

RUN_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SETUP_ENV="$RUN_DIR/setup.env"
test -f "$SETUP_ENV" || { echo "ERROR: missing $SETUP_ENV" >&2; exit 1; }

# shellcheck disable=SC1090
set -a
source "$SETUP_ENV"
set +a

REPO="${REPO:-$(cd "$RUN_DIR/../.." && pwd)}"
UV="${UV:-$(command -v uv || true)}"
test -n "$UV" || { echo "ERROR: uv not found; set UV" >&2; exit 1; }

POLL_SECONDS="${POLL_SECONDS:-30}"
WATCHER_DRY_RUN="${WATCHER_DRY_RUN:-no}"

S3_BUCKET="${S3_BUCKET:-toy-act}"
S3_CHECKPOINT_BASE="${S3_CHECKPOINT_BASE:-checkpoints/act_v2}"
S3_RUNS_BASE="${S3_RUNS_BASE:-runs/act_v2}"
AWS_PROFILE="${AWS_PROFILE:-toy-pickplace-backup}"
AWS_REGION="${AWS_REGION:-ap-south-1}"

REMOTE_PROJECT="${REMOTE_PROJECT:-/workspace/toy-act}"
REMOTE_STATE="${REMOTE_STATE:-$REMOTE_PROJECT/.vast-train/state}"
REMOTE_LOG="${REMOTE_LOG:-$REMOTE_PROJECT/.vast-train/logs/training.log}"
TRAIN_SESSION="${TRAIN_SESSION:-train}"

SSH_SESSION="${SSH_SESSION:-}"
TB_SESSION="${TB_SESSION:-}"
case "$SSH_KNOWN_HOSTS" in
  "") KNOWN_HOSTS="$RUN_DIR/known_hosts" ;;
  /*) KNOWN_HOSTS="$SSH_KNOWN_HOSTS" ;;
  *) KNOWN_HOSTS="$REPO/$SSH_KNOWN_HOSTS" ;;
esac

LOG="${WATCHER_LOG:-$RUN_DIR/watcher.log}"
REPORT="${REPORT:-$RUN_DIR/report.txt}"
PID_FILE="${WATCHER_PID:-$RUN_DIR/watcher.pid}"
OWNER_FILE="${LOCAL_OWNER_FILE:-}"
LOCK_FILE=/tmp/toy-act-local-wrapper.lock

S3_CKPT_PREFIX="s3://${S3_BUCKET}/${S3_CHECKPOINT_BASE}"
S3_RUNS_PREFIX="s3://${S3_BUCKET}/${S3_RUNS_BASE}"

BACKUP_FINAL_TIMEOUT_SECONDS="${BACKUP_FINAL_TIMEOUT_SECONDS:-900}"
BACKUP_FINAL_SHORT_SECONDS="${BACKUP_FINAL_SHORT_SECONDS:-300}"

for required in INSTANCE_ID INSTANCE_LABEL SSH_HOST SSH_PORT CHECKPOINT_EVERY STEPS; do
  eval "value=\${$required:-}"
  test -n "$value" || { echo "ERROR: $required missing from $SETUP_ENV" >&2; exit 1; }
done

if [ -z "${VAST_API_KEY:-}" ]; then
  VAST_API_KEY="$(cat "$HOME/.config/vastai/vast_api_key" 2>/dev/null || true)"
fi
if [ -z "${VAST_API_KEY:-}" ] && [ -f "$REPO/.env" ]; then
  VAST_API_KEY="$(sed -n 's/^VAST_API_KEY=//p' "$REPO/.env" | head -n 1)"
fi
export VAST_API_KEY AWS_PROFILE AWS_REGION

SSH_OPTS=(
  -o "UserKnownHostsFile=$KNOWN_HOSTS"
  -o StrictHostKeyChecking=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)

OUTCOME=unknown
FAILURE_REASON=""
RUN_NAME="${RUN_NAME:-}"
S3_VERIFY=not_checked
TB_VERIFY=not_checked
CLEANUP_STATUS=not_run
FINISHED=no
RUN_STARTED=no
TERMINAL_CONFIRMED=no
START_EPOCH=$(date +%s)

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >>"$LOG"
}

ssh_remote() {
  ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "root@$SSH_HOST" "$@"
}

# Diagnostic only: exists/absent/unknown. Never authorizes cleanup by itself.
instance_status() {
  local raw rc
  raw=$(vastai show instances --raw 2>/dev/null)
  rc=$?
  if [ "$rc" -ne 0 ] || [ -z "$raw" ]; then
    echo unknown
    return
  fi
  if ! printf '%s' "$raw" | jq -e 'type=="array"' >/dev/null 2>&1; then
    echo unknown
    return
  fi
  if printf '%s' "$raw" | jq -e --argjson id "$INSTANCE_ID" 'any(.[]; .id == $id)' >/dev/null 2>&1; then
    echo exists
  else
    echo absent
  fi
}

probe() {
  ssh_remote "cd $REMOTE_STATE 2>/dev/null || exit 3
    if tmux has-session -t $TRAIN_SESSION 2>/dev/null; then echo TRAIN_SESSION=yes; else echo TRAIN_SESSION=no; fi
    if [ -e completed ]; then echo COMPLETED=yes; else echo COMPLETED=no; fi
    if [ -e failed ]; then echo FAILED=yes; else echo FAILED=no; fi
    if [ -e run-status ]; then printf 'RUN_STATUS=%s\n' \"\$(cat run-status)\"; fi
    if [ -e run-name ]; then printf 'RUN_NAME=%s\n' \"\$(cat run-name)\"; fi
    line=\$(tr '\r' '\n' < $REMOTE_LOG 2>/dev/null | grep -v '^$' | tail -n 1)
    printf 'LAST_LOG=%s\n' \"\$line\"" 2>/dev/null
}

wait_for_backup_final() {
  local budget="$1"
  local waited=0
  while [ "$waited" -lt "$budget" ]; do
    if ssh_remote "test -e $REMOTE_STATE/backup-final-succeeded" 2>/dev/null; then
      return 0
    fi
    sleep 15
    waited=$((waited + 15))
  done
  return 1
}

expected_snapshots() {
  local step
  for step in $(seq "$CHECKPOINT_EVERY" "$CHECKPOINT_EVERY" "$STEPS"); do
    printf 'step_%09d.pt\n' "$step"
  done
  if [ $((STEPS % CHECKPOINT_EVERY)) -ne 0 ]; then
    printf 'step_%09d.pt\n' "$STEPS"
  fi
}

verify_s3() {
  local run="$1"
  mapfile -t files < <(expected_snapshots)
  local expected="${#files[@]}"
  local ckpt_ok=no runs_ok=no

  log "verifying $expected periodic snapshots in $S3_CKPT_PREFIX/$run/ via s3_backup.py has-files"
  if (
    cd "$REPO" || exit 1
    S3_BUCKET="$S3_BUCKET" AWS_REGION="$AWS_REGION" \
      S3_CHECKPOINT_BASE="$S3_CHECKPOINT_BASE" AWS_PROFILE="$AWS_PROFILE" \
      "$UV" run --frozen --only-group train python scripts/s3_backup.py \
      has-files --components checkpoints "$run" "${files[@]}"
  ) >>"$LOG" 2>&1; then
    ckpt_ok=yes
  fi

  log "verifying config.json in $S3_RUNS_PREFIX/$run/ via s3_backup.py has-files"
  if (
    cd "$REPO" || exit 1
    S3_BUCKET="$S3_BUCKET" AWS_REGION="$AWS_REGION" \
      S3_RUNS_BASE="$S3_RUNS_BASE" AWS_PROFILE="$AWS_PROFILE" \
      "$UV" run --frozen --only-group train python scripts/s3_backup.py \
      has-files --components runs "$run" config.json
  ) >>"$LOG" 2>&1; then
    runs_ok=yes
  fi

  if [ "$ckpt_ok" = yes ] && [ "$runs_ok" = yes ]; then
    S3_VERIFY="verified $expected periodic snapshots and config.json via s3_backup.py has-files"
    return 0
  fi
  if [ "$ckpt_ok" != yes ]; then
    S3_VERIFY="failed: s3_backup.py has-files reported missing checkpoints"
  else
    S3_VERIFY="failed: s3_backup.py has-files reported missing config.json"
  fi
  return 1
}

# Verify the uploaded TensorBoard event files carry the final evaluation and
# throughput scalars. Best-effort: never changes the run outcome.
verify_tb_scalars() {
  local run="$1"
  local tmp result rc
  tmp="$(mktemp -d)"
  log "verifying TensorBoard scalars under $S3_RUNS_PREFIX/$run/"

  if ! aws s3 cp --recursive --only-show-errors \
      --exclude '*' --include '*tfevents*' --region "$AWS_REGION" \
      "$S3_RUNS_PREFIX/$run/" "$tmp" >>"$LOG" 2>&1; then
    TB_VERIFY="failed: could not download event files"
    rm -rf "$tmp"
    return 1
  fi

  cat >"$tmp/verify_tb.py" <<'PY'
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REQUIRED_TAGS = (
    "eval/success_rate",
    "throughput/train_steps_per_sec",
    "throughput/train_samples_per_sec",
    "throughput/rollout_env_steps_per_sec",
    "throughput/rollout_episodes_per_min",
)

counts: dict[str, int] = {}
for path in glob.glob(os.path.join(sys.argv[1], "**", "*tfevents*"), recursive=True):
    accumulator = EventAccumulator(path, size_guidance={"scalars": 0})
    accumulator.Reload()
    for tag in accumulator.Tags().get("scalars", []):
        counts[tag] = counts.get(tag, 0) + len(accumulator.Scalars(tag))

missing = [tag for tag in REQUIRED_TAGS if counts.get(tag, 0) == 0]
if missing:
    print("missing: " + ", ".join(missing))
    sys.exit(1)
print(", ".join(f"{tag}={counts[tag]}" for tag in REQUIRED_TAGS))
PY

  result="$( cd "$REPO" && "$UV" run --frozen --only-group train \
    python "$tmp/verify_tb.py" "$tmp" 2>>"$LOG" )"
  rc=$?
  rm -rf "$tmp"

  if [ "$rc" -eq 0 ]; then
    TB_VERIFY="verified: $result"
    return 0
  fi
  TB_VERIFY="failed: missing TensorBoard scalars ($result)"
  return 1
}

cleanup() {
  if [ "$RUN_STARTED" != yes ]; then
    log "cleanup gate open: setup phase (RUN_STARTED=no); destroying"
  elif [ "$TERMINAL_CONFIRMED" = yes ]; then
    log "cleanup gate open: terminal marker confirmed; destroying"
  else
    log "cleanup refused: RUN_STARTED=yes and TERMINAL_CONFIRMED=no; leaving instance untouched"
    CLEANUP_STATUS=refused_gate_closed
    return 0
  fi

  if [ "$WATCHER_DRY_RUN" = yes ]; then
    log "cleanup dry run: would destroy instance $INSTANCE_ID ($INSTANCE_LABEL)"
    CLEANUP_STATUS=dry_run
    return 0
  fi

  log "cleanup: destroying instance $INSTANCE_ID ($INSTANCE_LABEL)"
  if vastai destroy instance "$INSTANCE_ID" -y >>"$LOG" 2>&1; then
    log "destroy command accepted"
  else
    log "destroy command returned non-zero"
  fi

  local status gone=no i
  for i in $(seq 1 24); do
    status=$(instance_status)
    if [ "$status" = absent ]; then
      gone=yes
      break
    fi
    if [ "$status" = unknown ]; then
      log "cleanup: instance query unknown (attempt $i)"
    fi
    sleep 5
  done
  if [ "$gone" = yes ]; then
    CLEANUP_STATUS="destroyed_and_verified"
  else
    CLEANUP_STATUS="destroy_unverified"
  fi

  [ -n "$SSH_SESSION" ] && tmux kill-session -t "$SSH_SESSION" 2>/dev/null || true
  [ -n "$TB_SESSION" ] && tmux kill-session -t "$TB_SESSION" 2>/dev/null || true
  if [ -n "$OWNER_FILE" ]; then
    exec 9>"$LOCK_FILE" && flock 9 && rm -f "$OWNER_FILE"
  fi
  log "cleanup result: $CLEANUP_STATUS"
}

write_report() {
  local elapsed=$(( $(date +%s) - START_EPOCH ))
  {
    echo "toy-act vast-train report"
    echo "generated: $(date -Is)"
    echo "instance_id: $INSTANCE_ID"
    echo "instance_label: $INSTANCE_LABEL"
    echo "outcome: $OUTCOME"
    echo "failure_reason: ${FAILURE_REASON:-none}"
    echo "run_name: ${RUN_NAME:-unknown}"
    echo "s3_uri: $S3_CKPT_PREFIX/${RUN_NAME:-unknown}/"
    echo "s3_runs_uri: $S3_RUNS_PREFIX/${RUN_NAME:-unknown}/"
    echo "elapsed_seconds: $elapsed"
    echo "offer_price_usd_per_hour: ${OFFER_DPH_TOTAL:-unknown}"
    echo "actual_dph_usd_per_hour: ${ACTUAL_DPH_TOTAL:-unknown}"
    echo "pinned_commit: ${GIT_COMMIT:-unknown}"
    echo "tensorboard_url: ${TB_URL:-unknown}"
    echo "s3_verification: $S3_VERIFY"
    echo "tensorboard_verification: $TB_VERIFY"
    echo "final_cleanup_status: $CLEANUP_STATUS"
  } >"$REPORT"
}

on_exit() {
  local code=$?
  if [ "$FINISHED" = yes ]; then
    exit "$code"
  fi
  FINISHED=yes

  if [ "$OUTCOME" = success ]; then
    local remote_rn
    remote_rn=$(ssh_remote "cat $REMOTE_STATE/run-name 2>/dev/null" 2>/dev/null)
    if [ -n "$remote_rn" ]; then
      RUN_NAME="$remote_rn"
    fi
    if wait_for_backup_final "$BACKUP_FINAL_TIMEOUT_SECONDS"; then
      log "observed backup-final-succeeded"
      if [ -n "$RUN_NAME" ]; then
        verify_s3 "$RUN_NAME" || true
        verify_tb_scalars "$RUN_NAME" || true
      else
        S3_VERIFY="skipped_no_run_name"
        TB_VERIFY="skipped_no_run_name"
      fi
    else
      log "backup-final-succeeded not observed within timeout"
      S3_VERIFY="not_verified_backup_final_timeout"
      TB_VERIFY="not_verified_backup_final_timeout"
    fi
  elif [ "$OUTCOME" = failure ]; then
    # Preserve the full terminal traceback even when the final S3 sync fails.
    if ssh_remote "tail -c 2097152 $REMOTE_LOG" \
        > "$RUN_DIR/training-log-tail.txt" 2>>"$LOG"; then
      log "saved terminal training log to $RUN_DIR/training-log-tail.txt"
    else
      rm -f "$RUN_DIR/training-log-tail.txt"
      log "could not retrieve terminal training log"
    fi
    if ssh_remote "cat $REMOTE_PROJECT/.vast-train/logs/failure-resources.txt" \
        > "$RUN_DIR/failure-resources.txt" 2>>"$LOG"; then
      log "saved failure resource snapshot to $RUN_DIR/failure-resources.txt"
    else
      rm -f "$RUN_DIR/failure-resources.txt"
    fi
    if wait_for_backup_final "$BACKUP_FINAL_SHORT_SECONDS"; then
      log "observed best-effort backup-final-succeeded"
    else
      log "best-effort final sync not observed"
    fi
  fi

  cleanup
  write_report
  log "finished outcome=$OUTCOME cleanup=$CLEANUP_STATUS s3=$S3_VERIFY tb=$TB_VERIFY report=$REPORT"
  exit "$code"
}

trap on_exit EXIT
trap 'exit 130' INT TERM HUP

printf '%s\n' "$$" >"$PID_FILE"

if [ -z "$VAST_API_KEY" ]; then
  log "FATAL: VAST_API_KEY is empty; cannot own cleanup"
  OUTCOME=failure
  FAILURE_REASON="VAST_API_KEY missing"
  exit 1
fi

log "watcher start pid=$$ instance=$INSTANCE_ID label=$INSTANCE_LABEL host=$SSH_HOST port=$SSH_PORT commit=${GIT_COMMIT:-unknown}"
log "config steps=$STEPS checkpoint_every=$CHECKPOINT_EVERY tb=${TB_URL:-unknown} run_dir=$RUN_DIR"

PREV_LOG=""
while true; do
  OUTPUT=$(probe)
  PROBE_RC=$?
  if [ "$PROBE_RC" -ne 0 ]; then
    log "ssh probe failed (rc=$PROBE_RC); retrying in ${POLL_SECONDS}s"
    if [ "$RUN_STARTED" != yes ]; then
      status=$(instance_status)
      if [ "$status" = absent ]; then
        OUTCOME=failure
        FAILURE_REASON="instance absent during setup"
        break
      fi
      if [ "$status" = unknown ]; then
        log "instance query unknown during setup; continuing"
      fi
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  TRAIN_SESSION_STATE=$(printf '%s\n' "$OUTPUT" | sed -n 's/^TRAIN_SESSION=//p')
  COMPLETED=$(printf '%s\n' "$OUTPUT" | sed -n 's/^COMPLETED=//p')
  FAILED=$(printf '%s\n' "$OUTPUT" | sed -n 's/^FAILED=//p')
  RUN_STATUS=$(printf '%s\n' "$OUTPUT" | sed -n 's/^RUN_STATUS=//p')
  RN=$(printf '%s\n' "$OUTPUT" | sed -n 's/^RUN_NAME=//p')
  LAST=$(printf '%s\n' "$OUTPUT" | sed -n 's/^LAST_LOG=//p')

  if [ -n "$RN" ]; then
    RUN_NAME="$RN"
  fi
  if [ -n "$LAST" ] && [ "$LAST" != "$PREV_LOG" ]; then
    log "progress: $LAST"
    PREV_LOG="$LAST"
  fi

  if [ "$FAILED" = yes ]; then
    TERMINAL_CONFIRMED=yes
    OUTCOME=failure
    FAILURE_REASON="state/failed marker present"
    break
  fi
  if [ "$COMPLETED" = yes ]; then
    TERMINAL_CONFIRMED=yes
    OUTCOME=success
    break
  fi

  if [ "$RUN_STARTED" != yes ] && [ "$RUN_STATUS" = running ]; then
    RUN_STARTED=yes
    log "run-status=running observed; cleanup gate armed (requires terminal marker)"
  fi
  if [ "$RUN_STARTED" = yes ] && [ "$TRAIN_SESSION_STATE" != yes ]; then
    log "inconsistency: train session missing without terminal marker; continuing to poll"
  fi

  sleep "$POLL_SECONDS"
done

log "training loop ended outcome=$OUTCOME reason=${FAILURE_REASON:-none} run=$RUN_NAME"
if [ "$OUTCOME" = success ]; then
  exit 0
fi
exit 1
