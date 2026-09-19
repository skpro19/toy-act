#!/bin/bash
set -o pipefail

readonly CONTROL_DIR=/workspace/toy-act/.vast-train
readonly PROJECT_DIR=/workspace/toy-act
readonly CHECKPOINT_ROOT=checkpoints/act_v1
readonly RUNS_ROOT=runs/act_v1
readonly BACKUP_LOG="$CONTROL_DIR/logs/backup.log"

write_marker() {
  local path="$1"
  local value="$2"
  printf '%s\n' "$value" > "${path}.tmp"
  mv "${path}.tmp" "$path"
}

fail_backup() {
  local exit_code="${1:-1}"
  printf '{"timestamp":"%s","exit_code":%d}\n' \
    "$(date -Iseconds)" "$exit_code" > "${CONTROL_DIR}/state/backup-failed.tmp"
  mv "${CONTROL_DIR}/state/backup-failed.tmp" "${CONTROL_DIR}/state/backup-failed"
  exit "$exit_code"
}

test -d "${CONTROL_DIR}/state" || { echo "ERROR: missing state directory" >&2; exit 1; }
test -f "${CONTROL_DIR}/s3-env.env" || { echo "ERROR: missing S3 environment file" >&2; fail_backup 1; }
test ! -e "${CONTROL_DIR}/state/backup-failed" || { echo "ERROR: backup failure marker already exists" >&2; exit 1; }
cd "$PROJECT_DIR" || fail_backup 1

write_marker "${CONTROL_DIR}/state/backup-running" "$(date +%s%N)"

# Discover the single run directory once training has produced a real artifact.
# The directory is created at training start, but a file in it confirms that
# the run is live and worth synchronizing.
run_name=""
for _ in $(seq 1 120); do
  mapfile -t candidates < <(
    find "$CHECKPOINT_ROOT" "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d \
      -printf '%f\n' 2>/dev/null | sort -u
  )
  if [ "${#candidates[@]}" -eq 1 ]; then
    candidate="${candidates[0]}"
    if [ -n "$(find "$CHECKPOINT_ROOT/$candidate" "$RUNS_ROOT/$candidate" \
      -type f -print -quit 2>/dev/null)" ]; then
      run_name="$candidate"
      break
    fi
  fi
  sleep 5
done

test -n "$run_name" || {
  echo "ERROR: no run-specific training artifact appeared within 600 seconds" >&2
  fail_backup 1
}
write_marker "${CONTROL_DIR}/state/backup-artifact-ready" "$(date +%s%N)"
write_marker "${CONTROL_DIR}/state/run-name" "$run_name"

run_backup_cycle() {
  cycle_id=$(date +%s%N)
  write_marker "${CONTROL_DIR}/state/backup-cycle-started" "$cycle_id"

  retry_delay=30
  attempt=0
  exit_code=0
  while [ "$attempt" -lt 3 ]; do
    attempt=$((attempt + 1))
    write_marker "${CONTROL_DIR}/state/backup-heartbeat" \
      "cycle=${cycle_id} attempt=${attempt} timestamp=$(date -Iseconds)"
    timeout --signal=TERM --kill-after=30s 30m \
      /root/.local/bin/uv run --frozen --only-group train \
      --env-file "${CONTROL_DIR}/s3-env.env" \
      python scripts/s3_backup.py upload \
      --components checkpoints,runs "$run_name" \
      2>&1 | tee -a "$BACKUP_LOG"
    exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -eq 0 ]; then
      write_marker "${CONTROL_DIR}/state/backup-last-succeeded" "$cycle_id"
      return 0
    fi
    if [ "$attempt" -lt 3 ]; then
      echo "backup attempt ${attempt} failed (exit=${exit_code}), retrying in ${retry_delay}s"
      sleep "$retry_delay"
      retry_delay=$((retry_delay * 2))
    fi
  done

  printf 'FAILED cycle=%s exit=%d attempts=%d timestamp=%s\n' \
    "$cycle_id" "$exit_code" "$attempt" "$(date -Iseconds)" >> "$BACKUP_LOG"
  printf '{"timestamp":"%s","exit_code":%d,"attempts":%d}\n' \
    "$(date -Iseconds)" "$exit_code" "$attempt" \
    > "${CONTROL_DIR}/state/backup-failed.tmp"
  mv "${CONTROL_DIR}/state/backup-failed.tmp" "${CONTROL_DIR}/state/backup-failed"
  return "$exit_code"
}

while true; do
  run_backup_cycle || exit "$?"

  # A terminal marker is written only after the training process exits, but a
  # checkpoint can be written between this cycle's file listing and the marker.
  # Always sync once more after observing it so that checkpoint is not missed.
  if [ -e "${CONTROL_DIR}/state/completed" ] || [ -e "${CONTROL_DIR}/state/failed" ]; then
    run_backup_cycle || exit "$?"
    write_marker "${CONTROL_DIR}/state/backup-final-succeeded" "$cycle_id"
    exit 0
  fi

  sleep 120
done
