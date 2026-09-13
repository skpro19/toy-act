#!/bin/bash
set -o pipefail
_R=__RUN_NAME__
_A=__ARCH__
_C=__CONTROL_DIR__

write_marker() {
  local path="$1"
  local value="$2"
  printf '%s\n' "$value" > "${path}.tmp"
  mv "${path}.tmp" "$path"
}

fail_backup() {
  local exit_code="${1:-1}"
  printf '{"timestamp":"%s","exit_code":%d}\n' \
    "$(date -Iseconds)" "$exit_code" > "${_C}/state/backup-failed.tmp"
  mv "${_C}/state/backup-failed.tmp" "${_C}/state/backup-failed"
  exit "$exit_code"
}

test -d "${_C}/state" || { echo "ERROR: missing state directory" >&2; exit 1; }
test -f "${_C}/s3-env.env" || { echo "ERROR: missing S3 environment file" >&2; fail_backup 1; }
test ! -e "${_C}/state/backup-failed" || { echo "ERROR: backup failure marker already exists" >&2; exit 1; }
cd /workspace/toy-pickplace || fail_backup 1
BACKUP_LOG="${_C}/logs/backup-${_R}.log"

write_marker "${_C}/state/backup-running" "$(date +%s%N)"

# Provenance exists before training starts, so require a training-produced file.
artifact_ready=false
for i in $(seq 1 120); do
  for d in checkpoints/flywheel/${_A}/${_R} \
           runs/flywheel/${_A}/${_R} \
           data/flywheel/${_A}/${_R}; do
    if test -d "$d" && test -n "$(find "$d" -type f -print -quit 2>/dev/null)"; then
      artifact_ready=true
      break 2
    fi
  done
  sleep 5
done

test "$artifact_ready" = true || {
  echo "ERROR: no run-specific training artifact appeared within 600 seconds" >&2
  fail_backup 1
}
write_marker "${_C}/state/backup-artifact-ready" "$(date +%s%N)"

while true; do
  cycle_id=$(date +%s%N)
  write_marker "${_C}/state/backup-cycle-started" "$cycle_id"

  final_token=""
  if [ -f "${_C}/state/backup-final-requested" ]; then
    final_token=$(cat "${_C}/state/backup-final-requested" 2>/dev/null || true)
  fi

  retry_delay=30
  attempt=0
  while [ "$attempt" -lt 3 ]; do
    attempt=$((attempt + 1))
    write_marker "${_C}/state/backup-heartbeat" \
      "cycle=${cycle_id} attempt=${attempt} timestamp=$(date -Iseconds)"
    timeout --signal=TERM --kill-after=30s 30m \
      /root/.local/bin/uv run --env-file "${_C}/s3-env.env" \
      python scripts/s3_backup.py --arch "${_A}" upload \
      --components checkpoints,runs,results,dagger "${_R}" \
      2>&1 | tee -a "$BACKUP_LOG"
    exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -eq 0 ]; then
      write_marker "${_C}/state/backup-last-succeeded" "$cycle_id"
      if [ -n "$final_token" ]; then
        write_marker "${_C}/state/backup-final-succeeded" "$final_token"
      fi
      break
    fi
    if [ "$attempt" -lt 3 ]; then
      echo "backup attempt ${attempt} failed (exit=${exit_code}), retrying in ${retry_delay}s"
      sleep "$retry_delay"
      retry_delay=$((retry_delay * 2))
    fi
  done

  if [ "$exit_code" -ne 0 ]; then
    printf 'FAILED cycle=%s exit=%d attempts=%d timestamp=%s\n' \
      "$cycle_id" "$exit_code" "$attempt" "$(date -Iseconds)" >> "$BACKUP_LOG"
    printf '{"timestamp":"%s","exit_code":%d,"attempts":%d}\n' \
      "$(date -Iseconds)" "$exit_code" "$attempt" \
      > "${_C}/state/backup-failed.tmp"
    mv "${_C}/state/backup-failed.tmp" "${_C}/state/backup-failed"
    exit 1
  fi
  sleep 120
done
