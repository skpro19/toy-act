#!/bin/bash
set -o pipefail

readonly CONTROL_DIR=/workspace/toy-act/.vast-train
readonly PROJECT_DIR=/workspace/toy-act
readonly LOG_FILE="$CONTROL_DIR/logs/training.log"
readonly TRAIN_MODULE="${TRAIN_MODULE:-scripts.train_v1}"

write_marker() {
  local path="$1"
  local value="$2"
  printf '%s\n' "$value" > "${path}.tmp"
  mv "${path}.tmp" "$path"
}

terminal_marker_written=false
runner_pid=""

mark_failed() {
  local exit_code="${1:-1}"
  if [ -f "${CONTROL_DIR}/state/completed" ]; then
    terminal_marker_written=true
    return
  fi
  if [ "$terminal_marker_written" != true ]; then
    write_marker "${CONTROL_DIR}/state/failed" "failed ${exit_code}"
    terminal_marker_written=true
  fi
}

stop_runner() {
  local exit_code="$1"
  if [ -n "$runner_pid" ]; then
    kill -TERM -- "-$runner_pid" 2>/dev/null || true
    wait "$runner_pid" 2>/dev/null || true
  fi
  mark_failed "$exit_code"
  exit "$exit_code"
}

on_exit() {
  local exit_code=$?
  if [ "$terminal_marker_written" != true ]; then
    test "$exit_code" -ne 0 || exit_code=1
    mark_failed "$exit_code"
  fi
}

trap on_exit EXIT
trap 'stop_runner 129' HUP
trap 'stop_runner 130' INT
trap 'stop_runner 143' TERM

test -d "${CONTROL_DIR}/logs" || { echo "ERROR: missing log directory" >&2; exit 1; }
test -d "${CONTROL_DIR}/state" || { echo "ERROR: missing state directory" >&2; exit 1; }
test ! -e "${CONTROL_DIR}/state/completed" || { echo "ERROR: completed marker already exists" >&2; exit 1; }
test ! -e "${CONTROL_DIR}/state/failed" || { echo "ERROR: failed marker already exists" >&2; exit 1; }
test ! -e "${CONTROL_DIR}/state/run-status" || { echo "ERROR: run-status marker already exists" >&2; exit 1; }

cd "$PROJECT_DIR" || exit 1
touch "$LOG_FILE" || exit 1

setsid bash -c '
  set -o pipefail
  /root/.local/bin/uv run --frozen --only-group train \
    python -m "$TRAIN_MODULE" 2>&1 | tee -a "$1"
' bash "$LOG_FILE" &
runner_pid=$!
write_marker "${CONTROL_DIR}/state/runner-pid" "$runner_pid"

# Do not publish readiness until the process group survives initialization.
sleep 2
if ! kill -0 "$runner_pid" 2>/dev/null; then
  wait "$runner_pid"
  exit_code=$?
  test "$exit_code" -ne 0 || exit_code=1
  mark_failed "$exit_code"
  exit "$exit_code"
fi
write_marker "${CONTROL_DIR}/state/run-status" "running"

wait "$runner_pid"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
  write_marker "${CONTROL_DIR}/state/completed" "succeeded 0"
  terminal_marker_written=true
else
  mark_failed "$exit_code"
fi

exit "$exit_code"
