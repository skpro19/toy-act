#!/bin/bash
set -o pipefail

readonly CONTROL_DIR=/workspace/toy-act/.vast-train
readonly PROJECT_DIR=/workspace/toy-act
readonly LOG_FILE="$CONTROL_DIR/logs/rollout.log"
readonly RUNS_FILE="$CONTROL_DIR/rollout-runs.txt"
readonly VERSION="${ROLLOUT_VERSION:-act_v2}"
readonly N_ROLLOUTS="${ROLLOUT_N_ROLLOUTS:-30}"
readonly HORIZON="${ROLLOUT_HORIZON:-250}"

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
test -f "$RUNS_FILE" || { echo "ERROR: missing run list $RUNS_FILE" >&2; exit 1; }
test -f "${CONTROL_DIR}/s3-env.env" || { echo "ERROR: missing S3 environment file" >&2; exit 1; }
grep -q '[^[:space:]]' "$RUNS_FILE" || { echo "ERROR: no run names in $RUNS_FILE" >&2; exit 1; }

cd "$PROJECT_DIR" || exit 1
touch "$LOG_FILE" || exit 1

export CONTROL_DIR PROJECT_DIR LOG_FILE RUNS_FILE VERSION N_ROLLOUTS HORIZON
export GIT_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || true)"

setsid bash -c '
  set -o pipefail
  args=()
  while IFS= read -r run_name; do
    test -n "$run_name" && args+=(--run-name "$run_name")
  done < "$RUNS_FILE"
  export MUJOCO_GL=egl
  /root/.local/bin/uv run --frozen --only-group train \
    --env-file "$CONTROL_DIR/s3-env.env" \
    python -m scripts.rollout_sweep_s3 \
      "${args[@]}" \
      --version "$VERSION" \
      --n-rollouts "$N_ROLLOUTS" \
      --horizon "$HORIZON" \
      --no-on-screen 2>&1 | tee -a "$LOG_FILE"
' &
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
