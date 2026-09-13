#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${CONTROL_DIR:?CONTROL_DIR is required}"
: "${EXPERT_EPISODES:?EXPERT_EPISODES is required}"
: "${EXPERT_SEED:?EXPERT_SEED is required}"
: "${EXPERT_MAX_STEPS:?EXPERT_MAX_STEPS is required}"
: "${TRAIN_CAPTURE_HZ:?TRAIN_CAPTURE_HZ is required}"
GENERATION=${GENERATION:-data-1}
ATTEMPT=${ATTEMPT:-1}
cd "$PROJECT_ROOT"

status_path="$CONTROL_DIR/status/expert.status"
temporary_status="$status_path.tmp.$$"
printf 'running %s %s\n' "$GENERATION" "$ATTEMPT" >"$temporary_status"
mv -f "$temporary_status" "$status_path"

set +e
MUJOCO_GL=egl /usr/bin/time -v -o "$CONTROL_DIR/time/expert-a$ATTEMPT.txt" \
  /root/.local/bin/uv run python scripts/data.py \
    --episodes "$EXPERT_EPISODES" \
    --out-dir "$CONTROL_DIR/data/expert" \
    --seed "$EXPERT_SEED" \
    --max-steps "$EXPERT_MAX_STEPS" \
    --capture-hz "$TRAIN_CAPTURE_HZ" \
    --save-images \
  >"$CONTROL_DIR/logs/expert-a$ATTEMPT.log" 2>&1
command_status=$?
set -e

if [[ "$command_status" -eq 0 ]]; then
  file_count=$(find "$CONTROL_DIR/data/expert" -maxdepth 1 -type f -name '*.npz' | wc -l)
  if [[ "$file_count" -ne "$EXPERT_EPISODES" ]]; then
    command_status=125
  fi
fi

if [[ "$command_status" -eq 0 ]]; then
  status_text="succeeded 0 $GENERATION $ATTEMPT"
else
  status_text="failed $command_status $GENERATION $ATTEMPT"
fi
temporary_status="$status_path.tmp.$$"
printf '%s\n' "$status_text" >"$temporary_status"
mv -f "$temporary_status" "$status_path"
exit "$command_status"
