#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR=/workspace/toy-act
readonly CHECKPOINT_ROOT=checkpoints/act_v1
readonly S3_ROOT=s3://toy-act/checkpoints/act_v1
readonly STATUS_FILE="$PROJECT_DIR/training.status"

cd "$PROJECT_DIR"
set -a
source "$PROJECT_DIR/.aws.env"
set +a

upload_checkpoint() {
  local path=$1
  local relative_path=${path#"$CHECKPOINT_ROOT"/}

  for attempt in 1 2 3; do
    if /root/.local/bin/aws s3 cp \
      "$path" "$S3_ROOT/$relative_path" \
      --region ap-south-1 --only-show-errors; then
      printf 'uploaded checkpoint => %s/%s\n' "$S3_ROOT" "$relative_path"
      return 0
    fi
    sleep $((attempt * 5))
  done

  printf 'failed to upload checkpoint after 3 attempts: %s\n' "$path" >&2
  return 1
}

watch_checkpoints() {
  inotifywait --monitor --recursive --event close_write \
    --format '%w%f' "$CHECKPOINT_ROOT" 2>/dev/null |
    while IFS= read -r path; do
      [[ $path == *.pt ]] || continue
      upload_checkpoint "$path"
    done
}

finalize() {
  local exit_status=$?

  if [[ -n ${watcher_pid:-} ]]; then
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
  fi

  /root/.local/bin/aws s3 sync \
    "$CHECKPOINT_ROOT" "$S3_ROOT" \
    --region ap-south-1 --only-show-errors || exit_status=1

  if [[ $exit_status -eq 0 ]]; then
    printf 'succeeded\n' > "$STATUS_FILE"
  else
    printf 'failed:%s\n' "$exit_status" > "$STATUS_FILE"
  fi
}
trap finalize EXIT

mkdir -p "$CHECKPOINT_ROOT"
printf 'running\n' > "$STATUS_FILE"
watch_checkpoints &
watcher_pid=$!

/root/.local/bin/uv run --frozen --only-group train \
  python -m scripts.train 2>&1 | tee "$PROJECT_DIR/training.log"
