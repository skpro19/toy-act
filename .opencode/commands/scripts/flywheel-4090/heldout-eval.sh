#!/bin/bash
set -o pipefail
export MUJOCO_GL=egl
_R=__RUN_NAME__
_A=__ARCH__
_C=__CONTROL_DIR__

write_marker() {
  local path="$1"
  local value="$2"
  printf '%s\n' "$value" > "${path}.tmp"
  mv "${path}.tmp" "$path"
}

terminal_marker_written=false

fail_heldout() {
  local exit_code="${1:-1}"
  if [ -f "${_C}/state/heldout-completed" ]; then
    terminal_marker_written=true
    exit "$exit_code"
  fi
  if [ "$terminal_marker_written" != true ]; then
    write_marker "${_C}/state/heldout-failed" "failed ${exit_code}"
    terminal_marker_written=true
  fi
  exit "$exit_code"
}

on_exit() {
  local exit_code=$?
  if [ -f "${_C}/state/heldout-completed" ]; then
    terminal_marker_written=true
    return
  fi
  if [ "$terminal_marker_written" != true ]; then
    test "$exit_code" -ne 0 || exit_code=1
    write_marker "${_C}/state/heldout-failed" "failed ${exit_code}"
  fi
}

trap on_exit EXIT
trap 'fail_heldout 129' HUP
trap 'fail_heldout 130' INT
trap 'fail_heldout 143' TERM

if test -f "${_C}/state/heldout-completed" || \
   test -f "${_C}/state/heldout-failed" || \
   test -f "${_C}/state/heldout-running"; then
  echo "ERROR: held-out evaluation already started" >&2
  terminal_marker_written=true
  exit 1
fi

METRICS="/workspace/toy-pickplace/results/flywheel/${_A}/${_R}/metrics.json"
test -d "${_C}/logs" || { echo "ERROR: missing log directory" >&2; exit 1; }
test -f "${_C}/s3-env.env" || { echo "ERROR: missing S3 environment file" >&2; exit 1; }
if test ! -f "$METRICS"; then
  echo "ERROR: metrics.json not found at $METRICS" >&2
  fail_heldout 1
fi

cd /workspace/toy-pickplace || fail_heldout 1
write_marker "${_C}/state/heldout-running" "$(date +%s%N)"

/root/.local/bin/uv run --env-file "${_C}/s3-env.env" \
  python scripts/final_score.py --run-name "${_R}" --arch "${_A}" 2>&1 | \
  tee -a "${_C}/logs/heldout-eval-${_R}.log"
exit_code=${PIPESTATUS[0]}

if [ "$exit_code" -ne 0 ]; then
  fail_heldout "$exit_code"
fi

for f in "results/flywheel/${_A}/${_R}/final_scores.json" \
         "results/flywheel/${_A}/${_R}/final-placement-score.png" \
         "results/flywheel/${_A}/${_R}/final-score-curve.png"; do
  if test ! -f "/workspace/toy-pickplace/$f"; then
    echo "ERROR: missing output $f" >&2
    fail_heldout 1
  fi
done

upload_and_verify() {
  local src="$1" key="$2"
  /root/.local/bin/uv run --env-file "${_C}/s3-env.env" python -c "
import boto3, hashlib, os, sys
from pathlib import Path
cfg = {}
with open('${_C}/s3-env.env') as f:
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k] = v
c = boto3.client('s3', region_name=cfg.get('AWS_REGION', ''))
digest = hashlib.sha256(Path('$src').read_bytes()).hexdigest()
c.upload_file(
    '$src',
    cfg['S3_BUCKET'],
    '$key',
    ExtraArgs={'Metadata': {'sha256': digest}},
)
r = c.head_object(Bucket=cfg['S3_BUCKET'], Key='$key')
sz = r['ContentLength']
local_size = os.path.getsize('$src')
if sz != local_size:
    print(f'size mismatch: {sz} vs {local_size}', file=sys.stderr)
    sys.exit(1)
if r.get('Metadata', {}).get('sha256') != digest:
    print('sha256 metadata mismatch', file=sys.stderr)
    sys.exit(1)
remote = c.get_object(Bucket=cfg['S3_BUCKET'], Key='$key')['Body'].read()
if hashlib.sha256(remote).hexdigest() != digest:
    print('remote sha256 mismatch', file=sys.stderr)
    sys.exit(1)
bucket = cfg['S3_BUCKET']
print(f'Verified: s3://{bucket}/$key ({sz} bytes)')
" || { echo "upload failed for $2" >&2; return 1; }
}

upload_and_verify \
  "$METRICS" \
  "results/flywheel/${_A}/${_R}/metrics.json" || fail_heldout 1
upload_and_verify \
  "/workspace/toy-pickplace/results/flywheel/${_A}/${_R}/final_scores.json" \
  "results/flywheel/${_A}/${_R}/final_scores.json" || fail_heldout 1
upload_and_verify \
  "/workspace/toy-pickplace/results/flywheel/${_A}/${_R}/final-placement-score.png" \
  "results/flywheel/${_A}/${_R}/final-placement-score.png" || fail_heldout 1
upload_and_verify \
  "/workspace/toy-pickplace/results/flywheel/${_A}/${_R}/final-score-curve.png" \
  "results/flywheel/${_A}/${_R}/final-score-curve.png" || fail_heldout 1

write_marker "${_C}/state/heldout-completed" "succeeded 0"
terminal_marker_written=true
exit 0
