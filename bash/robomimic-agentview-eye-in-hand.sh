#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly INPUT="${INPUT:-${PROJECT_DIR}/datasets/can/ph/low_dim_v15.hdf5}"
readonly TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
readonly OUTPUT="${1:-${PROJECT_DIR}/datasets/can/ph/${TIMESTAMP}_agentview_robot0_eye_in_hand.hdf5}"
readonly LOG="${OUTPUT%.hdf5}.log"

echo "Input:  ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Log:    ${LOG}"

cd "$PROJECT_DIR"
uv run python scripts/extract_act_dataset.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --cameras agentview robot0_eye_in_hand \
  2>&1 | tee "$LOG"

echo "Done. Wrote ${OUTPUT}"
