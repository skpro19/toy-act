#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# Extract ACT training datasets from the robomimic CAN multi-human (MH) low-dim data,
# one file per operator-quality subset and camera configuration.
#
# Prerequisites (run once):
#   bash/robomimic-download-mh.sh                 # fetch the raw + low-dim MH data
#   uv run python scripts/utils/add_quality_filter_keys.py
#
# Usage:
#   bash/robomimic-quality-subsets.sh <quality> [cameras...]
#
# Examples:
#   bash/robomimic-quality-subsets.sh better agentview
#   bash/robomimic-quality-subsets.sh better_okay agentview robot0_eye_in_hand
#   bash/robomimic-quality-subsets.sh all
#
# QUALITIES: better, okay, worse, better_okay, better_okay_worse, all
# CAMERAS:   defaults to both "agentview" and "agentview robot0_eye_in_hand"
#
# Output names follow the existing convention:
#   <timestamp>_<quality>_<camera-list>.hdf5

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly INPUT="${INPUT:-${PROJECT_DIR}/datasets/can/mh/low_dim_v15.hdf5}"

readonly ALL_CAMERA_SETS=("agentview" "agentview robot0_eye_in_hand")

usage() {
  sed -n '5,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-1}"
}

# Map a quality name to the filter key used inside the hdf5.
quality_to_filter_key() {
  case "$1" in
    better) echo "better" ;;
    okay) echo "okay" ;;
    worse) echo "worse" ;;
    better_okay) echo "better_okay" ;;
    better_okay_worse) echo "better_okay_worse" ;;
    *) return 1 ;;
  esac
}

expand_qualities() {
  case "$1" in
    all) echo "better okay worse better_okay better_okay_worse" ;;
    better | okay | worse | better_okay | better_okay_worse) echo "$1" ;;
    *) return 1 ;;
  esac
}

if [ "$#" -lt 1 ]; then
  usage 1
fi

case "$1" in
  -h | --help) usage 0 ;;
esac

readonly QUALITY_ARG="$1"
shift

if ! expand_qualities "$QUALITY_ARG" >/dev/null; then
  echo "ERROR: unknown quality '${QUALITY_ARG}'" >&2
  echo "" >&2
  usage 1
fi

readonly QUALITIES="$(expand_qualities "$QUALITY_ARG")"

if [ "$#" -gt 0 ]; then
  CAMERA_SETS=("$*")
else
  CAMERA_SETS=("${ALL_CAMERA_SETS[@]}")
fi

if [ ! -f "$INPUT" ]; then
  echo "ERROR: input dataset not found: ${INPUT}" >&2
  echo "       run bash/robomimic-download-mh.sh first, or set INPUT=..." >&2
  exit 1
fi

echo "Input:  ${INPUT}"
echo ""

for quality in $QUALITIES; do
  filter_key="$(quality_to_filter_key "$quality")"
  for cameras in "${CAMERA_SETS[@]}"; do
    timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
    camera_list="${cameras// /_}"
    output="${PROJECT_DIR}/datasets/can/mh/${timestamp}_${quality}_${camera_list}.hdf5"
    log="${output%.hdf5}.log"

    echo "Quality:  ${quality} (mask/${filter_key})"
    echo "Cameras:  ${cameras}"
    echo "Output:   ${output}"
    echo "Log:      ${log}"

    uv run --project "$PROJECT_DIR" python "$PROJECT_DIR/scripts/extract_act_dataset.py" \
      --input "$INPUT" \
      --output "$output" \
      --filter-key "$filter_key" \
      --cameras $cameras \
      2>&1 | tee "$log"

    echo ""
  done
done

echo "Done."
