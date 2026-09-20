#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

readonly AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-toy-pickplace-backup}"
readonly AWS_REGION_NAME="${AWS_REGION_NAME:-ap-south-1}"
readonly BUCKET="${BUCKET:-toy-act}"
readonly PREFIX="${PREFIX:-checkpoints/act_v1}"
readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_name="${1:-}"

if [ -z "$run_name" ]; then
  run_name="$(
    aws s3 ls "s3://${BUCKET}/${PREFIX}/" \
      --profile "$AWS_PROFILE_NAME" --region "$AWS_REGION_NAME" \
      | awk '/ PRE /{print $2}' | sed 's:/$::' | sort | tail -n 1
  )"
fi

test -n "$run_name" || {
  echo "ERROR: no checkpoint run found under s3://${BUCKET}/${PREFIX}/" >&2
  exit 1
}

readonly SOURCE="s3://${BUCKET}/${PREFIX}/${run_name}/"
readonly DESTINATION="${PROJECT_DIR}/${PREFIX}/${run_name}"

echo "Latest run:  ${run_name}"
echo "Source:      ${SOURCE}"
echo "Destination: ${DESTINATION}"

mkdir -p "$DESTINATION"
aws s3 cp "$SOURCE" "$DESTINATION" \
  --recursive \
  --profile "$AWS_PROFILE_NAME" \
  --region "$AWS_REGION_NAME"

echo "Done. Downloaded to ${DESTINATION}"
