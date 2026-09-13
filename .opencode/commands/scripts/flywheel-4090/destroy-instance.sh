#!/bin/bash
# Idempotently destroy one known Vast instance and retry until removal is verified.

set -o pipefail

if [ "$#" -ne 1 ]; then
  echo "ERROR: destroy-instance.sh requires exactly one instance ID" >&2
  exit 2
fi

INSTANCE_ID="$1"
case "$INSTANCE_ID" in
  ""|*[!0-9]*) echo "ERROR: INSTANCE_ID must be numeric" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(readlink -f "${SCRIPT_DIR}/../../../..")
set -a
. "${PROJECT_ROOT}/.env" || {
  echo "ERROR: could not load ${PROJECT_ROOT}/.env" >&2
  exit 1
}
set +a

destroy_attempt=0
while true; do
  destroy_attempt=$((destroy_attempt + 1))
  echo "Destroy request $destroy_attempt for instance $INSTANCE_ID"
  vastai destroy instance -y "$INSTANCE_ID" 2>&1 || \
    echo "WARNING: destroy request failed; verification will continue"

  for poll in $(seq 1 30); do
    if ! instances=$(vastai show instances --raw 2>/dev/null); then
      echo "WARNING: instance query failed ($poll/30)"
      sleep 10
      continue
    fi
    if ! count=$(printf '%s\n' "$instances" | INSTANCE_ID="$INSTANCE_ID" \
      uv run python -c '
import json, os, sys
data = json.load(sys.stdin)
if isinstance(data, dict):
    data = data.get("instances", [data])
target = os.environ["INSTANCE_ID"]
print(sum(str(item.get("id")) == target for item in data))
' 2>/dev/null); then
      echo "WARNING: invalid instance query response ($poll/30)"
      sleep 10
      continue
    fi
    if [ "$count" = "0" ]; then
      echo "Instance $INSTANCE_ID destroyed and removed"
      exit 0
    fi
    sleep 10
  done

  echo "WARNING: instance $INSTANCE_ID is still present; retrying in 30 seconds"
  sleep 30
done
