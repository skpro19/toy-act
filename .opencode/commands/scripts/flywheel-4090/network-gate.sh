#!/bin/bash
# Network gate - tests operation latency and upload speed to the actual S3 key.
# Runs on the Vast.ai instance during provisioning, before cloning.
#
# Environment:
#   S3_PRESIGNED_PUT    – presigned S3 PUT URL for test upload
#   S3_PRESIGNED_DELETE – presigned S3 DELETE URL for cleanup
#
# Exits 0 (accept) or 1 (reject).

set -o pipefail

LATENCY_MAX_MS=${NETWORK_GATE_LATENCY_MS:-${NETWORK_GATE_RTT_MS:-5000}}
UPLOAD_MIN_KBPS=${NETWORK_GATE_UPLOAD_KBPS:-1000}
LATENCY_SAMPLES=${NETWORK_GATE_LATENCY_SAMPLES:-7}
LATENCY_MAX_ATTEMPTS=${NETWORK_GATE_LATENCY_MAX_ATTEMPTS:-10}
UPLOAD_SAMPLES=${NETWORK_GATE_UPLOAD_SAMPLES:-3}
UPLOAD_MAX_ATTEMPTS=${NETWORK_GATE_UPLOAD_MAX_ATTEMPTS:-5}
UPLOAD_SIZE_MB=${NETWORK_GATE_UPLOAD_SIZE_MB:-4}
TEST_FILE="/tmp/.netgate-test.bin"

require_positive_integer() {
  local name=$1
  local value=$2

  case "$value" in
    ""|*[!0-9]*|0)
      echo "ERROR: ${name} must be a positive integer" >&2
      exit 2
      ;;
  esac
}

median() {
  printf '%s\n' "$@" | sort -n | {
    mapfile -t sorted
    printf '%s\n' "${sorted[$((${#sorted[@]} / 2))]}"
  }
}

cleanup() {
  rm -f "$TEST_FILE"
  if [ -n "${S3_PRESIGNED_DELETE:-}" ]; then
    for attempt in 1 2 3; do
      curl --silent --fail --output /dev/null --request DELETE \
        --max-time 15 "$S3_PRESIGNED_DELETE" 2>/dev/null && return
      sleep "$attempt"
    done
    echo "WARNING: could not delete the temporary S3 network-gate object" >&2
  fi
}

trap cleanup EXIT

test -n "${S3_PRESIGNED_PUT:-}" || {
  echo "ERROR: S3_PRESIGNED_PUT is required" >&2
  exit 2
}
test -n "${S3_PRESIGNED_DELETE:-}" || {
  echo "ERROR: S3_PRESIGNED_DELETE is required" >&2
  exit 2
}

require_positive_integer "NETWORK_GATE_LATENCY_MS" "$LATENCY_MAX_MS"
require_positive_integer "NETWORK_GATE_UPLOAD_KBPS" "$UPLOAD_MIN_KBPS"
require_positive_integer "NETWORK_GATE_LATENCY_SAMPLES" "$LATENCY_SAMPLES"
require_positive_integer "NETWORK_GATE_LATENCY_MAX_ATTEMPTS" "$LATENCY_MAX_ATTEMPTS"
require_positive_integer "NETWORK_GATE_UPLOAD_SAMPLES" "$UPLOAD_SAMPLES"
require_positive_integer "NETWORK_GATE_UPLOAD_MAX_ATTEMPTS" "$UPLOAD_MAX_ATTEMPTS"
require_positive_integer "NETWORK_GATE_UPLOAD_SIZE_MB" "$UPLOAD_SIZE_MB"

if [ "$LATENCY_MAX_ATTEMPTS" -lt "$LATENCY_SAMPLES" ]; then
  echo "ERROR: latency max attempts must be at least the sample count" >&2
  exit 2
fi
if [ "$UPLOAD_MAX_ATTEMPTS" -lt "$UPLOAD_SAMPLES" ]; then
  echo "ERROR: upload max attempts must be at least the sample count" >&2
  exit 2
fi

: > "$TEST_FILE" || {
  echo "ERROR: could not create latency test file" >&2
  exit 2
}

# Empty PUTs measure the complete operation against the exact bucket and key,
# including DNS, connection setup, TLS, and S3 response time. A median rejects
# persistent poor routing without allowing one cold or transient request to
# discard an otherwise suitable instance.
echo "=== S3 PUT operation latency ==="
declare -a connect_samples=()
declare -a tls_samples=()
declare -a first_byte_samples=()
declare -a latency_samples=()

for attempt in $(seq 1 "$LATENCY_MAX_ATTEMPTS"); do
  metrics=$(curl --silent --fail --output /dev/null --upload-file "$TEST_FILE" \
    --connect-timeout 10 --max-time 20 \
    --write-out '%{time_connect} %{time_appconnect} %{time_starttransfer} %{time_total}' \
    "$S3_PRESIGNED_PUT" 2>/dev/null) || {
      echo "Attempt ${attempt}: request failed"
      continue
    }

  read -r connect_seconds tls_seconds first_byte_seconds total_seconds <<< "$metrics"
  connect_ms=$(awk -v value="$connect_seconds" 'BEGIN { printf "%.0f", value * 1000 }')
  tls_ms=$(awk -v value="$tls_seconds" 'BEGIN { printf "%.0f", value * 1000 }')
  first_byte_ms=$(awk -v value="$first_byte_seconds" 'BEGIN { printf "%.0f", value * 1000 }')
  total_ms=$(awk -v value="$total_seconds" 'BEGIN { printf "%.0f", value * 1000 }')

  connect_samples+=("$connect_ms")
  tls_samples+=("$tls_ms")
  first_byte_samples+=("$first_byte_ms")
  latency_samples+=("$total_ms")
  echo "Attempt ${attempt}: connect=${connect_ms}ms tls=${tls_ms}ms first-byte=${first_byte_ms}ms total=${total_ms}ms"

  if [ "${#latency_samples[@]}" -ge "$LATENCY_SAMPLES" ]; then
    break
  fi
done

if [ "${#latency_samples[@]}" -lt "$LATENCY_SAMPLES" ]; then
  echo ""
  echo "REJECTED: only ${#latency_samples[@]} of ${LATENCY_SAMPLES} required latency requests succeeded"
  exit 1
fi

CONNECT_MEDIAN_MS=$(median "${connect_samples[@]}")
TLS_MEDIAN_MS=$(median "${tls_samples[@]}")
FIRST_BYTE_MEDIAN_MS=$(median "${first_byte_samples[@]}")
LATENCY_MEDIAN_MS=$(median "${latency_samples[@]}")
echo "Median: connect=${CONNECT_MEDIAN_MS}ms tls=${TLS_MEDIAN_MS}ms first-byte=${FIRST_BYTE_MEDIAN_MS}ms total=${LATENCY_MEDIAN_MS}ms (max: ${LATENCY_MAX_MS}ms)"

if [ "$LATENCY_MEDIAN_MS" -gt "$LATENCY_MAX_MS" ]; then
  echo ""
  echo "REJECTED: median S3 PUT latency ${LATENCY_MEDIAN_MS}ms exceeds max ${LATENCY_MAX_MS}ms"
  exit 1
fi

echo ""
echo "=== S3 upload speed ==="
dd if=/dev/zero bs=1M count="$UPLOAD_SIZE_MB" of="$TEST_FILE" 2>/dev/null || {
  echo "ERROR: could not create upload test file" >&2
  exit 2
}

declare -a upload_samples=()
for attempt in $(seq 1 "$UPLOAD_MAX_ATTEMPTS"); do
  upload_bps=$(curl --silent --fail --output /dev/null --upload-file "$TEST_FILE" \
    --connect-timeout 10 --max-time 60 --write-out '%{speed_upload}' \
    "$S3_PRESIGNED_PUT" 2>/dev/null) || {
      echo "Attempt ${attempt}: upload failed"
      continue
    }

  upload_kbps=$(awk -v value="$upload_bps" 'BEGIN { printf "%.0f", value / 1024 }')
  upload_samples+=("$upload_kbps")
  echo "Attempt ${attempt}: ${upload_kbps} KB/s"

  if [ "${#upload_samples[@]}" -ge "$UPLOAD_SAMPLES" ]; then
    break
  fi
done

if [ "${#upload_samples[@]}" -lt "$UPLOAD_SAMPLES" ]; then
  echo ""
  echo "REJECTED: only ${#upload_samples[@]} of ${UPLOAD_SAMPLES} required uploads succeeded"
  exit 1
fi

UPLOAD_MEDIAN_KBPS=$(median "${upload_samples[@]}")
echo "Median upload: ${UPLOAD_MEDIAN_KBPS} KB/s (min: ${UPLOAD_MIN_KBPS} KB/s)"

if [ "$UPLOAD_MEDIAN_KBPS" -lt "$UPLOAD_MIN_KBPS" ]; then
  echo ""
  echo "REJECTED: median upload ${UPLOAD_MEDIAN_KBPS} KB/s below min ${UPLOAD_MIN_KBPS} KB/s"
  exit 1
fi

echo ""
echo "PASSED  latency=${LATENCY_MEDIAN_MS}ms  upload=${UPLOAD_MEDIAN_KBPS} KB/s"
