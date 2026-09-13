#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 8 ]]; then
  printf '%s\n' 'usage: run_trial.sh CONTROL_DIR KIND TRIAL_ID GENERATION ATTEMPT RESULT -- COMMAND...' >&2
  exit 2
fi

CONTROL_DIR=$1
kind=$2
trial_id=$3
generation=$4
attempt=$5
result_path=$6
shift 6
[[ "$1" == -- ]]
shift

TOOLS="$CONTROL_DIR/tools"
status_path="$CONTROL_DIR/status/$trial_id.status"
log_path="$CONTROL_DIR/logs/$trial_id-a$attempt.log"
time_path="$CONTROL_DIR/time/$trial_id-a$attempt.txt"
gpu_path="$CONTROL_DIR/telemetry/$trial_id-a$attempt-gpu.csv"
pidstat_path="$CONTROL_DIR/telemetry/$trial_id-a$attempt-pidstat.txt"
vmstat_path="$CONTROL_DIR/telemetry/$trial_id-a$attempt-vmstat.txt"
omissions_path="$CONTROL_DIR/telemetry/$trial_id-a$attempt-omissions.json"

write_status() {
  local text=$1
  local temporary="$status_path.tmp.$$"
  printf '%s\n' "$text" >"$temporary"
  mv -f "$temporary" "$status_path"
}

write_status "running $generation $attempt"
rm -f "$result_path"

cool_samples=0
for _ in $(seq 1 120); do
  read -r utilization temperature < <(
    nvidia-smi --query-gpu=utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits | tr -d ','
  )
  if [[ "$utilization" -lt 5 && "$temperature" -lt 55 ]]; then
    cool_samples=$((cool_samples + 1))
    [[ "$cool_samples" -ge 5 ]] && break
  else
    cool_samples=0
  fi
  sleep 1
done
if [[ "$cool_samples" -lt 5 ]]; then
  write_status "failed 124 $generation $attempt"
  exit 124
fi

sampler_pids=()
command_pid=
cleanup() {
  local pid
  for pid in "${sampler_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${sampler_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  if [[ -n "$command_pid" ]] && kill -0 "$command_pid" 2>/dev/null; then
    kill -TERM -- "-$command_pid" 2>/dev/null || true
    wait "$command_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

gpu_fields='timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.graphics,clocks.current.memory,memory.used,memory.total'
omitted_fields=()
for optional_field in pcie.rx_util pcie.tx_util; do
  optional_value=$(nvidia-smi --query-gpu="$optional_field" \
    --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ "$optional_value" =~ ^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$ ]]; then
    gpu_fields="$gpu_fields,$optional_field"
  else
    omitted_fields+=("$optional_field")
  fi
done
/root/.local/bin/uv run --directory /workspace/toy-pickplace python -c \
  'import json,sys; json.dump({"omitted_gpu_fields":sys.argv[2:]},open(sys.argv[1],"w"))' \
  "$omissions_path" "${omitted_fields[@]}"

(
  printf '%s\n' "$gpu_fields"
  while true; do
    nvidia-smi --query-gpu="$gpu_fields" --format=csv,noheader,nounits
    sleep 1
  done
) >"$gpu_path" 2>&1 &
sampler_pids+=("$!")
vmstat 1 >"$vmstat_path" 2>&1 &
sampler_pids+=("$!")

set +e
setsid /usr/bin/time -v -o "$time_path" "$@" >"$log_path" 2>&1 &
command_pid=$!
pidstat -dur -p "$command_pid" -T ALL 1 >"$pidstat_path" 2>&1 &
sampler_pids+=("$!")
wait "$command_pid"
command_status=$?
set -e
cleanup
trap - EXIT INT TERM

validation_status=0
if [[ "$command_status" -eq 0 ]]; then
  set +e
  /root/.local/bin/uv run --directory /workspace/toy-pickplace python \
    "$TOOLS/validate_telemetry.py" \
    --gpu "$gpu_path" \
    --pidstat "$pidstat_path" \
    --vmstat "$vmstat_path" \
    --time "$time_path" >>"$log_path" 2>&1
  validation_status=$?
  set -e
fi
if [[ "$command_status" -eq 0 && "$validation_status" -eq 0 ]]; then
  set +e
  /root/.local/bin/uv run --directory /workspace/toy-pickplace python \
    "$TOOLS/validate_result.py" \
    --control-dir "$CONTROL_DIR" \
    --kind "$kind" \
    --result "$result_path" \
    --trial-id "$trial_id" \
    --generation "$generation" \
    --attempt "$attempt" >>"$log_path" 2>&1
  validation_status=$?
  set -e
fi

if [[ "$command_status" -ne 0 ]]; then
  final_status=$command_status
elif [[ "$validation_status" -ne 0 ]]; then
  final_status=125
elif [[ ! -s "$time_path" || ! -s "$gpu_path" || ! -s "$pidstat_path" || ! -s "$vmstat_path" ]]; then
  final_status=125
else
  final_status=0
fi

if [[ "$final_status" -eq 0 ]]; then
  write_status "succeeded 0 $generation $attempt"
else
  write_status "failed $final_status $generation $attempt"
fi
exit "$final_status"
