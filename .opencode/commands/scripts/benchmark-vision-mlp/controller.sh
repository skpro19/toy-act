#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${CONTROL_DIR:?CONTROL_DIR is required}"
: "${CONTRACT_JSON:?CONTRACT_JSON is required}"

TOOLS="$CONTROL_DIR/tools"
GENERATION="$(date -u +%Y%m%dT%H%M%SZ)-$$-$(cat /proc/sys/kernel/random/uuid)"
export MUJOCO_GL=egl

json_value() {
  /root/.local/bin/uv run --directory "$PROJECT_ROOT" python -c \
    'import json,sys; value=json.load(open(sys.argv[1])); [value := value[key] for key in sys.argv[2].split(".")]; print(str(value).lower() if isinstance(value, bool) else value)' \
    "$CONTRACT_JSON" "$1"
}

GIT_COMMIT=$(json_value commit)
BATCH_SIZE=$(json_value batch_size)
EXPERT_EPISODES=$(json_value fixed.expert_episodes)
EXPERT_MAX_STEPS=$(json_value fixed.expert_max_steps)
TRAIN_CAPTURE_HZ=$(json_value fixed.train_capture_hz)
EVAL_EPISODES=$(json_value fixed.eval_episodes)
EVAL_MAX_STEPS=$(json_value fixed.eval_max_steps)
EVAL_CAPTURE_HZ=$(json_value fixed.eval_capture_hz)
DAGGER_EPISODES=$(json_value fixed.dagger_episodes)
ROLLOUT_MAX_STEPS=$(json_value fixed.dagger_max_steps)
INTERVENTION_THRESHOLD=$(json_value fixed.intervention_threshold)
INTERVENTION_STEPS=$(json_value fixed.intervention_steps)
EXPERT_SEED=$(json_value seeds.expert)
TRAIN_SEED=$(json_value seeds.train)
EVAL_SEED=$(json_value seeds.eval)
DAGGER_SEED=$(json_value seeds.dagger)

mkdir -p \
  "$CONTROL_DIR/data/expert" \
  "$CONTROL_DIR/plans" \
  "$CONTROL_DIR/logs" \
  "$CONTROL_DIR/telemetry" \
  "$CONTROL_DIR/time" \
  "$CONTROL_DIR/results/training" \
  "$CONTROL_DIR/results/workers" \
  "$CONTROL_DIR/status/history" \
  "$CONTROL_DIR/state" \
  "$CONTROL_DIR/bootstrap"

exec 9>"$CONTROL_DIR/state/controller.lock"
flock -n 9 || exit 75

current_phase=initialization
current_trial=none
on_error() {
  local code=$?
  local attempt=unknown
  local log=none
  local status_path="$CONTROL_DIR/status/$current_trial.status"
  if [[ -f "$status_path" ]]; then
    read -r -a failed_status <"$status_path"
    attempt=${failed_status[-1]:-unknown}
    log="$CONTROL_DIR/logs/$current_trial-a$attempt.log"
  fi
  printf 'phase=%s trial=%s exit=%s generation=%s attempt=%s log=%s\n' \
    "$current_phase" "$current_trial" "$code" "$GENERATION" "$attempt" "$log" \
    >"$CONTROL_DIR/state/failed.tmp.$$"
  mv -f "$CONTROL_DIR/state/failed.tmp.$$" "$CONTROL_DIR/state/failed"
  exit "$code"
}
trap on_error ERR

cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$GIT_COMMIT" ]]
/root/.local/bin/uv run python "$TOOLS/verify_scripts.py" \
  --tools-dir "$TOOLS" \
  --manifest "$CONTROL_DIR/script-manifest.json"
printf '%s\n' "$GENERATION" >"$CONTROL_DIR/state/started.tmp.$$"
mv -f "$CONTROL_DIR/state/started.tmp.$$" "$CONTROL_DIR/state/started"

write_plan() {
  local destination=$1
  local temporary="$destination.tmp.$$"
  cat >"$temporary"
  if [[ -f "$destination" ]]; then
    cmp -s "$temporary" "$destination" || {
      printf 'ERROR: existing plan differs: %s\n' "$destination" >&2
      return 1
    }
    rm -f "$temporary"
  else
    mv "$temporary" "$destination"
  fi
}

write_plan "$CONTROL_DIR/plans/training.plan" <<EOF
order|repetition|batch_size|dataloader_workers|persistent_workers|trial_id
1|1|$BATCH_SIZE|0|false|dw0-p0-r1
2|1|$BATCH_SIZE|2|false|dw2-p0-r1
3|1|$BATCH_SIZE|2|true|dw2-p1-r1
4|1|$BATCH_SIZE|4|false|dw4-p0-r1
5|1|$BATCH_SIZE|4|true|dw4-p1-r1
6|1|$BATCH_SIZE|8|false|dw8-p0-r1
7|1|$BATCH_SIZE|8|true|dw8-p1-r1
8|2|$BATCH_SIZE|8|true|dw8-p1-r2
9|2|$BATCH_SIZE|8|false|dw8-p0-r2
10|2|$BATCH_SIZE|4|true|dw4-p1-r2
11|2|$BATCH_SIZE|4|false|dw4-p0-r2
12|2|$BATCH_SIZE|2|true|dw2-p1-r2
13|2|$BATCH_SIZE|2|false|dw2-p0-r2
14|2|$BATCH_SIZE|0|false|dw0-p0-r2
15|3|$BATCH_SIZE|4|false|dw4-p0-r3
16|3|$BATCH_SIZE|4|true|dw4-p1-r3
17|3|$BATCH_SIZE|8|false|dw8-p0-r3
18|3|$BATCH_SIZE|8|true|dw8-p1-r3
19|3|$BATCH_SIZE|0|false|dw0-p0-r3
20|3|$BATCH_SIZE|2|false|dw2-p0-r3
21|3|$BATCH_SIZE|2|true|dw2-p1-r3
EOF

write_plan "$CONTROL_DIR/plans/workers.plan" <<EOF
order|repetition|workload|workers|trial_id
1|1|evaluation|6|evaluation-w6-r1
2|1|dagger|6|dagger-w6-r1
3|1|evaluation|12|evaluation-w12-r1
4|1|dagger|12|dagger-w12-r1
5|2|evaluation|12|evaluation-w12-r2
6|2|dagger|12|dagger-w12-r2
7|2|evaluation|6|evaluation-w6-r2
8|2|dagger|6|dagger-w6-r2
9|3|evaluation|6|evaluation-w6-r3
10|3|dagger|6|dagger-w6-r3
11|3|evaluation|12|evaluation-w12-r3
12|3|dagger|12|dagger-w12-r3
EOF

status_attempt() {
  local trial_id=$1
  local count
  count=$(find "$CONTROL_DIR/status/history" -maxdepth 1 -type f -name "$trial_id-a*.status" | wc -l)
  printf '%s\n' "$((count + 1))"
}

prepare_trial() {
  local kind=$1
  local trial_id=$2
  local result=$3
  local status_path="$CONTROL_DIR/status/$trial_id.status"
  if [[ -f "$status_path" ]]; then
    read -r -a status_values <"$status_path"
    if [[ "${status_values[0]:-}" == succeeded && "${status_values[1]:-}" == 0 && "${#status_values[@]}" -eq 4 ]]; then
      local status_generation=${status_values[2]}
      local status_attempt_value=${status_values[3]}
      if /root/.local/bin/uv run python "$TOOLS/validate_result.py" \
          --control-dir "$CONTROL_DIR" \
          --kind "$kind" \
          --result "$result" \
          --trial-id "$trial_id" \
          --generation "$status_generation" \
          --attempt "$status_attempt_value"; then
        return 1
      fi
      printf 'failed 125 %s %s\n' "$status_generation" "$status_attempt_value" \
        >"$status_path.tmp.$$"
      mv -f "$status_path.tmp.$$" "$status_path"
      printf 'ERROR: succeeded trial failed validation: %s\n' "$trial_id" >&2
      return 2
    fi
    if [[ "${status_values[0]:-}" == running && "${#status_values[@]}" -eq 3 ]]; then
      local status_generation=${status_values[1]}
      local status_attempt_value=${status_values[2]}
      printf 'failed 125 %s %s\n' "$status_generation" "$status_attempt_value" \
        >"$status_path.tmp.$$"
      mv -f "$status_path.tmp.$$" "$status_path"
    fi
    printf 'ERROR: trial requires explicit recovery: %s (%s)\n' \
      "$trial_id" "$(cat "$status_path")" >&2
    return 2
  fi
  ATTEMPT=$(status_attempt "$trial_id")
  return 0
}

current_phase=expert-collection
current_trial=expert
if [[ -f "$CONTROL_DIR/status/expert.status" ]]; then
  read -r -a expert_status <"$CONTROL_DIR/status/expert.status"
  if [[ "${expert_status[0]:-}" == running && "${#expert_status[@]}" -eq 3 ]]; then
    printf 'failed 125 %s %s\n' "${expert_status[1]}" "${expert_status[2]}" \
      >"$CONTROL_DIR/status/expert.status.tmp.$$"
    mv -f "$CONTROL_DIR/status/expert.status.tmp.$$" \
      "$CONTROL_DIR/status/expert.status"
  fi
  [[ "${expert_status[0]:-}" == succeeded && "${expert_status[1]:-}" == 0 && "${#expert_status[@]}" -eq 4 ]] || {
    printf '%s\n' 'ERROR: expert collection requires explicit recovery' >&2
    exit 1
  }
else
  EXPERT_ATTEMPT=$(status_attempt expert)
  PROJECT_ROOT="$PROJECT_ROOT" \
  CONTROL_DIR="$CONTROL_DIR" \
  EXPERT_EPISODES="$EXPERT_EPISODES" \
  EXPERT_SEED="$EXPERT_SEED" \
  EXPERT_MAX_STEPS="$EXPERT_MAX_STEPS" \
  TRAIN_CAPTURE_HZ="$TRAIN_CAPTURE_HZ" \
  GENERATION="$GENERATION" \
  ATTEMPT="$EXPERT_ATTEMPT" \
    "$TOOLS/collect_data.sh"
fi

current_phase=dataset-validation
/root/.local/bin/uv run python "$TOOLS/validate_dataset.py" \
  --data-dir "$CONTROL_DIR/data/expert" \
  --expected-episodes "$EXPERT_EPISODES" \
  --output "$CONTROL_DIR/data/manifest.json"
DATASET_DIGEST=$(
  /root/.local/bin/uv run python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["aggregate_sha256"])' \
    "$CONTROL_DIR/data/manifest.json"
)
/root/.local/bin/uv run python "$TOOLS/initialize_metadata.py" \
  --contract "$CONTRACT_JSON" \
  --manifest "$CONTROL_DIR/data/manifest.json" \
  --output "$CONTROL_DIR/metadata.json"

run_training_trial() {
  local trial_id=$1
  local workers=$2
  local persistent=$3
  local measured_epochs=$4
  local result="$CONTROL_DIR/results/training/$trial_id.json"
  local persistence_flag=--no-persistent-workers
  [[ "$persistent" == true ]] && persistence_flag=--persistent-workers
  if prepare_trial training "$trial_id" "$result"; then
    "$TOOLS/run_trial.sh" "$CONTROL_DIR" training "$trial_id" "$GENERATION" "$ATTEMPT" "$result" -- \
      /root/.local/bin/uv run python "$TOOLS/benchmark_training.py" \
        --project-root "$PROJECT_ROOT" \
        --data-dir "$CONTROL_DIR/data/expert" \
        --batch-size "$BATCH_SIZE" \
        --dataloader-workers "$workers" \
        "$persistence_flag" \
        --sample-seed "$TRAIN_SEED" \
        --model-seed "$TRAIN_SEED" \
        --warmup-epochs 1 \
        --measured-epochs "$measured_epochs" \
        --trial-id "$trial_id" \
        --generation "$GENERATION" \
        --attempt "$ATTEMPT" \
        --git-commit "$GIT_COMMIT" \
        --dataset-digest "$DATASET_DIGEST" \
        --output "$result"
  else
    local status=$?
    [[ "$status" -eq 1 ]] || return "$status"
  fi
}

current_phase=preflight
current_trial=preflight
run_training_trial preflight 0 false 1

current_phase=training
while IFS='|' read -r _ _ _ workers persistent trial_id; do
  [[ "$trial_id" == trial_id ]] && continue
  current_trial=$trial_id
  run_training_trial "$trial_id" "$workers" "$persistent" 5
done <"$CONTROL_DIR/plans/training.plan"

current_phase=bootstrap
current_trial=bootstrap
bootstrap_result="$CONTROL_DIR/results/workers/bootstrap.json"
if prepare_trial bootstrap bootstrap "$bootstrap_result"; then
  "$TOOLS/run_trial.sh" "$CONTROL_DIR" bootstrap bootstrap "$GENERATION" "$ATTEMPT" "$bootstrap_result" -- \
    /root/.local/bin/uv run python "$TOOLS/benchmark_bootstrap.py" \
      --project-root "$PROJECT_ROOT" \
      --control-dir "$CONTROL_DIR" \
      --batch-size "$BATCH_SIZE" \
      --sample-seed "$TRAIN_SEED" \
      --eval-seed "$EVAL_SEED" \
      --eval-max-steps "$EVAL_MAX_STEPS" \
      --eval-capture-hz "$EVAL_CAPTURE_HZ" \
      --trial-id bootstrap \
      --generation "$GENERATION" \
      --attempt "$ATTEMPT" \
      --git-commit "$GIT_COMMIT" \
      --dataset-digest "$DATASET_DIGEST" \
      --output "$bootstrap_result"
else
  status=$?
  [[ "$status" -eq 1 ]] || exit "$status"
fi

run_worker_trial() {
  local kind=$1
  local workers=$2
  local repetition=$3
  local trial_id="$kind-w$workers-r$repetition"
  local result="$CONTROL_DIR/results/workers/$trial_id.json"
  current_trial=$trial_id
  if prepare_trial "$kind" "$trial_id" "$result"; then
    :
  else
    local status=$?
    [[ "$status" -eq 1 ]] && return 0
    return "$status"
  fi
  if [[ "$kind" == evaluation ]]; then
    "$TOOLS/run_trial.sh" "$CONTROL_DIR" evaluation "$trial_id" "$GENERATION" "$ATTEMPT" "$result" -- \
      /root/.local/bin/uv run python "$TOOLS/benchmark_worker.py" evaluation \
        --project-root "$PROJECT_ROOT" \
        --checkpoint-metadata "$CONTROL_DIR/bootstrap/checkpoint.json" \
        --seed "$EVAL_SEED" \
        --episodes "$EVAL_EPISODES" \
        --max-steps "$EVAL_MAX_STEPS" \
        --capture-hz "$EVAL_CAPTURE_HZ" \
        --workers "$workers" \
        --trial-id "$trial_id" \
        --generation "$GENERATION" \
        --attempt "$ATTEMPT" \
        --git-commit "$GIT_COMMIT" \
        --dataset-digest "$DATASET_DIGEST" \
        --output "$result"
  else
    "$TOOLS/run_trial.sh" "$CONTROL_DIR" dagger "$trial_id" "$GENERATION" "$ATTEMPT" "$result" -- \
      /root/.local/bin/uv run python "$TOOLS/benchmark_worker.py" dagger \
        --project-root "$PROJECT_ROOT" \
        --checkpoint-metadata "$CONTROL_DIR/bootstrap/checkpoint.json" \
        --seed "$DAGGER_SEED" \
        --episodes "$DAGGER_EPISODES" \
        --max-steps "$ROLLOUT_MAX_STEPS" \
        --capture-hz "$TRAIN_CAPTURE_HZ" \
        --workers "$workers" \
        --intervention-threshold "$INTERVENTION_THRESHOLD" \
        --intervention-steps "$INTERVENTION_STEPS" \
        --output-dir "$CONTROL_DIR/results/workers/$trial_id-a$ATTEMPT-data" \
        --trial-id "$trial_id" \
        --generation "$GENERATION" \
        --attempt "$ATTEMPT" \
        --git-commit "$GIT_COMMIT" \
        --dataset-digest "$DATASET_DIGEST" \
        --output "$result"
  fi
}

current_phase=workers
while IFS='|' read -r _ repetition kind workers trial_id; do
  [[ "$trial_id" == trial_id ]] && continue
  run_worker_trial "$kind" "$workers" "$repetition"
done <"$CONTROL_DIR/plans/workers.plan"

current_phase=initial-analysis
current_trial=none
/root/.local/bin/uv run python "$TOOLS/analyze.py" \
  --control-dir "$CONTROL_DIR" \
  --output-dir "$CONTROL_DIR"

stability_training_configurations=$(
  /root/.local/bin/uv run python -c \
    'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["training"]))' \
    "$CONTROL_DIR/stability-needed.json"
)
if [[ ! -f "$CONTROL_DIR/plans/stability-training.plan" ]]; then
  {
    printf '%s\n' 'order|repetition|batch_size|dataloader_workers|persistent_workers|trial_id'
    order=0
    while read -r configuration; do
      [[ -z "$configuration" ]] && continue
      workers=${configuration#dw}
      workers=${workers%%-*}
      persistent=false
      [[ "$configuration" == *-p1 ]] && persistent=true
      for repetition in 4 5; do
        order=$((order + 1))
        printf '%s|%s|%s|%s|%s|%s-r%s\n' \
          "$order" "$repetition" "$BATCH_SIZE" "$workers" "$persistent" \
          "$configuration" "$repetition"
      done
    done <<<"$stability_training_configurations"
  } >"$CONTROL_DIR/plans/stability-training.plan.tmp.$$"
  mv "$CONTROL_DIR/plans/stability-training.plan.tmp.$$" \
    "$CONTROL_DIR/plans/stability-training.plan"
fi

stability_worker_configurations=$(
  /root/.local/bin/uv run python -c \
    'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["workers"]))' \
    "$CONTROL_DIR/stability-needed.json"
)
if [[ ! -f "$CONTROL_DIR/plans/stability-workers.plan" ]]; then
  {
    printf '%s\n' 'order|repetition|workload|workers|trial_id'
    order=0
    while read -r workers; do
      [[ -z "$workers" ]] && continue
      for repetition in 4 5; do
        for kind in evaluation dagger; do
          order=$((order + 1))
          printf '%s|%s|%s|%s|%s-w%s-r%s\n' \
            "$order" "$repetition" "$kind" "$workers" "$kind" "$workers" \
            "$repetition"
        done
      done
    done <<<"$stability_worker_configurations"
  } >"$CONTROL_DIR/plans/stability-workers.plan.tmp.$$"
  mv "$CONTROL_DIR/plans/stability-workers.plan.tmp.$$" \
    "$CONTROL_DIR/plans/stability-workers.plan"
fi

current_phase=stability
while IFS='|' read -r _ _ _ workers persistent trial_id; do
  [[ "$trial_id" == trial_id ]] && continue
  run_training_trial "$trial_id" "$workers" "$persistent" 5
done <"$CONTROL_DIR/plans/stability-training.plan"
while IFS='|' read -r _ repetition kind workers trial_id; do
  [[ "$trial_id" == trial_id ]] && continue
  run_worker_trial "$kind" "$workers" "$repetition"
done <"$CONTROL_DIR/plans/stability-workers.plan"

printf '{"finished_utc":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$CONTROL_DIR/state/workload-finished.json.tmp.$$"
mv "$CONTROL_DIR/state/workload-finished.json.tmp.$$" \
  "$CONTROL_DIR/state/workload-finished.json"

current_phase=final-analysis
/root/.local/bin/uv run python "$TOOLS/analyze.py" \
  --control-dir "$CONTROL_DIR" \
  --output-dir "$CONTROL_DIR" \
  --require-complete

summary_sha256=$(sha256sum "$CONTROL_DIR/benchmark-summary.json" | cut -d ' ' -f1)
printf 'generation=%s\nsummary_sha256=%s\n' "$GENERATION" "$summary_sha256" \
  >"$CONTROL_DIR/state/completed.tmp.$$"
mv "$CONTROL_DIR/state/completed.tmp.$$" "$CONTROL_DIR/state/completed"
rm -f "$CONTROL_DIR/state/failed"
trap - ERR
