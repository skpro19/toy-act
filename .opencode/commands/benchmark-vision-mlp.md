---
description: Benchmark vision_mlp DataLoader and simulator parallelism on one RTX 4090
agent: build
---

Run a short, controlled infrastructure benchmark for `vision_mlp` on a new
Vast.ai RTX 4090. This command measures DataLoader behavior and simulator
parallelism only. It does not run a complete flywheel, sweep batch size, compare
policy quality, or change experiment hyperparameters. Use
`/ablate-flywheel-params batch_size` separately for batch-size and policy-quality
experiments.

This command owns baseline confirmation, provisioning, hardware acceptance,
fixed-data collection, versioned benchmark harnesses, sequential execution,
telemetry, analysis, local result download, recovery, and cleanup. It is
self-contained: follow this workflow without consulting another runbook.

Executable benchmark logic lives in
`.opencode/commands/scripts/benchmark-vision-mlp/`. Treat those files as part of
this command. The selected commit must contain them. Materialize their exact
Git-object bytes at that commit into `$CONTROL_DIR/tools`, and record their
SHA-256 digests in control artifacts. Never copy scripts from the local working
tree. Do not regenerate, paraphrase, or partially reimplement them during a run.

Never reuse, modify, stop, restart, signal, SSH into, or compete with an
existing training instance. Provision a separate instance even when another
RTX 4090 is already running.

## Workflow at a glance

1. Confirm the remote branch, commit, config, and fixed benchmark contract.
2. Provision a separate RTX 4090 with the exact hardware filters below.
3. Verify the actual CPU allocation, RAM, GPU, PCIe link, and throttling state.
4. Collect one fixed 100-episode image dataset and validate its manifest.
5. Run 21 short, sequential `vision_mlp` training-throughput trials.
6. Train one bootstrap checkpoint and run 12 simulator-worker timing trials.
7. Produce local Markdown, JSON, and CSV reports with infrastructure
   recommendations.
8. Offer cleanup only after all reports and raw measurements are downloaded.

## Scope and non-goals

This workflow answers only these questions:

| Question | Candidates | Primary metric |
|---|---|---|
| How many DataLoader workers should one `vision_mlp` run use? | `0, 2, 4, 8` | median samples/s |
| Would persistent DataLoader workers help? | `false, true` when workers > 0 | matched throughput speedup |
| How many simulator workers should evaluation and DAgger use? | `6, 12` | median paired wall time |

The following are fixed controls, not ablation dimensions:

| Setting | Source or value |
|---|---|
| Architecture | `vision_mlp` |
| Batch size | exact confirmed config value, normally `768` |
| Root seed | exact confirmed config value, normally `0` |
| Expert episodes | exact confirmed config value, required to be `100` |
| Training capture rate | exact confirmed config value, normally `60` Hz |
| Evaluation episodes | exact confirmed config value, normally `25` |
| Evaluation maximum steps | exact confirmed config value, normally `1400` |
| Evaluation capture rate | exact confirmed config value, normally `60` Hz |
| DAgger episodes | exact confirmed config value, normally `50` |
| DAgger maximum steps | exact confirmed config value, normally `1400` |
| Intervention threshold | exact confirmed config value, normally `0.1` |
| Intervention steps | exact confirmed config value, normally `50` |

Do not sweep batch size, episode counts, learning rate, seed, DAgger parameters,
evaluation parameters, architecture, or model quality. Do not run any complete
flywheel or use `scripts/flywheel.py` to produce a quality result. Do not claim
that the fastest infrastructure setting produces a better policy.

## Exact benchmark plan

### Training-throughput matrix

Use the config's fixed batch size for every trial. Test exactly seven
configurations:

| ID | `dataloader_workers` | `persistent_workers` | Production status |
|---|---:|---|---|
| `dw0-p0` | 0 | false | supported |
| `dw2-p0` | 2 | false | supported |
| `dw2-p1` | 2 | true | supported |
| `dw4-p0` | 4 | false | supported |
| `dw4-p1` | 4 | true | supported |
| `dw8-p0` | 8 | false | supported |
| `dw8-p1` | 8 | true | supported |

Run each configuration three times with the same dataset, model seed, sampler
seed, and batch size:

```text
7 configurations * 3 repetitions = 21 measured training trials
```

Each trial constructs a fresh model and executes one unmeasured warm-up epoch
followed by five measured epochs. Repetitions are timing repetitions, not
DAgger rounds and not complete training runs.

Test persistent and non-persistent workers as separate production-supported
groups. Keep the bootstrap fixed at zero workers and non-persistent because it
only supplies a representative checkpoint for simulator timing.

### Simulator-worker matrix

Train one bootstrap checkpoint, then run:

| Workload | Workers | Repetitions | Trials |
|---|---|---:|---:|
| Evaluation | `6, 12` | 3 | 6 |
| Threshold DAgger collection | `6, 12` | 3 | 6 |

Run all 12 worker trials sequentially with the same checkpoint, seeds, episode
counts, and maximum steps. These repetitions measure runtime noise. They do not
constitute DAgger rounds and their generated data is not used for training.

### Expected duration

Estimate and show expected setup time, trial time, and cost before provisioning.
The measured workload contains 21 short training trials, one 120-epoch bootstrap,
and 12 simulator trials. It should normally finish in approximately 15-25
minutes after environment setup. Do not promise an exact duration.

## Measurement contract

### Training trial

Each training trial must:

1. use the same validated 100-episode expert image dataset;
2. use the config's fixed batch size;
3. build the production `PickPlaceVisionDataset`, weighted sampler, VisionMLP,
   Adam optimizer, losses, and `run_epoch` implementation;
4. use the same model and sampler seeds;
5. execute one unmeasured warm-up epoch;
6. reset CUDA peak-memory statistics;
7. execute five measured epochs;
8. call `torch.cuda.synchronize()` immediately before and after each measured
   epoch;
9. include DataLoader iterator creation, worker creation, and worker teardown
   that occur inside each measured epoch;
10. record per-epoch seconds, total measured seconds, samples/s, batches/epoch,
    final losses, measured-window timestamps, peak allocated VRAM, peak reserved
    VRAM, and exit status;
11. record one-second GPU, process-tree, and host telemetry;
12. record `/usr/bin/time -v` output.

Calculate throughput as:

```text
samples_per_second = (dataset.samples_per_epoch * 5) / measured_seconds
```

Exclude dataset construction, model construction, and the warm-up epoch from
ranked throughput. Report whole-process wall time separately.

### Worker trial

An evaluation trial uses the bootstrap checkpoint with the config's derived
evaluation seed, configured evaluation episodes, configured maximum steps, and
candidate worker count.

A DAgger trial uses the same bootstrap checkpoint with the config's derived
DAgger seed, configured DAgger episodes, configured rollout maximum steps,
threshold intervention mode, configured threshold and intervention length,
headless EGL, and a unique output directory. Record wall time and telemetry.
Validate the exact episode count before accepting success.

### Telemetry

Collect these GPU fields once per second when supported:

```text
timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,
clocks.current.graphics,clocks.current.memory,memory.used,memory.total,
pcie.rx_util,pcie.tx_util
```

Also collect:

- `pidstat` process-tree CPU, memory, faults, and I/O once per second;
- `vmstat` runnable tasks, CPU utilization, I/O wait, swapping, and block I/O
  once per second;
- `/usr/bin/time -v` CPU time, wall time, maximum RSS, faults, filesystem I/O,
  and context switches.

If one NVIDIA field is unsupported, remove only that field and record the
omission in metadata. Never change GPU power limits, clocks, persistence mode,
compute mode, fan settings, or application clocks.

Use measured-window timestamps from the harness to rank training utilization.
Report setup, warm-up, and teardown telemetry separately.

## Ranking and stability

For every configuration, report all three repetitions plus median, minimum,
maximum, and relative spread:

```text
relative_spread = (maximum - minimum) / median
```

If a top-three configuration in either separately ranked training group, or
either worker candidate, has relative spread greater than 10%, run two
additional repetitions for only that candidate and use the median of all five.
Do not discard or hide outliers.

Rank production-supported training candidates separately from persistent-worker
candidates. Rank production-supported settings by:

1. highest median samples/s;
2. lowest median measured epoch time;
3. lower p95 GPU idle fraction;
4. lower peak RAM;
5. fewer DataLoader workers when throughput differs by no more than 2%.

For every persistent candidate, show speedup relative to the matching
non-persistent candidate.

For worker repetition `r`, define paired time as:

```text
paired_seconds[r] = evaluation_seconds[r] + dagger_seconds[r]
```

Rank worker candidates by median paired time, then peak RAM and CPU. Prefer the
lower worker count when median paired time differs by no more than 2%.

## Baseline confirmation

Ask the user to select the Git branch to clone, defaulting to `dev`. Set:

```text
FLYWHEEL_CONFIG=configs/flywheel/mlp_vision/default_mlp_vision_instance.yaml
BENCHMARK_STAMP=YYYYMMDD-HHMMSS
INSTANCE_LABEL=toy-pickplace-benchmark-vision-mlp-${BENCHMARK_STAMP}
```

Validate the branch, fetch it, resolve the exact commit, and load every source
file needed for planning without changing the local checkout:

```bash
case "$GIT_BRANCH" in
  ""|*[!A-Za-z0-9._/-]*)
    printf '%s\n' 'ERROR: Branch may contain only letters, digits, ., _, /, and -' >&2
    exit 1
    ;;
esac
git check-ref-format --branch "$GIT_BRANCH"
git fetch origin "$GIT_BRANCH:refs/remotes/origin/$GIT_BRANCH" || exit 1
GIT_COMMIT=$(git rev-parse "origin/$GIT_BRANCH^{commit}") || exit 1
for path in \
  "$FLYWHEEL_CONFIG" \
  scripts/data.py \
  scripts/eval.py \
  scripts/rollout.py \
  scripts/flywheel.py \
  scripts/train.py \
  scripts/dataset.py \
  scripts/train_core/dataloader.py \
  scripts/train_core/engine.py \
  scripts/train_core/recipes/vision_mlp.py \
  scripts/models/vision_mlp.py \
  .opencode/commands/scripts/benchmark-vision-mlp/analyze.py \
  .opencode/commands/scripts/benchmark-vision-mlp/benchmark_bootstrap.py \
  .opencode/commands/scripts/benchmark-vision-mlp/benchmark_training.py \
  .opencode/commands/scripts/benchmark-vision-mlp/benchmark_worker.py \
  .opencode/commands/scripts/benchmark-vision-mlp/capture_hardware.py \
  .opencode/commands/scripts/benchmark-vision-mlp/collect_data.sh \
  .opencode/commands/scripts/benchmark-vision-mlp/common.py \
  .opencode/commands/scripts/benchmark-vision-mlp/controller.sh \
  .opencode/commands/scripts/benchmark-vision-mlp/create_contract.py \
  .opencode/commands/scripts/benchmark-vision-mlp/initialize_metadata.py \
  .opencode/commands/scripts/benchmark-vision-mlp/run_trial.sh \
  .opencode/commands/scripts/benchmark-vision-mlp/validate_checkpoint.py \
  .opencode/commands/scripts/benchmark-vision-mlp/validate_dataset.py \
  .opencode/commands/scripts/benchmark-vision-mlp/validate_result.py \
  .opencode/commands/scripts/benchmark-vision-mlp/validate_telemetry.py \
  .opencode/commands/scripts/benchmark-vision-mlp/verify_scripts.py; do
  git cat-file -e "$GIT_COMMIT:$path" || {
    printf 'ERROR: %s is missing at %s\n' "$path" "$GIT_COMMIT" >&2
    exit 1
  }
done
REMOTE_CONFIG=$(git show "$GIT_COMMIT:$FLYWHEEL_CONFIG") || exit 1
printf 'Branch: %s\nCommit: %s\nConfig: %s\n\n%s\n' \
  "$GIT_BRANCH" "$GIT_COMMIT" "$FLYWHEEL_CONFIG" "$REMOTE_CONFIG"
CONFIG_SNAPSHOT=$(mktemp)
git show "$GIT_COMMIT:$FLYWHEEL_CONFIG" >"$CONFIG_SNAPSHOT"
CONFIG_VALUES=$(uv run python -c '
import json, sys, yaml
required = (
    "arch", "batch_size", "global_seed", "num_expert_episodes", "max_steps",
    "train_capture_hz", "eval_episodes", "eval_max_steps", "eval_capture_hz",
    "dagger_episodes", "rollout_max_steps", "intervention_threshold",
    "intervention_steps", "dataloader_workers", "persistent_workers", "workers",
)
config = yaml.safe_load(open(sys.argv[1]))
missing = [key for key in required if key not in config]
if missing:
    raise SystemExit(f"missing config keys: {missing}")
if config["arch"] != "vision_mlp" or config["num_expert_episodes"] != 100:
    raise SystemExit("benchmark requires vision_mlp and 100 expert episodes")
print(json.dumps({key: config[key] for key in required}, indent=2, sort_keys=True))
' "$CONFIG_SNAPSHOT")
rm -f "$CONFIG_SNAPSHOT"
printf '%s\n' "$CONFIG_VALUES"
GLOBAL_SEED=$(printf '%s\n' "$CONFIG_VALUES" | jq -r .global_seed)
export GLOBAL_SEED
RESULT_PREFIX=benchmark-vision-mlp-seed${GLOBAL_SEED}-${BENCHMARK_STAMP}
LOCAL_RESULT_DIR=runs/benchmarks/${RESULT_PREFIX}
```

Parse `REMOTE_CONFIG` with `uv run python`. Require `arch == "vision_mlp"` and
`num_expert_episodes == 100`. Resolve batch size, root seed, expert episodes,
maximum expert steps, training capture rate, evaluation episodes, steps and
capture rate, DAgger episodes and steps, intervention threshold, and
intervention steps from this exact remote config.
Stop on a missing required value. Do not use the local working-tree config.

Before provisioning, display and ask the user to confirm:

- branch, commit, config path, and exact config contents;
- fixed batch size and all fixed workload controls;
- seven training configurations and 21 measured training trials;
- one bootstrap checkpoint and 12 worker trials;
- one fixed root seed and one fixed 100-episode dataset;
- sequential execution;
- estimated 15-25 minute post-setup duration and estimated cost range;
- exact hardware search and acceptance criteria below;
- that no full flywheel or policy-quality comparison will run;
- that a new instance will be provisioned and all existing IDs are protected.

Do not provision until confirmation is explicit.

## Prerequisites and secrets

Check these on the development machine:

| Requirement | Check |
|---|---|
| Vast.ai CLI | `which vastai` |
| `VAST_API_KEY` | set in `.env` |
| `tmux` | `which tmux` |
| `flock` | `which flock` |
| `jq` | `which jq` |

Load only the Vast token without shell tracing:

```bash
. ./.env
unset S3_BUCKET S3_PREFIX S3_ENDPOINT_URL AWS_REGION
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
test -n "${VAST_API_KEY:-}" || {
  printf '%s\n' 'ERROR: VAST_API_KEY is missing from .env' >&2
  exit 1
}
export VAST_API_KEY
```

Do not require, export, transfer, or use S3 configuration or AWS credentials.
This workflow creates no S3 backup. Unset `VAST_API_KEY` after provisioning and SSH endpoint
resolution. Reload it only for a later Vast status or confirmed destroy command.

Run `vastai show instances --raw` before searching. List every existing
instance with ID, label, status, GPU, and price. Record all IDs as protected.
Never SSH into or issue a mutating Vast command for a protected ID.

## Stage 0 - Provision a separate benchmark instance

### 0.1 Search offers

Use `SEARCH_EFFECTIVE_VCPUS=24`. Use this exact query without weakening or
omitting any condition:

```bash
OFFERS_JSON=$(vastai search offers \
  'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 gpu_ram>=24 gpu_max_power>=400 compute_cap>=890 total_flops>=80 cpu_cores_effective>=24 cpu_ram>=64 disk_bw>=1000 pci_gen>=4 pcie_bw>=20 inet_down>=500 inet_up>=200 reliability>=0.99 rentable=true verification=verified gpu_display_active=false' \
  --order dph_total+ \
  --raw) || exit 1
printf '%s\n' "$OFFERS_JSON"
```

The query requires exactly one full RTX 4090 with approximately 24 GB VRAM, at
least 400 W power, Ada compute capability 8.9, at least 80 TFLOPS, 64 GB RAM,
1000 MB/s advertised disk bandwidth, PCIe Gen4 with at least 20 GB/s measured
bandwidth, 500/200 Mb/s networking, reliability at least 0.99, Vast
verification, rentability, and no active display workload.

Show exactly these columns:

| Offer ID | CPU model | Effective vCPUs | RAM | Disk MB/s | PCIe GB/s | GPU power | Down/Up Mb/s | Reliability | $/hr | Location |

Rank eligible offers by:

1. CPU family: EPYC 9005, EPYC 9004, Threadripper 7000, EPYC 7003, modern
   Ryzen 7000/9000;
2. lowest price;
3. highest disk bandwidth;
4. highest reliability.

Reject EPYC 7001/7002. Other CPU model names are ranking hints only because the
advertised model may describe the whole host rather than allocated cores. Do
not use `cpu_ghz` as a criterion.

If no offer passes, show the failed filters and ask whether to wait or
explicitly relax named criteria. Never silently weaken a filter. Ask the user
to confirm a priority-ordered shortlist of three to five current offers.

### 0.2 Create exactly one instance

Try confirmed offers in order. Reconcile the exact unique label after every
attempt; CLI text parsing alone is not authoritative:

```bash
CREATED=false
for id in "${OFFER_IDS[@]}"; do
  existing=$(vastai show instances --raw | jq \
    --arg label "$INSTANCE_LABEL" '[.[] | select(.label == $label)] | length')
  test "$existing" -eq 0 || {
    printf 'ERROR: Label %s already exists before create\n' "$INSTANCE_LABEL" >&2
    exit 1
  }

  set +e
  output=$(vastai create instance "$id" \
    --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
    --disk 100 --ssh --direct --label "$INSTANCE_LABEL" \
    --cancel-unavail 2>&1)
  create_status=$?
  set -e

  matches='[]'
  count=0
  for poll in $(seq 1 10); do
    matches=$(vastai show instances --raw | jq --arg label "$INSTANCE_LABEL" \
      '[.[] | select(.label == $label)]')
    count=$(printf '%s\n' "$matches" | jq 'length')
    test "$count" -ne 0 && break
    sleep 3
  done
  if test "$count" -eq 1; then
    INSTANCE_ID=$(printf '%s\n' "$matches" | jq -r '.[0].id')
    INSTANCE_CREATED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    SELECTED_OFFER_JSON=$(printf '%s\n' "$OFFERS_JSON" | jq -c \
      --arg id "$id" '.[] | select((.id | tostring) == $id)')
    test -n "$SELECTED_OFFER_JSON"
    CREATED=true
    break
  fi
  if test "$count" -gt 1; then
    printf '%s\n' "$matches" >&2
    printf 'ERROR: Multiple contracts were created with label %s\n' \
      "$INSTANCE_LABEL" >&2
    exit 1
  fi
  if test "$create_status" -ne 0 && printf '%s\n' "$output" | \
      grep -Eqi 'unavail|not available|no longer available'; then
    continue
  fi
  printf '%s\n' "$output" >&2
  printf 'ERROR: Ambiguous create result for offer %s; refusing another attempt\n' \
    "$id" >&2
  exit 1
done
test "$CREATED" = true || {
  printf '%s\n' 'ERROR: Could not create any confirmed offer' >&2
  exit 1
}
```

Confirm `INSTANCE_ID` is numeric and not protected. If multiple exact-label
contracts exist, stop and list them. Protected IDs are never cleanup candidates.
Ask before destroying any newly created duplicate.

### 0.3 Wait for SSH and pin the host key

Poll `vastai show instance "$INSTANCE_ID" --raw` for `actual_status=running`,
up to 30 attempts ten seconds apart. Resolve the current endpoint only after it
runs:

```bash
SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
HOST=$(printf '%s\n' "$SSH_URL" | sed 's/.*@//;s/:.*//')
PORT=$(printf '%s\n' "$SSH_URL" | sed 's/.*://')
test -n "$HOST" && test -n "$PORT"
```

Probe up to 12 times with `StrictHostKeyChecking=accept-new`, `BatchMode=yes`,
and a ten-second timeout. After first success, require a pin from
`ssh-keygen -F "[$HOST]:$PORT"`. Every later SSH and SCP command uses
`StrictHostKeyChecking=yes`. Stop on a changed key. Then unset
`VAST_API_KEY` locally.

If running or SSH readiness fails, report ID, status, endpoint, and attempts.
Ask before destroying and returning to a fresh search.

## Stage 1 - Hardware acceptance

Treat the rental as provisional. Before cloning, run:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" '
  set -e
  printf "%s\n" "=== CPU topology ==="
  lscpu
  lscpu -e=CPU,CORE,SOCKET,ONLINE
  printf "%s\n" "=== Physical cores ==="
  lscpu -p=CORE,SOCKET | grep -v "^#" | sort -u | wc -l
  printf "%s\n" "=== Logical CPUs ==="
  nproc
  grep "^Cpus_allowed_list:" /proc/self/status
  printf "%s\n" "=== Memory ==="
  free -b
  if test -r /sys/fs/cgroup/memory.max; then
    printf "%s\n" "=== Cgroup memory limit ==="
    cat /sys/fs/cgroup/memory.max
  elif test -r /sys/fs/cgroup/memory/memory.limit_in_bytes; then
    printf "%s\n" "=== Cgroup memory limit ==="
    cat /sys/fs/cgroup/memory/memory.limit_in_bytes
  else
    printf "%s\n" "No readable memory limit file"
  fi
  printf "%s\n" "=== CPU quota ==="
  if test -r /sys/fs/cgroup/cpu.max; then
    cat /sys/fs/cgroup/cpu.max
  elif test -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us; then
    cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
    cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  else
    printf "%s\n" "No readable CPU quota file"
  fi
  printf "%s\n" "=== Filesystems ==="
  df -hT /workspace /
  printf "%s\n" "=== GPU and PCIe ==="
  nvidia-smi --query-gpu=name,memory.total,power.limit,power.default_limit,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv
'
```

Apply every requirement:

| Check | Requirement |
|---|---|
| Allowed physical cores | at least 24 unique online `(CORE, SOCKET)` pairs |
| CPU generation | Zen 3 or newer; reject EPYC 7001/7002 |
| SMT | prefer one thread per core; passing physical count is mandatory |
| CPU quota | at least 90% of advertised effective vCPUs |
| RAM | at least 64 GB allocated and consistent with the offer |
| GPU | exactly one RTX 4090 with approximately 24 GB VRAM |
| GPU power | at least 400 W |
| PCIe | Gen4 x16 capability and offer `pcie_bw >= 20` GB/s |
| Throttling | thermal and power-brake slowdown inactive |

Count physical cores only by intersecting online CPU IDs from
`lscpu -e=CPU,CORE,SOCKET,ONLINE` with `Cpus_allowed_list`, then counting unique
`(CORE, SOCKET)` pairs. The host-wide `lscpu -p` count is diagnostic only. A
12-core/24-thread CPU fails even if `nproc` reports 24. Fail if the allowed
cpuset cannot be established.

For cgroup v2 memory, `memory.max=max` means unlimited; otherwise enforce its
byte value. For cgroup v1, use `memory.limit_in_bytes` unless it is an effectively
unlimited sentinel. Enforce 64 GB against the finite cgroup limit when present,
not `free` alone. If unlimited, use visible memory and report that fact.

For cgroup v2 CPU, divide finite `cpu.max` quota by period; `max PERIOD` means
unlimited. For cgroup v1, divide `cpu.cfs_quota_us` by
`cpu.cfs_period_us`. A finite result from 90% through 100% of advertised
effective vCPUs passes but must be reported. A lower or unreadable allocation
fails when no other evidence establishes it.

Compare observed CPU model, logical CPUs, RAM, GPU, power, and PCIe capability
to the raw offer and list every mismatch. Accept disk and network throughput
from offer measurements. `df` verifies placement and capacity only. Current
PCIe generation may downshift while idle; maximum capability must be Gen4 x16
and the offer measurement must pass. Recheck under CUDA load if restriction is
suspected. Idle P2 or low idle utilization alone is not failure.

The diagnostics above are for review. After the pinned tools are materialized
in Stage 2, make acceptance executable and authoritative. Save the selected raw
offer object as `$CONTROL_DIR/offer.json`, excluding credentials and endpoint
fields, then run:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "/root/.local/bin/uv run --directory /workspace/toy-pickplace python \
     '$CONTROL_DIR/tools/capture_hardware.py' \
     --offer '$CONTROL_DIR/offer.json' \
     --output '$CONTROL_DIR/hardware.json' && \
   test \"\$(jq -r .success '$CONTROL_DIR/hardware.json')\" = true"
```

Do not launch the controller unless this command succeeds. Copy
`hardware.json` back for final launch confirmation.

On failure, stop before setup, list every failed rule, and ask whether to destroy
the provisional instance and restart with a fresh offer list. Do not relax a
rule without explicit user approval naming that rule.

## Stage 2 - Clone, automate acceptance, and prepare

After the manual diagnostics pass, clone only to materialize the pinned
acceptance and benchmark tools. This is not benchmark launch approval. Clone and
detach at the confirmed commit:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "git clone --branch '$GIT_BRANCH' --single-branch \
     https://github.com/skpro19/toy-pickplace.git /workspace/toy-pickplace && \
   cd /workspace/toy-pickplace && \
   git checkout --detach '$GIT_COMMIT' && \
   test \"\$(git rev-parse HEAD)\" = '$GIT_COMMIT' && \
   test -f '$FLYWHEEL_CONFIG' && \
   apt-get update -qq && apt-get install -y -qq \
     libgl1-mesa-glx libglib2.0-0 libegl1-mesa libgles2-mesa libglfw3 \
      sysstat time tmux util-linux jq && \
   command -v tmux >/dev/null && command -v flock >/dev/null && \
   curl -LsSf https://astral.sh/uv/install.sh | sh && \
   /root/.local/bin/uv sync --locked --directory /workspace/toy-pickplace && \
   MUJOCO_GL=egl /root/.local/bin/uv run --directory \
     /workspace/toy-pickplace python -c \
     'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import mujoco, glfw; print(mujoco.__version__, glfw.__version__)'"
```

Create:

```text
CONTROL_DIR=/workspace/toy-pickplace/.benchmark/${RESULT_PREFIX}
CONTROL_DIR/tools/
CONTROL_DIR/data/expert/
CONTROL_DIR/plans/
CONTROL_DIR/logs/
CONTROL_DIR/telemetry/
CONTROL_DIR/time/
CONTROL_DIR/results/training/
CONTROL_DIR/results/workers/
CONTROL_DIR/status/history/
CONTROL_DIR/state/
```

Create atomic `metadata.json` containing contract, commit, config, fixed values,
instance ID, non-secret offer fields, accepted hardware, plan, root and derived
seeds, UTC start, and software fingerprint. Include image, OS, kernel, NVIDIA
driver, CUDA, cuDNN, Python, PyTorch, MuJoCo, GLFW, uv, `uv.lock` SHA-256,
resolved packages, CPU governor, and Torch deterministic/TF32 backend flags.
Never store credentials or SSH endpoints in benchmark artifacts.

Materialize scripts from the detached remote Git commit, not from local files.
Do not edit, commit, or push the remote checkout:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "set -e; cd /workspace/toy-pickplace; \
   test \"\$(git rev-parse HEAD)\" = '$GIT_COMMIT'; \
   mkdir -p '$CONTROL_DIR/tools'; \
   git ls-tree -r --name-only '$GIT_COMMIT' \
     .opencode/commands/scripts/benchmark-vision-mlp | while read -r path; do \
       name=\$(basename \"\$path\"); \
       git show '$GIT_COMMIT':\"\$path\" >'$CONTROL_DIR/tools/'\"\$name\"; \
     done; \
   chmod +x '$CONTROL_DIR/tools/'*.sh && \
    /root/.local/bin/uv run --directory /workspace/toy-pickplace \
      python -m py_compile '$CONTROL_DIR/tools/'*.py"
```

Generate the expected manifest independently from Git blobs, then verify the
materialized directory has exactly those files and bytes:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "/root/.local/bin/uv run --directory /workspace/toy-pickplace python \
     '$CONTROL_DIR/tools/verify_scripts.py' \
     --tools-dir '$CONTROL_DIR/tools' \
     --project-root /workspace/toy-pickplace \
     --commit '$GIT_COMMIT' \
     --source-prefix .opencode/commands/scripts/benchmark-vision-mlp \
     --write-manifest '$CONTROL_DIR/script-manifest.json'"
```

A materialization interrupted before digest verification is incomplete and
must not launch a controller. Create a sanitized `$CONTROL_DIR/instance.json`
with `id`, `label`, `created_utc`, `price_per_hour`, and non-secret accepted
offer fields. Do not include `ssh_host`, `ssh_port`, public IPs, tokens, or
credentials. Run the authoritative `capture_hardware.py` gate from Stage 1 now.
Only after it passes, ask for final launch confirmation showing hardware,
commit, fixed batch size, 21 training trials, 12 worker trials, estimated
duration, and cost.

Use the selected object from the saved raw search response, not reconstructed
offer values. Materialize and transfer only the fields used by acceptance:

```bash
offer_file=$(mktemp)
instance_file=$(mktemp)
printf '%s\n' "$SELECTED_OFFER_JSON" | jq '{
  id, gpu_name, gpu_frac, num_gpus, gpu_ram, gpu_max_power, compute_cap,
  total_flops, cpu_name, cpu_cores_effective, cpu_ram, disk_bw, pci_gen,
  pcie_bw, inet_down, inet_up, reliability, rentable, verification,
  gpu_display_active, dph_total, geolocation
}' >"$offer_file"
jq -n \
  --argjson id "$INSTANCE_ID" \
  --arg label "$INSTANCE_LABEL" \
  --arg created_utc "$INSTANCE_CREATED_UTC" \
  --argjson price_per_hour "$(jq .dph_total "$offer_file")" \
  '{id:$id,label:$label,created_utc:$created_utc,price_per_hour:$price_per_hour}' \
  >"$instance_file"
scp -o StrictHostKeyChecking=yes -o BatchMode=yes -P "$PORT" \
  "$offer_file" "root@$HOST:$CONTROL_DIR/offer.json"
scp -o StrictHostKeyChecking=yes -o BatchMode=yes -P "$PORT" \
  "$instance_file" "root@$HOST:$CONTROL_DIR/instance.json"
rm -f "$offer_file" "$instance_file"
```

## Stage 3 - Collect fixed expert data

`create_contract.py` derives seeds on the remote host from the confirmed code
and root seed. The controller reads those exact values and collects exactly 100
image episodes once through `collect_data.sh`. Do not run a separate foreground
collection command. Tmux owns collection and every later phase.

Use `.opencode/commands/scripts/benchmark-vision-mlp/validate_dataset.py`. Fail
unless exactly 100 `.npz` files exist, every
file contains `obs`, `actions`, and `img_obs`, first dimensions match, images
are `uint8` with shape `(T, 64, 64, 3)`, no episode is empty, and total frames
are positive. Atomically write a manifest with filename, bytes, SHA-256, frame
count, shapes, totals, and aggregate digest. Revalidate immediately before the
first measured trial. Never regenerate or append after measurement starts.

No remote artifact backup is required. If the instance is lost, provision and
accept a new host, recollect the deterministic dataset, verify the same manifest
digest, and restart all timings. Never combine results from physical hosts.

## Stage 4 - Materialize benchmark tools

The canonical source directory must contain this complete set:

| Script | Responsibility |
|---|---|
| `common.py` | Atomic JSON and text writes |
| `capture_hardware.py` | Executable cpuset, cgroup, GPU, and offer acceptance |
| `create_contract.py` | Exact config parsing, seed derivation, and immutable contract |
| `verify_scripts.py` | Pinned script-manifest verification |
| `collect_data.sh` | Durable expert collection with truthful status |
| `validate_dataset.py` | Exact episode/schema validation and manifest |
| `benchmark_training.py` | Production-path training throughput measurement |
| `benchmark_bootstrap.py` | Bootstrap training and exact checkpoint discovery |
| `validate_checkpoint.py` | Runtime loading, epoch, normalization, and digest validation |
| `benchmark_worker.py` | Structured evaluation and direct-output DAgger trials |
| `validate_result.py` | Independent per-trial artifact validation |
| `validate_telemetry.py` | Independent GPU, process-tree, host, and time validation |
| `run_trial.sh` | Cooling, telemetry, timing, status, and command lifecycle |
| `controller.sh` | Sequential plan execution and stop-on-first-failure behavior |
| `initialize_metadata.py` | Merge the run contract with manifest/software facts |
| `analyze.py` | Cross-artifact validation, stability checks, ranking, and reports |

These scripts are the implementation. The prose below defines their contract
and invocation, not an invitation to create alternative transient versions.

### 4.1 Training harness

Use the transferred `CONTROL_DIR/tools/benchmark_training.py`. It accepts:

```text
--data-dir PATH
--batch-size INT
--dataloader-workers INT
--persistent-workers | --no-persistent-workers
--sample-seed INT
--model-seed INT
--warmup-epochs INT
--measured-epochs INT
--output PATH
```

It must:

1. reject persistence with zero workers;
2. require CUDA and an RTX 4090;
3. import production code from the confirmed checkout;
4. build the expert-only `PickPlaceVisionDataset` with production normalization
   and action space;
5. construct `WeightedRandomSampler` with production weights, sample count,
   replacement, and seeded generator;
6. construct DataLoader with fixed batch size, candidate worker count,
   production `pin_memory`, and candidate persistence;
7. construct a fresh seeded production VisionMLP, Adam optimizer, and losses;
8. execute production `run_epoch` for one warm-up and five measured epochs;
9. synchronize CUDA around measured epochs and reset peak memory after warm-up;
10. atomically write inputs, commit, device, frames, samples/epoch,
    batches/epoch, epoch times, total time, samples/s, losses, measured-window
    timestamps, VRAM peaks, and success;
11. atomically write failure type and message and exit nonzero on error.

Run one unrecorded preflight at fixed batch size, workers 0, non-persistent, one
warm-up and one measured epoch. Require valid schema, finite positive timing,
expected sample count, nonzero CUDA memory, and finite losses.

### 4.2 Trial wrapper

Use the transferred `run_trial.sh` with
`CONTROL_DIR KIND TRIAL_ID GENERATION ATTEMPT RESULT -- COMMAND...`. It must:

1. atomically write `running GENERATION ATTEMPT`;
2. wait up to two minutes for GPU utilization below 5% and temperature below
   55 C for five consecutive one-second samples;
3. start GPU, `pidstat`, and `vmstat` samplers and retain their exact PIDs;
4. trap exit and terminate only those sampler PIDs;
5. invoke the requested benchmark through `/usr/bin/time -v`;
6. stream output to an attempt-specific durable log;
7. verify result, time, and telemetry files;
8. run an independent artifact validator after the command exits zero;
9. atomically write `succeeded 0 GENERATION ATTEMPT` only when both the command
   and artifact validator exit zero, otherwise write
   `failed EXIT_CODE GENERATION ATTEMPT`;
10. never retry automatically.

Never derive success from `/usr/bin/time`, `tee`, elapsed time, or the presence
of a log. Preserve the producer status with `PIPESTATUS[0]` when a pipeline is
unavoidable. A result is successful only if all of these agree:

```text
wrapped command exit code == 0
independent result validator exit code == 0
status text == "succeeded 0 GENERATION ATTEMPT"
```

The validator must reject a missing result, a self-reported `success` other
than JSON `true`, mismatched trial/generation/attempt, non-finite timings,
incorrect fixed inputs, an incorrect episode count, or a missing checkpoint.

### 4.3 Analysis tool

Use the transferred `analyze.py`. It validates plans, statuses, result schemas,
fixed values, commit, dataset digest, and telemetry. It computes ranking and
stability metrics and writes reports atomically.

Before launch, copy metadata, manifest, tools, and plans to
`LOCAL_RESULT_DIR/control` and require successful local readback. This preserves
the benchmark definition, not partial timing results.

## Stage 5 - Plans and durable execution

### 5.1 Training plan

Create `training.plan`:

```text
order|repetition|batch_size|dataloader_workers|persistent_workers|trial_id
```

Every row uses the same confirmed batch size. Order seven configurations as:

- repetition 1: `dw0-p0,dw2-p0,dw2-p1,dw4-p0,dw4-p1,dw8-p0,dw8-p1`;
- repetition 2: exact reverse;
- repetition 3: rotate repetition 1 left by three positions.

Validate exactly 21 rows, 21 IDs, three appearances per configuration, one
batch size, and no `dw0-p1`.

### 5.2 Bootstrap

After training trials, create a representative checkpoint with:

```bash
/root/.local/bin/uv run python scripts/train.py \
  --arch vision_mlp \
  --base vision-benchmark-bootstrap \
  --checkpoint_root "$CONTROL_DIR/bootstrap/checkpoints" \
  --log_root "$CONTROL_DIR/bootstrap/runs" \
  --epochs 120 \
  --batch-size "$FIXED_BATCH_SIZE" \
  --npz "$CONTROL_DIR/data/expert" \
  --sample-seed "$TRAIN_SEED" \
  --eval-interval 120 \
  --eval-seed "$EVAL_SEED" \
  --eval-episodes 1 \
  --eval-max-steps "$EVAL_MAX_STEPS" \
  --eval-capture-hz "$EVAL_CAPTURE_HZ" \
  --eval-workers 12 \
  --dataloader-workers 0 \
  --early-stop-patience 0
```

Run through the common wrapper. Validate the checkpoint through the production
policy runtime and record its SHA-256. Its score is irrelevant; it provides a
representative trained architecture for simulator timing.

`scripts/train.py` creates a numbered run directory below `--checkpoint_root`.
The checkpoint is therefore not `$CONTROL_DIR/bootstrap/checkpoints/last.pt`.
After successful training, require exactly one newly-created immediate child
directory and publish its resolved checkpoint path atomically:

```bash
mapfile -t bootstrap_run_dirs < <(
  find "$CONTROL_DIR/bootstrap/checkpoints" \
    -mindepth 1 -maxdepth 1 -type d -print
)
test "${#bootstrap_run_dirs[@]}" -eq 1
BOOTSTRAP_RUN_DIR=${bootstrap_run_dirs[0]}
BOOTSTRAP_CHECKPOINT=$(realpath "$BOOTSTRAP_RUN_DIR/last.pt")
test -s "$BOOTSTRAP_CHECKPOINT"
test -s "$BOOTSTRAP_RUN_DIR/best.pt"
```

The versioned `validate_checkpoint.py` loads this exact path through
`policy_runtimes.registry.load_runtime`, requires architecture `vision_mlp`,
epoch `120`, finite normalization arrays, and successful CPU runtime setup,
then atomically writes `bootstrap/checkpoint.json` containing the absolute path
and SHA-256. Every worker trial must read the path and digest from that JSON.
Never reconstruct, guess, or independently glob the checkpoint path in a
worker command.

### 5.3 Worker plans

Create evaluation and DAgger plans with orders `6,12`, `12,6`, and `6,12` for
repetitions 1-3.

Evaluation shape:

```bash
MUJOCO_GL=egl /root/.local/bin/uv run python scripts/eval.py \
  --ckpt_path "$BOOTSTRAP_CHECKPOINT" \
  --seed "$EVAL_SEED" \
  --max_steps "$EVAL_MAX_STEPS" \
  --capture-hz "$EVAL_CAPTURE_HZ" \
  --episodes "$EVAL_EPISODES" \
  --workers "$WORKERS"
```

DAgger shape:

```bash
MUJOCO_GL=egl /root/.local/bin/uv run python scripts/rollout.py \
  --model "$BOOTSTRAP_CHECKPOINT" \
  --seed "$DAGGER_SEED" \
  --episodes "$DAGGER_EPISODES" \
  --max-steps "$ROLLOUT_MAX_STEPS" \
  --capture-hz "$TRAIN_CAPTURE_HZ" \
  --train-npz-dir "$CONTROL_DIR/data/expert" \
  --dagger \
  --dagger-mode threshold \
  --intervention-threshold "$INTERVENTION_THRESHOLD" \
  --intervention-steps "$INTERVENTION_STEPS" \
  --dagger-dir "$UNIQUE_DAGGER_OUTPUT" \
  --no-log-rollout \
  --headless \
  --workers "$WORKERS"
```

Read each confirmed parser's `--help` before launch and adjust only spelling
proven different. Preserve semantics. Validate evaluation metrics and exactly
the configured DAgger episode count.

The rollout CLI creates a timestamped child under `--dagger-dir`. After the
command exits, recursively discover directories containing `.npz` files,
require exactly one such leaf directory, and require exactly
`$DAGGER_EPISODES` files there. Do not incorrectly check only the immediate
`$UNIQUE_DAGGER_OUTPUT` directory. The versioned Python worker harness instead
calls `rollout(..., create_dagger_subdir=False)` and validates the exact output
directory; this is preferred because it removes output-path discovery.

For evaluation, do not accept human-readable log output as evidence. The
versioned Python worker harness calls `eval.score_ckpt` and atomically
writes structured metrics. Require the score list length to equal
`$EVAL_EPISODES` and all metrics to be finite before success.

### 5.4 Controller

First create the immutable contract from the exact detached config and accepted
artifacts:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "/root/.local/bin/uv run --directory /workspace/toy-pickplace python \
     '$CONTROL_DIR/tools/create_contract.py' \
     --project-root /workspace/toy-pickplace \
     --branch '$GIT_BRANCH' \
     --commit '$GIT_COMMIT' \
     --config '/workspace/toy-pickplace/$FLYWHEEL_CONFIG' \
     --instance '$CONTROL_DIR/instance.json' \
     --hardware '$CONTROL_DIR/hardware.json' \
     --script-manifest '$CONTROL_DIR/script-manifest.json' \
     --output '$CONTROL_DIR/contract.json'"
```

Read back `contract.json`, require its commit, config digest, fixed controls,
seeds, hardware, and script manifest to match the confirmed values, then launch
the transferred `controller.sh` in tmux session `vision-benchmark-controller`:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "set -e; test ! -e '$CONTROL_DIR/state/completed'; \
   test -z \"\$(tmux list-sessions -F '#{session_name}' 2>/dev/null | \
     grep -Fx vision-benchmark-controller || true)\"; \
   tmux new-session -d -s vision-benchmark-controller \
     \"env PROJECT_ROOT=/workspace/toy-pickplace \
       CONTROL_DIR='$CONTROL_DIR' CONTRACT_JSON='$CONTROL_DIR/contract.json' \
       bash '$CONTROL_DIR/tools/controller.sh' \
       >'$CONTROL_DIR/logs/controller.log' 2>&1\""
```

Never edit run-specific values into the versioned source. The controller must:

1. acquire an exclusive `flock` and refuse another controller session or lock;
2. create an atomic unique controller-generation marker;
3. treat a stale nonterminal status from another generation as `failed 125`;
4. process 21 training trials sequentially;
5. run and validate the bootstrap once;
6. process six evaluation and six DAgger trials sequentially;
7. run required high-spread extra repetitions only after initial analysis;
8. skip only validated `succeeded 0` results;
9. stop at first failure and atomically write trial ID, phase, code, and log to
   `state/failed`;
10. never auto-retry or auto-skip;
11. write `state/completed` only after analysis validates every required result.

Use one child process at a time. Do not create concurrent benchmark cells.
After launch, verify tmux and a matching generation in `state/started`. The
agent may disconnect after reporting recovery commands; tmux owns execution.

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux has-session -t vision-benchmark-controller && \
   test -s '$CONTROL_DIR/state/started' && \
   cat '$CONTROL_DIR/state/started'"
```

## Stage 6 - Results

After `state/completed`, download metadata, manifest, plans, tools, statuses,
logs, time files, telemetry, raw JSON, and generated reports with strict-host-key
SCP into `LOCAL_RESULT_DIR`. Re-run analysis locally and require identical
summary JSON.

```bash
mkdir -p "$LOCAL_RESULT_DIR"
scp -r -o StrictHostKeyChecking=yes -o BatchMode=yes -P "$PORT" \
  "root@$HOST:$CONTROL_DIR/." "$LOCAL_RESULT_DIR/"
remote_summary_sha256=$(sed -n 's/^summary_sha256=//p' \
  "$LOCAL_RESULT_DIR/state/completed")
test "$(sha256sum "$LOCAL_RESULT_DIR/benchmark-summary.json" | cut -d ' ' -f1)" \
  = "$remote_summary_sha256"
remote_summary_copy=$(mktemp)
cp "$LOCAL_RESULT_DIR/benchmark-summary.json" "$remote_summary_copy"
analysis_parent=$(mktemp -d)
analysis_worktree="$analysis_parent/checkout"
git worktree add --detach "$analysis_worktree" "$GIT_COMMIT"
BENCHMARK_PROJECT_ROOT="$analysis_worktree" uv run python \
  "$LOCAL_RESULT_DIR/tools/analyze.py" \
  --control-dir "$LOCAL_RESULT_DIR" \
  --output-dir "$LOCAL_RESULT_DIR" \
  --require-complete
cmp -s "$remote_summary_copy" "$LOCAL_RESULT_DIR/benchmark-summary.json"
git worktree remove "$analysis_worktree"
rm -rf "$analysis_parent"
rm -f "$remote_summary_copy"
```

Generate exactly:

```text
benchmark-summary.md
benchmark-summary.json
training-throughput.csv
worker-throughput.csv
```

The Markdown report must include:

1. branch, commit, exact config, fixed batch size, seeds, dataset digest,
   instance ID, price, and accepted hardware;
2. all 21 initial training trials and any stability reruns;
3. production-supported DataLoader ranking and recommendation;
4. separate persistent-worker results;
5. all evaluation and DAgger worker trials with paired ranking;
6. recommended `workers` value;
7. GPU utilization, power, VRAM, CPU, RAM, and I/O evidence;
8. speedup versus config baseline `dataloader_workers` and `workers`;
9. failures, retries, skips, missing telemetry, and limitations;
10. elapsed instance time and estimated benchmark cost;
11. an explicit statement that no batch-size or policy-quality conclusion was
    measured;
12. the command to run `/ablate-flywheel-params batch_size` if a later quality
    ablation is desired.

Do not automatically edit config files or runbooks unless the user explicitly
authorizes it. Present recommendations and ask whether the user wants the
recommended production settings applied separately.

## Failure and recovery

Status files and validated result artifacts are jointly authoritative; neither
one alone is success. Tmux membership, logs, and elapsed wall time are never
success.

On failure, the controller has already stopped. Report phase, trial, exit code,
log, time, telemetry, and result paths, then ask whether to retry or stop. Skips
are not accepted because completion requires the exact plan. Before an approved
retry, require no benchmark process remains, read the four status fields, and
preserve the failed canonical result if present:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "CONTROL_DIR='$CONTROL_DIR' TRIAL_ID='$TRIAL_ID' \
   RESULT_GROUP='$RESULT_GROUP' bash -s" <<'REMOTE_RECOVERY'
read -r state code generation attempt <"$CONTROL_DIR/status/$TRIAL_ID.status"
test "$state" = failed
test "$code" -ne 0
test -n "$generation" && test -n "$attempt"
mkdir -p "$CONTROL_DIR/results/history"
if test "$TRIAL_ID" = expert; then
  mkdir -p "$CONTROL_DIR/data/history"
  mv "$CONTROL_DIR/data/expert" \
    "$CONTROL_DIR/data/history/expert-a$attempt"
  mkdir -p "$CONTROL_DIR/data/expert"
else
  result_path="$CONTROL_DIR/results/$RESULT_GROUP/$TRIAL_ID.json"
  if test -e "$result_path"; then
    mv "$result_path" \
      "$CONTROL_DIR/results/history/$TRIAL_ID-a$attempt.json"
  fi
fi
mv "$CONTROL_DIR/status/$TRIAL_ID.status" \
  "$CONTROL_DIR/status/history/$TRIAL_ID-a$attempt.status"
if test -e "$CONTROL_DIR/state/failed"; then
  mv "$CONTROL_DIR/state/failed" \
    "$CONTROL_DIR/status/history/controller-$TRIAL_ID-a$attempt.failed"
fi
REMOTE_RECOVERY
```

Then relaunch the exact tmux controller command above. Never delete or overwrite
attempt-specific logs, telemetry, time files, bootstrap directories, or DAgger
output directories.

On same-instance resume, resolve the current SSH endpoint again, enforce the
pinned key, validate commit, config, dataset digest, generation, plans,
statuses, and results, then require confirmation before retry or stop. A missing
or changed host key is a hard stop.

If the instance is lost, provision a new accepted host, recollect and validate
the deterministic expert dataset, and restart all timing measurements. Do not
combine benchmark timings from different physical hosts.

## Cleanup

Before offering destruction, verify:

- `state/completed` exists;
- all four local report files exist and local analysis matches remote JSON;
- raw metadata, plans, statuses, logs, time files, and telemetry are local;
- no controller, training, evaluation, rollout, telemetry, or descendant process
  remains in `ps`, tmux, or `nvidia-smi` compute processes;
- no status claims active ownership and no benchmark file is open for writing.

If an orphan exists, report PID, process group, command, generation, GPU
ownership, and outputs. Never kill it implicitly. Require separate confirmation
to terminate that exact benchmark process tree and verify exit.

Then print, but do not execute without explicit confirmation:

```bash
. ./.env
export VAST_API_KEY
vastai destroy instance -y "$INSTANCE_ID"
unset VAST_API_KEY
```

After confirmation, destroy only the benchmark instance, verify it disappeared,
and re-list protected instances to demonstrate they were unchanged.

## Handoff output

At every handoff, print:

- current phase and authoritative state marker;
- branch, commit, config, fixed batch size, seed, and dataset digest;
- instance ID and `vastai ssh-url INSTANCE_ID` retrieval command;
- accepted hardware and hourly price;
- `CONTROL_DIR` and `LOCAL_RESULT_DIR`;
- planned, succeeded, failed, skipped, and pending counts;
- controller and tmux status;
- attach and log-follow commands;
- current recommendation only when supported by completed evidence;
- elapsed time and estimated cost;
- next confirmation or recovery action;
- cleanup command only after local result verification.

## Safety rules

- Never use or disrupt an existing instance.
- Never weaken hardware filters or acceptance gates without explicit approval.
- Never count SMT threads as physical cores.
- Never run benchmark trials concurrently.
- Never change batch size or other experiment hyperparameters.
- Never randomize seeds or regenerate data after timing starts.
- Never run a complete flywheel or make policy-quality claims.
- Never edit or commit the cloned repository for a transient benchmark.
- Never treat persistent workers as supported unless the selected commit exposes
  and validates the production setting.
- Never require, export, transfer, or use S3 configuration or AWS credentials.
- Never store secrets or SSH endpoints in benchmark artifacts. The required
  `known_hosts` pin is the sole endpoint-storage exception.
- Never destroy an instance without listing it and receiving confirmation for
  that exact ID.
