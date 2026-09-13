---
description: Provision one RTX 4090 and run a parameterized flywheel ablation grid
agent: build
---

Run a general-purpose flywheel ablation on one Vast.ai RTX 4090. This command
owns argument validation, grid planning, provisioning, hardware acceptance,
instance tuning, batched launch, checkpoint backup, and local monitoring. It is
self-contained: follow this workflow without consulting another runbook.

## Workflow at a glance

1. Confirm the remote branch and exact config contents.
2. Confirm sweep values, tier-derived tuning, and the complete run plan.
3. Load secrets, select an offer, and provision one instance.
4. Accept or reject the instance using SSH hardware checks.
5. Confirm actual concurrency and instance-side config overrides.
6. Clone, tune, launch the durable controller, and start backup.
7. Monitor controller state, handle failures, and print recovery commands.
8. Optionally run large held-out evaluation on the instance and upload only
   each run's `final_scores.json`.

## Command arguments and grid planning

Requested parameter names are `$1 $2 ... $N`. At least one name is required.

1. Accept a parameter name in its YAML `snake_case` form or CLI `kebab-case`
   form. Normalize CLI names by replacing `-` with `_`.
2. Accept only parameters with a matching `scripts/flywheel.py` CLI argument in
   the selected remote commit. Reject infrastructure-only values (`workers`,
   `dataloader_workers`), reject `global_seed` because every cell must share the
   confirmed root seed, and reject duplicate normalized names. Show the
   supported parameter catalog and stop for an invalid request.

   **Policy: the YAML config on disk is never modified.** All experiment
   parameter overrides must be supplied as CLI flags (`--intervention-threshold`,
   `--dagger-rounds`, etc.) to the flywheel invocation. Infrastructure values
   (`workers`, `dataloader_workers`, `batch_size`) are always left at their
   committed defaults. The instance-side `Batch 2` tuning step (tier-tuning
   `sed` overrides) and its Python YAML assertion are skipped entirely. The
   `workers`/`dataloader_workers` columns in the tier table are informational
   only — the config's committed values are always authoritative.
3. Ask the user to select the Git branch to clone, defaulting to `dev`, and set
   `GIT_BRANCH` to that exact value. Set
   `FLYWHEEL_CONFIG=configs/flywheel/mlp_vision/default_mlp_vision_instance.yaml`. Before
   any grid or infrastructure planning, verify that this config is committed on
   the selected remote branch and load its contents from that branch. Stop if it
   does not exist there; a local-only file is not sufficient. Display the exact
   branch, config path, and loaded config contents, then ask the user to confirm
   this experiment baseline. Do not collect sweep values, select a tier, or
   derive tuning overrides until the baseline is confirmed.

   Validate and load the remote file without changing the local checkout:

   ```bash
   case "$GIT_BRANCH" in
     ""|*[!A-Za-z0-9._/-]*)
       printf '%s\n' 'ERROR: Branch may contain only letters, digits, ., _, /, and -' >&2
       exit 1
       ;;
   esac
   git check-ref-format --branch "$GIT_BRANCH"
   git fetch origin \
     "$GIT_BRANCH:refs/remotes/origin/$GIT_BRANCH" || {
     printf 'ERROR: Could not fetch origin/%s\n' "$GIT_BRANCH" >&2
     exit 1
   }
   GIT_COMMIT=$(git rev-parse "origin/$GIT_BRANCH^{commit}") || exit 1
   REMOTE_CONFIG=$(git show "$GIT_COMMIT:$FLYWHEEL_CONFIG") || {
     printf 'ERROR: %s does not exist on origin/%s\n' \
       "$FLYWHEEL_CONFIG" "$GIT_BRANCH" >&2
     exit 1
   }
   REMOTE_FLYWHEEL=$(git show "$GIT_COMMIT:scripts/flywheel.py") || {
     printf 'ERROR: scripts/flywheel.py does not exist at %s\n' \
       "$GIT_COMMIT" >&2
     exit 1
   }
   printf 'Branch: %s\nCommit: %s\nConfig: %s\n\n%s\n' \
     "$GIT_BRANCH" "$GIT_COMMIT" "$FLYWHEEL_CONFIG" "$REMOTE_CONFIG"
   ```

   Parse defaults and `GLOBAL_SEED` from `REMOTE_CONFIG`, and derive supported
   flags and validation constraints from `REMOTE_FLYWHEEL`. Do not use either
   local working-tree copy for planning.
4. Ask for an exact comma- or newline-separated value list for every requested
   parameter. Display its default resolved from `FLYWHEEL_CONFIG`, its CLI
   flag, and the corresponding argument validation constraints before asking.
   Resolve `GLOBAL_SEED` from the same config.
5. After all sweep value lists are confirmed, ask the user for a comma- or
   newline-separated ordered list of one or more top-level `FLYWHEEL_CONFIG`
   parameter names to include in `BACKUP_PREFIX`. Validate every name against
   `REMOTE_CONFIG`, reject duplicates, unknown names, null values, and
   non-scalar values, and display each resolved prefix value before continuing.
   For a selected swept parameter, its prefix value is its confirmed ordered
   value list; for every other selected parameter, its prefix value is the
   scalar resolved from `REMOTE_CONFIG`. This selection is metadata only and
   never changes experiment overrides or config contents.
6. Let `GRID_CELLS` be the product of all confirmed value-list lengths:
   - one requested parameter is a 1D sweep;
   - two or more requested parameters form a full Cartesian grid, with every
     requested CLI override set on every cell.
7. Require `1 <= GRID_CELLS <= 30`. If the grid has more than 30 cells, show
   its computed size and ask the user to reduce the lists or split the work into
   multiple invocations. Do not provision.
8. Build `RUN_PLAN` in deterministic parameter-list order and value-list order.
   Each row includes:
   - a unique run name;
   - a unique tmux session;
   - all CLI overrides for that cell;
   - its provisional batch number based on the tier's maximum concurrency.
   Final batch numbers are assigned only after Workflow Step 6 computes actual
   concurrency from the accepted host.
9. Include the already-confirmed `GIT_BRANCH`, `GIT_COMMIT`, and
   `FLYWHEEL_CONFIG` for context with the complete `RUN_PLAN`, selected tier,
   search floor, physical-core acceptance gate, proposed tuning overrides,
   planned concurrency range, estimated batch count, and the exact resolved
   `BACKUP_PREFIX`, including its ordered selected parameter names and resolved
   values. Clearly distinguish config-resolved experiment values from
   tier-derived infrastructure overrides. Ask the user to explicitly confirm
   the entire run and infrastructure plan, including the S3 backup prefix name,
   before Step 0. Do not ask them to reconfirm the baseline. Do not reconfirm it
   after provisioning or immediately before cloning; the post-provision launch
   confirmation remains required.

### Supported parameter catalog

Use the parser loaded from the confirmed `GIT_COMMIT` as authoritative. The
common ablation parameters are:

| YAML key | CLI flag | Default | Constraint / suggested values |
|---|---|---:|---|
| `intervention_threshold` | `--intervention-threshold` | `0.1` | `>=0`; suggested `0.05, 0.1, 0.2, 0.3` |
| `dagger_intervention_ratio` | `--dagger-intervention-ratio` | `0.8` | `0..1`; suggested `0.2, 0.5, 0.8, 1.0` |
| `expert_ratio` | `--expert-ratio` | `0.5` | `0 < value < 1`; suggested `0.3, 0.5, 0.7` |
| `dagger_rounds` | `--dagger-rounds` | `10` | integer `>=1`; suggested `5, 10, 20` |
| `intervention_steps` | `--intervention-steps` | `50` | integer `>=1`; suggested `25, 50, 100` |
| `dagger_episodes` | `--dagger-episodes` | `50` | integer `>=1`; suggested `25, 50, 100` |
| `batch_size` | `--batch-size` | `768` | integer `>=1`; suggested `200, 384, 768` |
| `early_stop_patience` | `--early-stop-patience` | `50` | integer `>=0`; suggested `25, 50, 100` |
| `mode` | `--mode` | `mode-b` | `mode-a` or `mode-b` |
| `eval_interval` | `--eval-interval` | `20` | integer `>=1`; suggested `10, 20, 40` |
| `eval_episodes` | `--eval-episodes` | `25` | integer `>=1`; suggested `25, 50` |

The catalog defaults are the current common baseline only. Values loaded from
the confirmed remote `FLYWHEEL_CONFIG` are authoritative and must be shown
instead whenever they differ.

For additional parser-supported experiment parameters, validate against the
confirmed commit's `scripts/flywheel.py` and apply the same value validation it
enforces. Do not permit `workers`, `dataloader_workers`, or `global_seed` as
swept dimensions; those are fixed for the entire grid.

### Naming

For a 1D sweep, use:

```text
abl-{parameter_key}-{value_slug}
```

For a grid, use:

```text
abl-{parameter_key_1}-{value_slug_1}-{parameter_key_2}-{value_slug_2}-...
```

Convert a decimal point in each `value_slug` to `p` and replace other
non-alphanumeric separators with `-`. For general 1D and grid runs, use the
same slug in the tmux name, prefixed with `ablation-`.

Preserve these backward-compatible names for the two established 1D profiles:

| Parameter | Run name | tmux session |
|---|---|---|
| `intervention_threshold` | `abl-intervention_threshold-{value}` | `ablation-{value}` |
| `dagger_intervention_ratio` | `abl-dagger_intervention_ratio-{value}` | `ablation-{value}` |

The command may use the concise `abl-it{T}-dir{R}` / `ablation-it{T}-dir{R}`
form only when the two parameters are exactly `intervention_threshold` and
`dagger_intervention_ratio`; it must still be unique.

For `BACKUP_PREFIX`, preserve the user-selected parameter order. Serialize a
selected swept parameter's confirmed ordered values as a comma-separated list;
serialize every other selected parameter as its resolved scalar config value.
Keep each normalized config key unchanged and convert each serialized value to
a slug by replacing decimal points with `p` and every other non-alphanumeric
separator with `-`. This makes the prefix safe for shell use, S3 object keys,
and the remote control directory.
For example, selecting `mode`, `dagger_rounds`, and swept
`intervention_threshold` with values `0.05, 0.1` produces:

```text
flywheel-mode=mode-b_dagger_rounds=10_intervention_threshold=0p05-0p1-YYYYMMDD-HHMMSS
```

Set:

```text
INSTANCE_LABEL=toy-pickplace-ablation-{parameter-slugs}-{UNIX_TIME_NS}
GLOBAL_SEED=<global_seed resolved from the selected flywheel config>
BACKUP_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_PREFIX=flywheel-{selected_parameter_key}={resolved_value_slug}_{selected_parameter_key}={resolved_value_slug}-...-${BACKUP_TIMESTAMP}
```

Set `UNIX_TIME_NS=$(date +%s%N)` once before constructing `INSTANCE_LABEL`.
Set `BACKUP_TIMESTAMP` once after the selected prefix parameters and their
values are confirmed; do not regenerate it later in the workflow. This gives
each workflow invocation a collision-resistant instance label and backup
prefix. Include the resolved `GLOBAL_SEED`, selected prefix parameters, and
their resolved values in the displayed run plan, and use the same seed in every
grid-cell command.

## Tier routing

Select exactly one tier from the confirmed `GRID_CELLS`. The tier determines
the search `cpu_cores_effective` minimum, the post-provision physical-core
acceptance gate, parallelism limits, and instance-only tuning.

| Tier | Grid cells | Search effective vCPU minimum | Prefer effective vCPUs | SSH physical-core minimum | Maximum concurrent cells | `workers` / `dataloader_workers` (informational) |
|---|---:|---:|---:|---:|---:|---:|
| S | 1 | 24 | — | 24 | 1 | config-default / config-default |
| M | 2–4 | 32 | — | 24 | 4 | config-default / config-default |
| L | 5–8 | 48 | 64 | 48 | 8 | config-default / config-default |
| XL | 9–16 | 64 | 128 | 64 | 8 | config-default / config-default |
| XXL | 17–30 | 64 | 128 | 64 | 16 | config-default / config-default |

The YAML config is never modified; `workers`, `dataloader_workers`, and
`batch_size` always use the committed defaults from `FLYWHEEL_CONFIG`.

### YAML config is never modified

All infrastructure values (`workers`, `dataloader_workers`, `batch_size`) use
the committed defaults from `FLYWHEEL_CONFIG`. Experiment parameter overrides
are supplied exclusively as CLI flags to the flywheel invocation. The
instance-side tuning step (Batch 2) is skipped entirely.

For tiers L, XL, and XXL, reserve CPU headroom for simulator coordination,
training, TensorBoard, shell overhead, and cgroup scheduling:

```text
CPU_RESERVE = max(8, ceil(PHYSICAL_CORES * 0.20))
CPU_CAP = floor((PHYSICAL_CORES - CPU_RESERVE) / WORKERS)
ABLATION_CONCURRENCY = min(TIER_MAX_CONCURRENCY, CPU_CAP, GRID_CELLS)
```

For tiers S and M, use:

```text
CPU_CAP = floor(PHYSICAL_CORES / WORKERS)
ABLATION_CONCURRENCY = min(TIER_MAX_CONCURRENCY, CPU_CAP, GRID_CELLS)
```

If the accepted host cannot support at least one concurrent cell after this
calculation, reject it. Never exceed `ABLATION_CONCURRENCY`, and never launch
all grid cells together unless the grid itself is no larger than that value.

Examples:

| Verified physical cores | XXL CPU reserve | `CPU_CAP` at `workers=6` | XXL concurrency | 30-cell batches |
|---:|---:|---:|---:|---:|
| 64 | 13 | 8 | 8 | 4 |
| 96 | 20 | 12 | 12 | 3 |
| 128 | 26 | 17 | 16 (tier cap) | 2 |

## Prerequisites

Before starting, ensure these are available on the development machine:

| Tool / key | Check |
|---|---|
| `vastai` CLI | `which vastai` |
| `VAST_API_KEY` | Set in `.env` |
| `S3_BUCKET` and AWS credentials | Set in `.env`, unless the instance has an IAM role |
| `AWS_PROFILE` | Alternative to `AWS_ACCESSORY_KEY_ID`/`AWS_SECRET_ACCESS_KEY` when using `~/.aws/credentials` |
| `tmux` | `which tmux` |
| `flock` | `which flock` |

Source `.env` at the start and require `VAST_API_KEY` and `S3_BUCKET`. Unless
the instance has an IAM role, also require at least one of:
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or `AWS_PROFILE`. When
`AWS_PROFILE` is set, resolve its credentials from `~/.aws/credentials` before
the SSH transfer (see Step 0). Tell the user which variable is missing and that
it must be added to `.env`; never print secret values or enable shell tracing.

## Workflow

### 0. Load secrets

Load `.env` without shell tracing and require the configuration:

```bash
set -a
. ./.env
set +a
test -n "${VAST_API_KEY:-}" || {
  printf '%s\n' 'ERROR: VAST_API_KEY is missing from .env' >&2
  exit 1
}
test -n "${S3_BUCKET:-}" || {
  printf '%s\n' 'ERROR: S3_BUCKET is missing from .env' >&2
  exit 1
}
```

If `AWS_PROFILE` is set but `AWS_ACCESS_KEY_ID` is not, resolve the profile's
credentials into environment variables:

```bash
if test -n "${AWS_PROFILE:-}" && test -z "${AWS_ACCESS_KEY_ID:-}"; then
  AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile "$AWS_PROFILE") || {
    printf 'ERROR: Could not resolve %s from AWS profile %s\n' \
      'aws_access_key_id' "$AWS_PROFILE" >&2
    exit 1
  }
  AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile "$AWS_PROFILE") || {
    printf 'ERROR: Could not resolve %s from AWS profile %s\n' \
      'aws_secret_access_key' "$AWS_PROFILE" >&2
    exit 1
  }
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
fi
```

Do not print either value or run with `set -x`.

### 1. Search offers

Use the selected tier's `SEARCH_EFFECTIVE_VCPUS` in this query:

```bash
vastai search offers \
  'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 gpu_ram>=24 gpu_max_power>=400 compute_cap>=890 total_flops>=80 cpu_cores_effective>=SEARCH_EFFECTIVE_VCPUS cpu_ram>=64 disk_bw>=1000 pci_gen>=4 pcie_bw>=20 inet_down>=500 inet_up>=200 reliability>=0.99 rentable=true verification=verified gpu_display_active=false' \
  --order dph_total+ \
  --raw
```

The query contains every hard searchable requirement. In raw results,
`cpu_ram` may be reported in MiB even though the query value is in GB. Treat
`cpu_cores_effective` as logical CPU capacity only; it does not establish the
physical-core count or SMT topology. Those are verified over SSH after rental.

The non-CPU hard requirements are one full RTX 4090 with approximately 24 GB
VRAM, at least 400 W power, Ada compute capability 8.9, at least 80 TFLOPS, 64
GB RAM, 1000 MB/s disk, PCIe Gen4 with at least 20 GB/s measured bandwidth,
500/200 Mb/s down/up networking, reliability at least 0.99, Vast verification,
rentability, and no active display workload. Do not weaken any requirement
without explicit user approval.

Show exactly these columns:

| Offer ID | CPU model | Effective vCPUs | RAM | Disk MB/s | PCIe GB/s | GPU power | Down/Up Mb/s | Reliability | $/hr | Location |

Rank eligible offers by:

1. CPU family: EPYC 9005, EPYC 9004, Threadripper 7000, EPYC 7003, modern
   Ryzen 7000/9000;
2. offers at or above the tier's preferred effective-vCPU count, when present;
3. lowest $/hr;
4. highest disk bandwidth;
5. highest reliability.

Reject EPYC 7001/7002 from `cpu_name`. Treat every other CPU name only as a
ranking hint: model specifications describe the whole host CPU, not the cores
allocated to this rental. The SSH topology gate is authoritative. Do not use
`cpu_ghz` as a criterion.

If no offer satisfies the hard filters, show that outcome and ask whether to
wait or explicitly relax named criteria. Do not silently lower filters. Show a
recommendation, rationale, the selected tier, and ask the user to confirm an
offer or an ordered shortlist of 3–5 offers.

### 2. User confirms priority list

Ask the user to confirm one offer or a priority-ordered shortlist of three to
five offers from the current search result. Record that exact order. Do not run
a separate availability check: RTX 4090 offers are volatile, and each create
attempt below is the authoritative availability test.

### 3. Rapid-fire create

Try the confirmed offers in order with the fixed image and instance settings:

```bash
CREATED=false
OFFER_IDS=(OFFER_1 OFFER_2 OFFER_3)
for id in "${OFFER_IDS[@]}"; do
  output=$(vastai create instance "$id" \
    --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
    --disk 100 --ssh --direct --label "$INSTANCE_LABEL" \
    --cancel-unavail 2>&1)
  if printf '%s\n' "$output" | grep -q "new_contract"; then
    INSTANCE_ID=$(printf '%s\n' "$output" | grep -oP "new_contract': \K\d+")
    if test -n "$INSTANCE_ID"; then
      CREATED=true
      break
    fi
  fi
  sleep 1
done
test "$CREATED" = true || {
  printf '%s\n' 'ERROR: Could not create any confirmed offer' >&2
  exit 1
}
```

Populate `OFFER_IDS` with every user-confirmed offer. Vast CLI output is a
mixed `Started.` prefix and Python-dict-like text, not reliable JSON; parse the
plain-text `new_contract` field as shown. Stop after the first success. If all
offers fail or no numeric `INSTANCE_ID` is parsed, stop without polling,
cleanup, or setup against an empty ID.

### 4. Post-create duplicate cleanup

Run `vastai show instances --raw` and identify every instance whose label
equals `INSTANCE_LABEL`. If more than one exists, list their IDs, status,
hardware, and price. Ask which one to keep, or explicitly propose keeping the
best matching instance and destroying the others. Never destroy an instance
that has not first been listed to the user. After cleanup, verify exactly one
instance with the label remains and set `INSTANCE_ID` to its ID.

### 5. Poll for running and SSH readiness

Poll for the running state, then resolve the current SSH endpoint:

```bash
status=""
for i in $(seq 1 30); do
  status=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | \
    uv run python -c \
      "import json,sys; print(json.load(sys.stdin).get('actual_status', ''))" \
      2>/dev/null)
  printf 'Poll %s: status=%s\n' "$i" "$status"
  test "$status" = running && break
  sleep 10
done
test "$status" = running || {
  printf 'ERROR: Instance did not reach running state; latest status=%s\n' \
    "$status" >&2
  exit 1
}

SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
HOST=$(printf '%s\n' "$SSH_URL" | sed 's/.*@//;s/:.*//')
PORT=$(printf '%s\n' "$SSH_URL" | sed 's/.*://')
test -n "$HOST" && test -n "$PORT"
```

Do not reuse host or port values from the create response. Probe SSH separately
because `actual_status=running` does not guarantee that the mapped port is
ready:

```bash
SSH_READY=false
for i in $(seq 1 12); do
  printf 'SSH probe %s\n' "$i"
  if ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
      -o ConnectTimeout=10 -p "$PORT" "root@$HOST" true 2>/dev/null; then
    SSH_READY=true
    break
  fi
  sleep 10
done
test "$SSH_READY" = true
```

Do not clone, tune, launch tmux, or start backup until SSH is ready.

The first successful probe uses trust on first use and records the endpoint key
in `~/.ssh/known_hosts`. Verify that the key was pinned, then require strict
host-key checking for every subsequent SSH connection:

```bash
ssh-keygen -F "[$HOST]:$PORT"
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
  -p "$PORT" "root@$HOST" true
```

Stop if no pinned key exists, the host key changes, or strict checking fails.
Do not ask the user to compare fingerprints manually. Credential transfer must
never use `accept-new` or disable strict checking.

If the instance fails to become running or SSH-ready, report its ID, latest
status, SSH URL when available, and failed attempt count. Ask whether to destroy
it and retry. Do not destroy without confirmation. On confirmation, run
`vastai destroy instance -y "$INSTANCE_ID"`, verify the ID no longer appears in
`vastai show instances`, return to Step 1 for a fresh offer snapshot, and obtain
a newly confirmed priority list. Do not reuse the stale offer list, change the
image, install SSH manually, or repeatedly reboot as a workaround.

### 6. Verify provisioned hardware and select concurrency

Treat every rental as provisional. Before cloning or setup, run:

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
  grep '^Cpus_allowed_list:' /proc/self/status
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

Parse the output into an acceptance table and apply every requirement below.

| Check | Requirement |
|---|---|
| Physical cores | At least `TIER_PHYSICAL_CORE_MIN` allowed unique `(CORE, SOCKET)` pairs |
| CPU generation | Zen 3 or newer; reject EPYC 7001/7002 |
| SMT | Prefer one thread per core; passing physical count is mandatory |
| CPU quota | At least 90% of advertised effective vCPUs |
| RAM | At least 64 GB allocated and consistent with the selected offer |
| GPU | Exactly one RTX 4090 with approximately 24 GB VRAM |
| GPU power | At least 400 W |
| PCIe | Gen4 x16 capability and offer `pcie_bw >= 20` GB/s |
| Throttling | Thermal and power-brake slowdown inactive |

Compare the observed CPU model, logical CPUs, RAM, GPU identity, GPU power, and
PCIe capability with the recorded raw offer. List every mismatch even when the
observed value still passes its hard minimum. Disk and network throughput are
accepted from the selected offer's `disk_bw`, `inet_down`, and `inet_up`
measurements; `df` verifies filesystem placement and available capacity but is
not a throughput test. Do not claim that real destination upload speed or disk
throughput was measured.

Count physical cores only from online CPU IDs included in
`Cpus_allowed_list`. Intersect that allowed CPU set with the
`CPU,CORE,SOCKET,ONLINE` table and count unique `(CORE, SOCKET)` pairs. The raw
host-wide `lscpu -p` count is diagnostic only and must not satisfy the gate by
itself. Fail if the allowed cpuset cannot be established.

For cgroup v2 memory, `memory.max=max` means no cgroup limit; otherwise its byte
value is the allocation limit. For cgroup v1, use `memory.limit_in_bytes` unless
it is an effectively unlimited sentinel value. Enforce the 64 GB requirement
against the finite cgroup limit when present, not `free -b` alone. If no finite
limit exists, use visible memory and call out that the allocation is unlimited.

For cgroup v2, `cpu.max` is `QUOTA PERIOD`; `max PERIOD` means no quota.
Otherwise divide quota by period. For cgroup v1, divide
`cpu.cfs_quota_us` by `cpu.cfs_period_us`. A finite result from 90% up to 100%
of the advertised effective vCPUs passes but must be called out. A lower result,
or an unreadable quota with no other way to establish allocation, fails.

The current PCIe generation may downshift while the GPU is idle. Do not reject
solely for an idle current generation below Gen4 when maximum capability is
Gen4 x16 and the offer's measured `pcie_bw` passes. Recheck under CUDA load if
other evidence indicates a restricted link. An idle P2 state or low idle GPU
utilization alone is not a failure.

For `GRID_CELLS=1`, the physical minimum is 24. For 2–4 cells it remains 24.
For 5–8 cells it is 48. For 9–30 cells it is 64. A host that passes the
general 24-core criterion but fails the selected tier's criterion is rejected.

On any failure, stop before setup, list every failed criterion, and ask whether
to destroy the provisional instance and return to Step 1. Do not destroy
without confirmation. If confirmed, destroy the instance, verify it no longer
appears in `vastai show instances`, obtain a fresh offer snapshot, and obtain a
fresh user-confirmed priority list before retrying.

These checks validate allocation and hardware characteristics, not end-to-end
workload throughput. Do not claim that training, evaluation, or DAgger speed was
benchmarked unless a separate workload run was actually performed.

After acceptance, calculate and display `CPU_RESERVE`, `CPU_CAP`,
`ABLATION_CONCURRENCY`, and `BATCH_COUNT`. Reassign every `RUN_PLAN` row's final
batch number using that concurrency while preserving deterministic row order.
Print the config's committed `workers`, `dataloader_workers`, and `batch_size`
values and note that no instance-side overrides will be applied. Ask the user to
confirm this computed launch plan before proceeding.

### 7. Confirm the instance-side tuning policy

No instance-side config overrides are applied. The YAML config on the committed
branch is authoritative for `workers`, `dataloader_workers`, and `batch_size`.
Experiment parameter overrides are supplied only as per-cell CLI flags. Step 8
Batch 2 (tier-tuning `sed` overrides) and its Python YAML assertion are skipped
entirely.

### 8. Setup and run batches on the instance

Use the SSH URL from `vastai ssh-url "$INSTANCE_ID"`. Break setup into these
batches.

#### Batch 1 — clone, uv, CUDA verification

Substitute the confirmed `GIT_BRANCH`, `GIT_COMMIT`, and `FLYWHEEL_CONFIG`
before execution. Do not clone until the baseline and full run-plan
confirmations are complete. After cloning, require the checked-out branch and
commit to match exactly and verify that `FLYWHEEL_CONFIG` exists. If the remote
branch advanced after confirmation, the commit check fails; stop before uv
setup and restart baseline confirmation instead of running unconfirmed code.

Install system dependencies for headless MuJoCo rendering (EGL/GLFW) before
uv setup. The base image lacks GL libraries and `$DISPLAY`. Export
`MUJOCO_GL=egl` before every flywheel invocation.

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "git clone --branch 'GIT_BRANCH' --single-branch \
     https://github.com/skpro19/toy-pickplace.git /workspace/toy-pickplace && \
    cd /workspace/toy-pickplace && \
    test \"\$(git branch --show-current)\" = 'GIT_BRANCH' && \
    test \"\$(git rev-parse HEAD)\" = 'GIT_COMMIT' && \
    test -f \"FLYWHEEL_CONFIG\" && \
    printf 'Checked out branch: %s\nCommit: %s\nConfig: %s\n' \
      'GIT_BRANCH' 'GIT_COMMIT' \"FLYWHEEL_CONFIG\" && \
   apt-get update -qq && apt-get install -y -qq \
     libgl1-mesa-glx libglib2.0-0 libegl1-mesa libgles2-mesa libglfw3 && \
   curl -LsSf https://astral.sh/uv/install.sh | sh && \
   /root/.local/bin/uv sync --locked --directory /workspace/toy-pickplace && \
   MUJOCO_GL=egl /root/.local/bin/uv run python -c \
     'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); \
      import mujoco, glfw; print(\"MuJoCo\", mujoco.__version__, \"GLFW\", glfw.__version__)'"
```

#### Batch 2 — tmux configuration only (no tier tuning)

Skip all YAML overrides. The committed config is authoritative for `workers`,
`dataloader_workers`, and `batch_size`. Only configure tmux:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "set -e; \
   touch ~/.no_auto_tmux; \
   printf '%s\n' 'set -g mouse on' > ~/.tmux.conf"
```

#### Batch 3 — start and verify TensorBoard, then create local wrappers

Start TensorBoard once before grid batch 1:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s tensorboard \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main \
      --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'"
```

Verify TensorBoard before launching a grid cell. Poll its remote endpoint for
up to 12 attempts at five-second intervals; on failure, print its last 20
lines and stop:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "for i in \$(seq 1 12); do
     tmux has-session -t tensorboard 2>/dev/null &&
       curl -fsS http://127.0.0.1:6006/ >/dev/null && exit 0
     sleep 5
   done
   tmux capture-pane -t tensorboard -p -S -20 2>/dev/null || true
   exit 1"
```

Immediately create and verify the local wrappers. This is deliberately before
the first grid batch so TensorBoard is available while training is running.
Immediately before creating them, automatically select the lowest unused
non-negative `LOCAL_WORKFLOW_INDEX`. Set
`LOCAL_TB_PORT=6006 + LOCAL_WORKFLOW_INDEX`,
`LOCAL_SSH_SESSION=vast-ssh-LOCAL_WORKFLOW_INDEX`, and
`LOCAL_TB_SESSION=tb-ablation-LOCAL_WORKFLOW_INDEX`. Hold a local lock while
selecting and creating the wrappers so concurrently starting workflows cannot
select the same index. Do not ask the user for an index:

```bash
set -e
exec 9>/tmp/toy-pickplace-ablation-local-wrapper.lock
flock 9
LOCAL_WORKFLOW_INDEX=""
for index in $(seq 0 999); do
  candidate_ssh_session="vast-ssh-$index"
  candidate_tb_session="tb-ablation-$index"
  candidate_tb_port=$((6006 + index))
  if tmux has-session -t "$candidate_ssh_session" 2>/dev/null ||
    tmux has-session -t "$candidate_tb_session" 2>/dev/null ||
    ss -ltn "sport = :$candidate_tb_port" | grep -q LISTEN; then
    continue
  fi
  LOCAL_WORKFLOW_INDEX=$index
  LOCAL_SSH_SESSION=$candidate_ssh_session
  LOCAL_TB_SESSION=$candidate_tb_session
  LOCAL_TB_PORT=$candidate_tb_port
  break
done
test -n "$LOCAL_WORKFLOW_INDEX"
tmux new-session -d -s "$LOCAL_SSH_SESSION" \
  "ssh -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -p $PORT root@$HOST"
tmux new-session -d -s "$LOCAL_TB_SESSION" \
  "ssh -o StrictHostKeyChecking=yes -N \
    -L $LOCAL_TB_PORT:127.0.0.1:6006 -p $PORT root@$HOST"
tmux has-session -t "$LOCAL_SSH_SESSION" && tmux has-session -t "$LOCAL_TB_SESSION"
flock -u 9
exec 9>&-
```

Print `http://localhost:$LOCAL_TB_PORT` now. If either wrapper cannot be
created, stop before the grid launch and report the relevant local tmux output.

#### Batch 4 — materialize a durable remote controller

The agent must not remain responsible for waiting for long-running cells.
On the development machine, set and retain the absolute remote control path
before launching the grid:

```text
CONTROL_DIR="/workspace/toy-pickplace/.ablation/${BACKUP_PREFIX}"
CONTROL_DIR/cells/
CONTROL_DIR/logs/
CONTROL_DIR/status/
CONTROL_DIR/status/history/
CONTROL_DIR/state/
```

Materialize the confirmed `RUN_PLAN` there. Use a deterministic `plan` file
with one `batch|run_name|tmux_session` row per cell. For every row, create one
shell-safe executable `cells/{tmux_session}.sh` containing its full
`/root/.local/bin/uv run python scripts/flywheel.py` command, with
`--config FLYWHEEL_CONFIG`, `--run-name`, and every swept override. Do not
commit or copy these transient files back to the local repository.

Each cell wrapper must:

1. export `MUJOCO_GL=egl` for headless MuJoCo rendering;
2. write `running` atomically to `status/{run_name}`;
3. `cd /workspace/toy-pickplace`, then stream stdout and stderr to both its
   tmux pane and `logs/{run_name}.log` with `tee -a`;
4. write `succeeded 0` or `failed EXIT_CODE` atomically to
   `status/{run_name}` before exiting with that same code.

Materialize wrappers and the controller via base64 encoding. A
double-quoted `ssh "..."` command consumes the inner shell quoting, so
heredocs inside it silently expand variables on the remote side. Build the
wrapper text locally, pipe it through `base64 -w0`, and decode on the
instance. This avoids the quoting trap and keeps `CONTROL_DIR` paths baked
correctly into the remote script.

Use a temporary file plus `mv` for each status update. A missing tmux session
is never a success signal; the terminal status file is authoritative. Set
`pipefail` and record `${PIPESTATUS[0]}` after the `tee` pipeline so a
successful `tee` never masks a failed flywheel command. The persisted log is
the durable record; pane output is a live convenience only.

Create an executable `controller.sh` in `CONTROL_DIR` and start it once in a
remote `ablation-controller` tmux session. The controller must:

1. read the materialized plan into arrays first, then iterate those arrays
   deterministically. **Do not use `while read ... |` pipe patterns**: a
   pipe creates a subshell and the IFS-delimited `read` inside it may produce
   incorrect field parsing, especially for the last column. Instead, redirect
   the plan file directly into the while loop:

   ```bash
   BATCH_VALUES=(); RUN_NAMES=(); TMUX_SESSIONS=()
   while IFS="|" read -r batch run_name tmux_session; do
     BATCH_VALUES+=("$batch"); RUN_NAMES+=("$run_name"); TMUX_SESSIONS+=("$tmux_session")
   done < "$PLAN_FILE"
   ```

   Then iterate indices `for ((i = 0; i < ${#BATCH_VALUES[@]}; i++))` and
   access `${BATCH_VALUES[$i]}`, `${RUN_NAMES[$i]}`, `${TMUX_SESSIONS[$i]}`.
   This keeps all field values in the current shell.
2. skip cells whose current status is already `succeeded 0` or `skipped 0`, then
   launch at most `ABLATION_CONCURRENCY` remaining `ablation-*` sessions for a
   batch and atomically write `failed 125` if any `tmux new-session` call fails.
   Before launching, if a tmux session exists for a cell but no corresponding
   status file exists (orphaned from a prior controller invocation), kill the
   old session before retrying;
3. atomically write `state/batch-N-started` after those sessions are launched;
4. wait for `succeeded 0`, `skipped 0`, or `failed EXIT_CODE` for every cell in
   that batch;
5. treat any nonterminal cell whose tmux session disappears before writing
   `succeeded 0` or `failed EXIT_CODE` as `failed 125`, including a wrapper that
   exits before writing `running`;
6. write `state/batch-N-succeeded` and launch the next batch only if every
   status is `succeeded 0` or `skipped 0`;
7. on the first failure, write `state/failed` with the run name, exit code,
   and log path, then exit non-zero without launching later batches;
8. after the final successful batch, write `state/completed`.

   **`set -e` trap:** Under `set -eo pipefail`, a post-increment expression
   that evaluates to zero (`(( LAUNCHED++ ))` when `LAUNCHED=0`) causes an
   immediate exit. Use `LAUNCHED=$((LAUNCHED + 1))` instead. Likewise, avoid
   bare `$VAR && command` for boolean checks; use
   `[ "$VAR" = true ] && command`. Apply these rules to every arithmetic or
   compound expression in the controller.

`ablation-controller` is the durable batch scheduler. It may run for days
after the agent disconnects or times out. It must not auto-retry a failed cell
or continue after a failure: the user decides whether to retry, skip, or stop
when the agent later reads `state/failed`.

Start it with:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s ablation-controller \
     'cd '$CONTROL_DIR' && exec bash ./controller.sh'"
```

Verify `ablation-controller` exists and wait only until
`state/batch-1-started` appears:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "for i in \$(seq 1 12); do
     test -f '$CONTROL_DIR/state/batch-1-started' && exit 0
     tmux has-session -t ablation-controller 2>/dev/null || exit 1
     sleep 5
   done
   tmux capture-pane -t ablation-controller -p -S -20 2>/dev/null || true
   exit 1"
```

Confirm that the listed first-batch cell sessions and `tensorboard` are
present with `tmux ls`. If the controller exits before this marker, capture its
pane and stop.

#### Batch 5 — start checkpoint backup once

After `state/batch-1-started` exists, start one `ckpt-bkp` session once.
The uploader must use `BACKUP_PREFIX` and run every 120 seconds so newly written
checkpoints, TensorBoard logs, DAgger data, and results are copied during long
grids.

Transfer AWS credentials and S3 configuration over SSH standard input only. Do
not place credentials in the SSH command string, arguments, output, or a file.

Create a wrapper script on the instance with the loop, then start it via `env
VAR=value ... bash wrapper.sh` in `tmux new-session`. The `env` prefix sets
variables for the session's initial process without leaking them into the tmux
global environment or the SSH command string. `AWS_SESSION_TOKEN`,
`AWS_REGION`, `S3_PREFIX`, and `S3_ENDPOINT_URL` may be empty:

```bash
printf '%s\n' "$S3_BUCKET" "${AWS_ACCESS_KEY_ID:-}" \
  "${AWS_SECRET_ACCESS_KEY:-}" "${AWS_SESSION_TOKEN:-}" \
  "${AWS_REGION:-}" "${S3_PREFIX:-}" "${S3_ENDPOINT_URL:-}" \
  "${AWS_PROFILE:-}" | \
  ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" "
    set -e
    IFS= read -r BUCKET
    IFS= read -r ACCESS_KEY
    IFS= read -r SECRET_KEY
    IFS= read -r SESSION_TOKEN
    IFS= read -r REGION
    IFS= read -r PREFIX
    IFS= read -r ENDPOINT_URL
    IFS= read -r PROFILE
    test -n \"\$BUCKET\" || {
      printf '%s\\n' 'ERROR: S3_BUCKET transfer failed' >&2
      exit 1
    }
    if tmux has-session -t ckpt-bkp 2>/dev/null; then
      printf '%s\\n' 'ERROR: ckpt-bkp already exists' >&2
      exit 1
    fi
    mkdir -p $CONTROL_DIR/state/backup-history
    for marker in backup-cycle-started backup-last-succeeded backup-failed; do
      if test -f $CONTROL_DIR/state/\$marker; then
        mv \$marker $CONTROL_DIR/state/backup-history/\$marker-\$(date +%s%N)
      fi
    done
    # Stop any old session (cleanup)
    tmux kill-session -t ckpt-bkp 2>/dev/null || true
    # Write the wrapper script on the instance — no credentials in its text
    cat > $CONTROL_DIR/ckpt-bkp-wrapper.sh << 'WRAPPER_EOF'
#!/usr/bin/env bash
set -eo pipefail
cd /workspace/toy-pickplace
while true; do
  if ! find checkpoints/flywheel runs/flywheel results/flywheel \
       -type f -print -quit 2>/dev/null | grep -q .; then
    sleep 30
    continue
  fi
  touch $CONTROL_DIR/state/backup-cycle-started
  if /root/.local/bin/uv run python scripts/s3_backup.py upload \
       --prefix $BACKUP_PREFIX \
       --components checkpoints,runs,results,dagger; then
    touch $CONTROL_DIR/state/backup-last-succeeded
  else
    touch $CONTROL_DIR/state/backup-failed
    exit 1
  fi
  sleep 120
done
WRAPPER_EOF
    chmod +x $CONTROL_DIR/ckpt-bkp-wrapper.sh
    # Launch with env prefix — credentials never appear in files or tmux global env
    tmux new-session -d -s ckpt-bkp \\
      env S3_BUCKET=\"\$BUCKET\" \\
          AWS_ACCESS_KEY_ID=\"\$ACCESS_KEY\" \\
          AWS_SECRET_ACCESS_KEY=\"\$SECRET_KEY\" \\
          AWS_SESSION_TOKEN=\"\$SESSION_TOKEN\" \\
          AWS_REGION=\"\$REGION\" \\
          S3_PREFIX=\"\$PREFIX\" \\
          S3_ENDPOINT_URL=\"\$ENDPOINT_URL\" \\
          AWS_PROFILE=\"\$PROFILE\" \\
      bash $CONTROL_DIR/ckpt-bkp-wrapper.sh
    tmux has-session -t ckpt-bkp
  "
```

Use the absolute uv path because detached tmux sessions do not reliably inherit
the login `PATH`. The credentials are passed via `env` prefix to
`new-session`, not through `set-environment -g` (which is unreliable for
command-string inheritance). They are never written to a file, log, or tmux
global environment. Do not start another backup session if `ckpt-bkp` already
exists. Before a confirmed restart, old backup markers are moved to
`state/backup-history/` so they cannot satisfy or fail the new attempt. The
uploader waits for the first artifact file instead of treating an empty new run
as an upload failure.

Verify a completed initial upload, not merely the tmux session. Poll for up to
ten minutes and fail immediately if the uploader exits or writes
`state/backup-failed`:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "for i in \$(seq 1 60); do
     test -f '$CONTROL_DIR/state/backup-failed' && exit 1
     test -f '$CONTROL_DIR/state/backup-last-succeeded' && exit 0
     tmux has-session -t ckpt-bkp 2>/dev/null || exit 1
     sleep 10
   done
   tmux capture-pane -t ckpt-bkp -p -S -20 2>/dev/null || true
   exit 1"
```

After success, capture the last ten pane lines and confirm `tmux ls` still
contains `ckpt-bkp`. Stop and diagnose if the pane or marker reports
authentication, repository, file-limit, or upload errors. Never display the
token while diagnosing. During later monitoring, treat `state/backup-failed`
as a failure even if an earlier upload succeeded.

### 9. Monitor, resume, and handle failures

Use the remote controller state—not live tmux membership—to determine the
outcome. At any later time, including after an agent interruption, inspect:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "cd '$CONTROL_DIR' && \
    { cat state/completed 2>/dev/null || true; } && \
    { cat state/failed 2>/dev/null || true; } && \
    { test ! -f state/backup-last-succeeded || printf '%s\n' 'backup: succeeded'; } && \
    { test ! -f state/backup-failed || printf '%s\n' 'backup: failed'; } && \
    { tmux has-session -t ckpt-bkp 2>/dev/null && \
        printf '%s\n' 'backup session: active' || \
        printf '%s\n' 'backup session: stopped'; } && \
    find status -maxdepth 1 -type f -printf '%f: ' -exec cat {} \;"
```

After backup has started, a stopped `ckpt-bkp` session is a failure even when
`backup-last-succeeded` exists, unless the user intentionally stopped it after
the grid completed and the final artifacts were independently verified.

When `state/completed` appears, wait for an upload cycle that started after
completion and then succeeded. This ensures the final cell's checkpoints and
results were visible when the upload began:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "for i in \$(seq 1 60); do
     test -f '$CONTROL_DIR/state/backup-failed' && exit 1
     tmux has-session -t ckpt-bkp 2>/dev/null || exit 1
     if test '$CONTROL_DIR/state/backup-cycle-started' \
          -nt '$CONTROL_DIR/state/completed' && \
        test '$CONTROL_DIR/state/backup-last-succeeded' \
          -nt '$CONTROL_DIR/state/backup-cycle-started'; then
       exit 0
     fi
     sleep 10
   done
   exit 1"
```

Do not report the grid as fully backed up, print cleanup instructions as ready
to run, or destroy the instance until this freshness check passes.

Attach to the controller with
`ssh -t -p "$PORT" "root@$HOST" "tmux attach -t ablation-controller"`.
Attach to a currently running cell with its generated `ablation-*` session
name to view its live `tee` output. Read the authoritative output with
`tail -f "$CONTROL_DIR/logs/{run_name}.log"`; use the persisted log for completed
or failed cells and do not rely on `tmux capture-pane` after a cell has exited.

When `state/failed` exists, report the failed run and log. Ask the user whether
to retry it, skip it, or stop. Do not launch later cells until the user answers.
A retry or skip requires a new, explicitly confirmed controller invocation.

Before that invocation, verify the old controller has exited. Create
`status/history/` and atomically move the failed cell's current status to
`status/history/{run_name}.attempt-{N}`; likewise move `state/failed` to a
uniquely numbered history file. Never delete or overwrite those records. Keep
successful current statuses in place so the new controller does not relaunch
completed cells.

For a retry, leave `status/{run_name}` absent so the relaunched wrapper can
write a fresh `running` status; append to the existing durable log. For a skip,
atomically write `skipped 0` to `status/{run_name}` using a temporary file and
`mv`. Then start the newly confirmed controller. This ordering prevents stale
terminal status from being mistaken for the new attempt and lets the controller
resume at the failed batch without rerunning successful cells.

### 10. Instance-side held-out evaluation and JSON-only upload

After `state/completed` exists and the final backup freshness check passes, ask
whether to evaluate every round's `best.pt` on a larger held-out set before the
instance is destroyed. This stage is optional, but when confirmed it is part of
the durable workflow and must finish or fail explicitly.

Confirm these settings before launch:

| Setting | Default | Constraint |
|---|---:|---|
| `FINAL_EVAL_EPISODES` | `100` | integer at least 5 and divisible by 5 |
| `FINAL_EVAL_SEED` | `20260716` | non-negative integer |
| `FINAL_EVAL_WORKERS` | committed config `workers` | integer at least 1 |

`scripts/final_score.py` derives five held-out seeds from
`FINAL_EVAL_SEED`, divides `FINAL_EVAL_EPISODES` evenly between them, and
evaluates every round's `best.pt`. Process `RUN_PLAN` rows sequentially so one
GPU is never shared by multiple held-out evaluations.

Run this stage entirely on the accepted instance. Do not download checkpoints,
metrics, or results to the development machine. Instance paths use the original
`RUN_PLAN` run names, such as `checkpoints/flywheel/RUN_NAME` and
`results/flywheel/RUN_NAME`; `BACKUP_PREFIX` is an S3 session namespace and is
not part of those instance-side paths.

#### Stop the periodic backup after final freshness

The periodic `ckpt-bkp` uploader uploads the whole results component and would
also upload plots generated by `final_score.py`. Only after the final backup
freshness check has passed, intentionally stop it and verify it is gone:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "test -f '$CONTROL_DIR/state/completed' && \
   test '$CONTROL_DIR/state/backup-cycle-started' \
     -nt '$CONTROL_DIR/state/completed' && \
   test '$CONTROL_DIR/state/backup-last-succeeded' \
     -nt '$CONTROL_DIR/state/backup-cycle-started' && \
   tmux kill-session -t ckpt-bkp && \
   ! tmux has-session -t ckpt-bkp 2>/dev/null"
```

This is the intentional post-completion stop allowed by Step 9. Do not stop
`ckpt-bkp` before final freshness passes.

#### Materialize the durable evaluator

Create `CONTROL_DIR/heldout-eval.sh` via the same local-build, base64-transfer
method used for cell wrappers. Do not interpolate secrets into its text. The
script must:

1. read `CONTROL_DIR/plan` in deterministic order and process each unique run
   name once;
2. verify `results/flywheel/{run_name}/metrics.json` and at least one
   `checkpoints/flywheel/{run_name}/round-*/best.pt` exist before evaluating;
3. archive an existing `final_scores.json` under
   `state/heldout-eval-history/` before a confirmed rerun instead of deleting or
   overwriting the prior result without a record;
4. run `scripts/final_score.py --run-name {run_name}` with the confirmed
   `--eval-episodes`, `--final-eval-seed`, and `--workers`, exporting
   `MUJOCO_GL=egl`;
5. stream output to both the tmux pane and
   `logs/heldout-eval-{run_name}.log`, preserving the evaluation exit code with
   `pipefail` and `${PIPESTATUS[0]}`;
6. validate that `results/flywheel/{run_name}/final_scores.json` is non-empty,
   valid JSON, records the confirmed evaluation settings, and contains a result
   for every discovered `best.pt` round;
7. upload exactly that JSON file with `boto3.client("s3").upload_file` to
   `BACKUP_PREFIX/results/{run_name}/final_scores.json`;
8. verify that exact object with `head_object` and its content length;
9. atomically write per-run running, succeeded, or failed status and stop on
   the first failure without evaluating later runs;
10. atomically write `state/heldout-completed` only after every run's JSON was
    uploaded and verified.

Use `boto3` directly, not `scripts/s3_backup.py upload --components
results`: component upload would include `metrics.json`, plots, and any other
files in the results directories. `final_score.py` may generate plots on the
instance, but this stage must not upload them. They can be regenerated later
from `final_scores.json` with `final_score.py --plot-only`.

The evaluator's exact-file upload and verification should use this pattern,
with values passed as arguments rather than embedded credentials:

```bash
/root/.local/bin/uv run python -c '
import json
import os
import sys
from pathlib import Path

import boto3

result_path = Path(sys.argv[1])
s3_key = sys.argv[2]
expected_episodes = int(sys.argv[3])
expected_seed = int(sys.argv[4])
expected_rounds = int(sys.argv[5])

data = json.loads(result_path.read_text())
assert data["eval_episodes"] == expected_episodes, data["eval_episodes"]
assert data["final_eval_seed"] == expected_seed, data["final_eval_seed"]
assert len(data["rounds"]) == expected_rounds, len(data["rounds"])

bucket = os.environ["S3_BUCKET"]
key_prefix = os.environ.get("S3_PREFIX", "").strip("/")
object_key = "/".join(part for part in (key_prefix, s3_key) if part)
client = boto3.client(
    "s3",
    region_name=os.environ.get("AWS_REGION"),
    endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
)
client.upload_file(str(result_path), bucket, object_key)
metadata = client.head_object(Bucket=bucket, Key=object_key)
assert metadata["ContentLength"] == result_path.stat().st_size, metadata
' \
  "results/flywheel/$run_name/final_scores.json" \
  "$BACKUP_PREFIX/results/$run_name/final_scores.json" \
  "$FINAL_EVAL_EPISODES" "$FINAL_EVAL_SEED" "$expected_rounds"
```

Start the materialized script once in a remote `heldout-eval` tmux session.
Transfer AWS credentials and S3 configuration over SSH standard input only, let
the new tmux session inherit them, then immediately remove them from tmux's
global environment and remove credentials from the remote setup shell:

```bash
printf '%s\n' "$S3_BUCKET" "${AWS_ACCESS_KEY_ID:-}" \
  "${AWS_SECRET_ACCESS_KEY:-}" "${AWS_SESSION_TOKEN:-}" \
  "${AWS_REGION:-}" "${S3_PREFIX:-}" "${S3_ENDPOINT_URL:-}" \
  "${AWS_PROFILE:-}" | \
  ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" "
    set -e
    IFS= read -r S3_BUCKET
    IFS= read -r AWS_ACCESS_KEY_ID
    IFS= read -r AWS_SECRET_ACCESS_KEY
    IFS= read -r AWS_SESSION_TOKEN
    IFS= read -r AWS_REGION
    IFS= read -r S3_PREFIX
    IFS= read -r S3_ENDPOINT_URL
    IFS= read -r AWS_PROFILE
    test -n \"\$S3_BUCKET\" || exit 1
    ! tmux has-session -t heldout-eval 2>/dev/null || {
      printf '%s\\n' 'ERROR: heldout-eval already exists' >&2
      exit 1
    }
    for name in S3_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_REGION S3_PREFIX S3_ENDPOINT_URL AWS_PROFILE; do
      test -z \"\${!name}\" || tmux set-environment -g \"\$name\" \"\${!name}\"
    done
    trap 'for name in S3_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_REGION S3_PREFIX S3_ENDPOINT_URL AWS_PROFILE; do tmux set-environment -gu "\$name"; done' EXIT
    tmux new-session -d -s heldout-eval \
      'exec bash $CONTROL_DIR/heldout-eval.sh'
    for name in S3_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_REGION S3_PREFIX S3_ENDPOINT_URL AWS_PROFILE; do
      tmux set-environment -gu \"\$name\"
    done
    trap - EXIT
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
    tmux has-session -t heldout-eval
  "
```

Inherited credentials exist only in the evaluator process environment. Never
write them to a file, command argument, log, status marker, or pane output.

#### Monitor held-out evaluation

Use state files as authoritative. A missing tmux session is not success:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "cd '$CONTROL_DIR' && \
   { test ! -f state/heldout-completed || printf '%s\n' 'heldout: completed'; } && \
   { test ! -f state/heldout-failed || cat state/heldout-failed; } && \
   find state -maxdepth 1 -type f -name 'heldout-*.status' \
     -printf '%f: ' -exec cat {} \; && \
   { tmux has-session -t heldout-eval 2>/dev/null && \
       printf '%s\n' 'heldout session: active' || \
       printf '%s\n' 'heldout session: stopped'; }"
```

Attach with
`ssh -t -p "$PORT" "root@$HOST" "tmux attach -t heldout-eval"`.
Use the persisted per-run log after a process exits.

Do not report held-out evaluation as complete or destroy the instance until
`state/heldout-completed` exists and every expected
`BACKUP_PREFIX/results/{run_name}/final_scores.json` path has been verified on
S3. On failure, report the run, exit code, and log path; do not auto-retry or
upload a stale JSON. Ask the user whether to retry or stop.

### 11. Follow-up download and replay

After the grid and final backup freshness check have completed, print the
following as optional commands only. Do not download anything automatically.
If the held-out stage ran, `final_scores.json` is already on S3 and its plots
can be recreated after an explicitly requested download with `--plot-only`:

```bash
set -a; . ./.env; set +a
uv run python scripts/s3_backup.py download BACKUP_PREFIX/RUN_NAME
uv run python scripts/final_score.py --run-name RUN_NAME
uv run python scripts/final_score.py --run-name RUN_NAME --plot-only
```

Generate one `final_score.py` command per `RUN_PLAN` row. Use one local
TensorBoard instance for all downloaded `abl-*` runs.

## Final output

Provisioning setup is complete after the selected instance passes the tier
gate, TensorBoard, `ablation-controller`, `ckpt-bkp`, and local wrappers are
verified. Grid execution completes only when `state/completed` exists and the
final backup freshness check passes, or a failure has been explicitly handled
by the user. If the user confirms held-out evaluation, the overall workflow is
complete only when `state/heldout-completed` exists and every expected
`final_scores.json` has been verified on S3.

Print:

- parameter names, exact value lists, selected backup-prefix parameters and
  resolved values, `GRID_CELLS`, selected tier, and `RUN_PLAN`;
- confirmed `GIT_BRANCH`, `GIT_COMMIT`, and `FLYWHEEL_CONFIG`, plus all three
  verifications performed after cloning;
- instance ID and `vastai ssh-url` retrieval command;
- advertised effective vCPUs, verified physical cores, CPU quota, CPU reserve,
  CPU cap, computed concurrency, and batch count;
- applied `workers`, `dataloader_workers`, and batch-size policy;
- `CONTROL_DIR`, controller state, current/completed batch status, and all
  run-specific tmux attach commands;
- local `LOCAL_SSH_SESSION` and `LOCAL_TB_SESSION` attach commands;
- the indexed local TensorBoard URL;
- S3 backup prefix, download commands, and `final_score.py` commands;
- held-out episode count, root seed, workers, state, per-run log and tmux attach
  commands, and exact uploaded S3 JSON paths when the held-out stage runs;
- cleanup destroy command.

## Safety notes

- Do not store API keys, SSH details, tokens, or transient host/port in files.
- RTX 4090 offers are volatile: use the confirmed shortlist directly with
  `--cancel-unavail`; stop after the first successful create.
- A high advertised effective-vCPU count never bypasses the SSH physical-core
  gate.
- This command rejects grids larger than 30 cells rather than silently changing
  the experiment design or provisioning more instances.
- Never use a component-wide results upload for held-out evaluation. Upload and
  verify only `BACKUP_PREFIX/results/{run_name}/final_scores.json`.
