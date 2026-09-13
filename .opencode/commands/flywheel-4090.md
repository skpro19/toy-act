---
description: Provision one RTX 4090 and run one config-defined flywheel
agent: flywheel-4090
---

> **SSH banner handling**: Vast.ai emits a welcome banner on stderr for every SSH
> connection. When capturing SSH command output in a variable, always use
> `2>/dev/null | tail -1` to extract the actual value. Never use `2>&1` in any
> SSH capture or status check — it swallows the banner into the captured string
> and breaks exact-match probes (e.g. `echo SSH_OK`). All SSH probes in this
> workflow use `2>/dev/null` (exit code) or `2>/dev/null | tail -1` (captured
> value). Non-SSH commands (vastai CLI, local commands) are unaffected; they
> can use `2>&1` or no redirect freely.

Run one standard flywheel on one Vast.ai RTX 4090. This command owns baseline
confirmation, provisioning, hardware acceptance, setup, durable launch,
checkpoint backup, automatic held-out evaluation, result upload, and instance
destruction. It is
self-contained: follow this workflow without consulting another runbook.

The experiment is fixed by a committed experiment definition and its referenced
committed base config. One seed is selected interactively from the definition's
committed `global_seeds` list. Do not collect sweep parameters, construct a grid,
or modify the config on the instance. Do not add experiment CLI overrides. The
only flywheel CLI arguments supplied by this workflow are `--config` and
`--run-name`. The run name is derived from the suite, experiment name, and
selected seed.

## Workflow at a glance

1. Confirm the remote branch, commit, experiment definition, base config, and
   selected seed. Resolve and display the complete merged config.
2. Load secrets, select an offer, and provision one instance.
3. Accept or reject the instance using SSH hardware checks.
4. Clone the exact commit and verify CUDA and headless MuJoCo.
5. Start TensorBoard, local wrappers, one durable flywheel runner, and backup.
6. Start a durable local supervisor that monitors terminal state.
7. After an acknowledged post-completion backup, automatically run held-out
   evaluation, upload finalized metrics, its JSON, and plots, then destroy the
   instance. On failure, upload diagnostics best-effort and destroy the instance.

## Baseline and run planning

This command accepts no experiment parameter arguments. If arguments are
provided, explain that this workflow runs the committed experiment and stop.

Before asking questions, inspect only `configs/flywheel/mlp_vision/` subdirectories
for experiment definition files. These are regular files ending in `.yaml` or
`.yml` inside any immediate subdirectory of `mlp_vision/` (e.g.
`BASE/baseline.yaml`, `SMOKE_TEST/baseline.yaml`). Exclude files directly under
`mlp_vision/` itself. Rank them newest first by the later of their filesystem
creation and modification timestamps (use the modification timestamp when
creation time is unavailable), and retain the top five. Resolve ties by path in
ascending order. Do not read file contents or run any other workflow commands
yet.

Then make exactly one call to the built-in Question tool containing both
of these questions at the same time:

1. **Branch**: select the Git branch to clone. Recommend `dev`, offer `main`,
   and allow a custom branch.
2. **Experiment definition**: select the committed experiment config path. Offer
   the five paths discovered above in newest-first order and allow a custom
   path. Mark the newest path as recommended. A custom path must be under
   `configs/flywheel/mlp_vision/` and end in `.yaml` or `.yml`.

Do not ask these two questions separately or repeat any of them later. From
the single Question-tool response, set:

```text
GIT_BRANCH=<confirmed branch>
EXPERIMENT_CONFIG=<confirmed committed experiment config path>
```

Validate and load the remote files without changing the local checkout.
Fetch the experiment definition and its referenced base config. Resolve
`base_config` relative to the experiment file's parent directory.

```bash
case "$GIT_BRANCH" in
  ""|*[!A-Za-z0-9._/-]*)
    printf '%s\n' 'ERROR: Branch may contain only letters, digits, ., _, /, and -' >&2
    exit 1
    ;;
esac
git check-ref-format --branch "$GIT_BRANCH"
git fetch origin "$GIT_BRANCH:refs/remotes/origin/$GIT_BRANCH" || {
  printf 'ERROR: Could not fetch origin/%s\n' "$GIT_BRANCH" >&2
  exit 1
}
GIT_COMMIT=$(git rev-parse "origin/$GIT_BRANCH^{commit}") || exit 1
REMOTE_EXPERIMENT=$(git show "$GIT_COMMIT:$EXPERIMENT_CONFIG") || {
  printf 'ERROR: %s does not exist on origin/%s\n' \
    "$EXPERIMENT_CONFIG" "$GIT_BRANCH" >&2
  exit 1
}
REMOTE_FLYWHEEL=$(git show "$GIT_COMMIT:scripts/flywheel.py") || {
  printf 'ERROR: scripts/flywheel.py does not exist at %s\n' \
    "$GIT_COMMIT" >&2
  exit 1
}
REMOTE_RESOLVER=$(git show "$GIT_COMMIT:scripts/resolve_experiment.py") || {
  printf 'ERROR: scripts/resolve_experiment.py does not exist at %s\n' \
    "$GIT_COMMIT" >&2
  exit 1
}
printf 'Branch: %s\nCommit: %s\nExperiment: %s\n\n%s\n' \
  "$GIT_BRANCH" "$GIT_COMMIT" "$EXPERIMENT_CONFIG" "$REMOTE_EXPERIMENT"
```

Confirm from `REMOTE_FLYWHEEL` that `--config` and `--run-name` are supported.
Confirm from `REMOTE_RESOLVER` that it supports `--experiment`, `--seed`,
`--sha256`, and `--json-manifest` flags. Ensure the resolver is present in
the local working tree at `scripts/resolve_experiment.py` so planning
resolution uses the exact commit's logic.

Write the experiment definition to a temporary file and resolve the base
config path from its `base_config` field (relative to the experiment's parent
directory). Verify the base config exists in the same remote commit:

```bash
EXPERIMENT_BASE_CONFIG=$(
  uv run python -c "
import sys, yaml
from pathlib import Path
exp = yaml.safe_load(sys.stdin.read())
base_rel = exp['base_config']
resolved = str((Path('$EXPERIMENT_CONFIG').parent / base_rel).resolve().relative_to(Path.cwd()))
sys.stdout.write(resolved)
" <<< "$REMOTE_EXPERIMENT"
) || exit 1
git show "$GIT_COMMIT:$EXPERIMENT_BASE_CONFIG" > /dev/null || {
  printf 'ERROR: base config %s does not exist on origin/%s\n' \
    "$EXPERIMENT_BASE_CONFIG" "$GIT_BRANCH" >&2
  exit 1
}
```

Parse `REMOTE_EXPERIMENT` with `yaml.safe_load`. Validate the experiment schema:

- Only `description`, `base_config`, `overrides`, and `global_seeds` are
  permitted.
- `description` is a non-empty string.
- `base_config` is a non-empty string resolving inside
  `configs/flywheel/mlp_vision/`.
- `overrides` is a mapping (may be empty); `global_seed` and `run_name` are
  forbidden inside overrides.
- `global_seeds` is a non-empty list of distinct, non-negative integers.
- Every override key is a valid `scripts/flywheel.py` parameter.

Reject the experiment before provisioning if any validation fails.

Display the experiment as a table:

| Field | Value |
|---|---|
| Experiment path | `EXPERIMENT_CONFIG` |
| Description | The experiment description |
| Base config | Resolved `EXPERIMENT_BASE_CONFIG` |
| Available seeds | Committed `global_seeds` list |

Then make exactly one more call to the built-in Question tool asking:

**Seed**: select one seed from the experiment's `global_seeds` list. Offer each
seed as a choice and allow a custom integer. Reject values not in the committed
list.

Set:

```text
SELECTED_SEED=<confirmed integer>
```

Resolve the complete config deterministically using the local resolver
(which must match the remote commit's logic):

```bash
RESOLVED_CONFIG_SHA256=$(
  uv run python scripts/resolve_experiment.py \
    --experiment "$EXPERIMENT_CONFIG" --seed "$SELECTED_SEED" --sha256
) || exit 1

FLYWHEEL_ARCH=$(
  uv run python -c "
import yaml, sys
from pathlib import Path
import subprocess
result = subprocess.run(
    [sys.executable, 'scripts/resolve_experiment.py',
     '--experiment', sys.argv[1], '--seed', sys.argv[2]],
    capture_output=True, text=True)
config = yaml.safe_load(result.stdout)
print(config['arch'])
" "$EXPERIMENT_CONFIG" "$SELECTED_SEED"
) || exit 1
```

Display the complete resolved config as tables grouped by section (Run,
Flywheel loop, Expert data collection, Training, Dataset mixing, DAgger
rollout, In-loop evaluation, Held-out evaluation, Parallelism, Seeds).
Resolve and list every value. Include the resolved config SHA-256 line.
Never dump raw YAML in the response — always use the grouped table format.

Then set each value once:

```bash
RUN_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
UNIX_TIME_NS=$(date +%s%N)
BASE_SUITE=$(basename "$(dirname "$EXPERIMENT_CONFIG")")
EXPERIMENT_NAME=$(basename "$EXPERIMENT_CONFIG" .yaml | sed 's/\.yml$//')
RUN_NAME="${RUN_TIMESTAMP}_${UNIX_TIME_NS}_${BASE_SUITE}_${EXPERIMENT_NAME}_seed${SELECTED_SEED}"
INSTANCE_LABEL="toy-pickplace-${RUN_NAME}-${UNIX_TIME_NS}"
CONTROL_DIR="/workspace/toy-pickplace/.flywheel/${RUN_NAME}"
```

Keep the generated `RUN_NAME` unchanged and derive `INSTANCE_LABEL` and
`CONTROL_DIR` from it. A new launch must not reuse an existing S3 prefix. A
recovery may retain its previously recorded prefix only when the matching
instance and control state establish that it is the same workflow.

Display the complete plan for information, then proceed without asking for
another confirmation:

| Item | Required value |
|---|---|
| Branch and commit | Confirmed `GIT_BRANCH` and immutable `GIT_COMMIT` |
| Experiment | Committed `EXPERIMENT_CONFIG` path and description |
| Base config | Committed `EXPERIMENT_BASE_CONFIG` |
| Overrides | Exact committed override mapping |
| Root seed | Selected from committed `global_seeds` list |
| Arch | Resolved `arch` value (`FLYWHEEL_ARCH`) |
| Resolved config | Complete merged config with SHA-256 `RESOLVED_CONFIG_SHA256` |
| Run name | `RUN_NAME` (suite_experiment_seed_timestamp) |
| Held-out evaluation | Committed `final_eval_episodes`, `final_eval_seed`, `final_eval_workers`, and `final_eval_capture_hz` |
| Hardware gate | One RTX 4090, 24 physical cores, 64 GB RAM |
| Launch command | `scripts/flywheel.py --config CONTROL_DIR/resolved-config.yaml --run-name RUN_NAME` |

After a unique numeric `INSTANCE_ID` has been established, every unrecoverable
terminal failure must trigger best-effort diagnostics followed by automatic
destruction and removal verification. Before the local supervisor starts, the
agent performs this policy directly. After it starts, the supervisor owns the
policy. Never apply automatic destruction while duplicate-instance identity is
ambiguous; list duplicates and resolve the target with the user first.

For direct pre-supervisor cleanup, always run the idempotent helper below. It
does not return until Vast positively confirms removal; a failed status query is
never treated as absence:

```bash
.opencode/commands/scripts/flywheel-4090/destroy-instance.sh "$INSTANCE_ID"
```

Every failure branch after establishing one numeric `INSTANCE_ID` and before
`supervisor-running` must invoke this helper before stopping or retrying.

## Fixed hardware profile

Use one fixed profile. There are no tiers, concurrency calculations, or
batches of experiment runs.

| Requirement | Value |
|---|---:|
| Search effective vCPU minimum | 24 |
| SSH physical-core minimum | 24 |
| Concurrent flywheel runs | 1 |
| `workers` | Committed config value |
| `dataloader_workers` | Committed config value |
| `batch_size` | Committed config value |
| Held-out evaluation | Committed config values; always runs after final backup |

## Prerequisites

| Tool / key | Check |
|---|---|
| `vastai` CLI | `which vastai` |
| `VAST_API_KEY` | Set in `.env` |
| `S3_BUCKET` and AWS credentials | Set in `.env`, unless the instance has an IAM role |

| `AWS_PROFILE` | Alternative to access-key variables via `~/.aws/credentials` |
| `aws` CLI | May be required to log in when `AWS_PROFILE` uses SSO |
| `tmux` | `which tmux` |
| `flock` | `which flock` |

## Workflow

### 0. Load secrets

Load `.env` without shell tracing and require `VAST_API_KEY` and `S3_BUCKET`:

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

Unless the instance has an IAM role, require either a complete access-key pair
or `AWS_PROFILE`. Reject a partial access-key pair. If only `AWS_PROFILE` is
set, resolve it before SSH transfer:

```bash
if test -n "${AWS_ACCESS_KEY_ID:-}" || test -n "${AWS_SECRET_ACCESS_KEY:-}"; then
  test -n "${AWS_ACCESS_KEY_ID:-}" && test -n "${AWS_SECRET_ACCESS_KEY:-}" || {
    printf '%s\n' 'ERROR: Both AWS access-key variables are required' >&2
    exit 1
  }
elif test -n "${AWS_PROFILE:-}"; then
  mapfile -t RESOLVED_AWS < <(uv run python -c '
import os
import boto3

session = boto3.Session(profile_name=os.environ["AWS_PROFILE"])
credentials = session.get_credentials()
if credentials is None:
    raise SystemExit(1)
frozen = credentials.get_frozen_credentials()
print(frozen.access_key)
print(frozen.secret_key)
print(frozen.token or "")
print(session.region_name or "")
') || exit 1
  test "${#RESOLVED_AWS[@]}" -eq 4 || {
    printf 'ERROR: Could not resolve credentials from AWS profile %s\n' \
      "$AWS_PROFILE" >&2
    exit 1
  }
  AWS_ACCESS_KEY_ID=${RESOLVED_AWS[0]}
  AWS_SECRET_ACCESS_KEY=${RESOLVED_AWS[1]}
  AWS_SESSION_TOKEN=${RESOLVED_AWS[2]}
  test -n "${AWS_REGION:-}" || AWS_REGION=${RESOLVED_AWS[3]}
  test -n "$AWS_ACCESS_KEY_ID" && test -n "$AWS_SECRET_ACCESS_KEY" || exit 1
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_REGION
  unset RESOLVED_AWS
fi
```

Tell the user which required variable is missing. Never print secret values or
enable shell tracing. Before provisioning, use `boto3` to list at most one
object under any of these artifact prefixes:
`checkpoints/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/`, `runs/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/`,
`data/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/`, or `results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/`. If any object
exists, stop and require the user to choose a new run name. Continue only when
recovering the matching instance and control state rather than launching a new
run. Explain that the backup sync deletes remote keys absent locally, so
prefix reuse can be destructive.

### 1. Search and select offers

Search with every hard searchable requirement:

```bash
vastai search offers \
  'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 gpu_ram>=24 gpu_max_power>=400 compute_cap>=890 total_flops>=80 cpu_cores_effective>=24 cpu_ram>=64 disk_bw>=1000 pci_gen>=4 pcie_bw>=20 inet_down>=500 inet_up>=200 reliability>=0.99 rentable=true verification=verified gpu_display_active=false' \
  --order dph_total+ \
  --raw
```

Show exactly these columns:

| Offer ID | CPU model | Effective vCPUs | RAM | Disk MB/s | PCIe GB/s | GPU power | Down/Up Mb/s | Reliability | $/hr | Location |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

Rank eligible offers by CPU family (EPYC 9005, EPYC 9004, Threadripper 7000,
EPYC 7003, then modern Ryzen 7000/9000), lowest price, disk bandwidth, and
reliability. Reject EPYC 7001/7002. CPU model is only a ranking hint because
the SSH allocation gate is authoritative. Do not use `cpu_ghz`.

Automatically select the best offer (or an ordered shortlist of up to three)
as long as every selected offer is priced below $0.60/hr. Do not ask the user
to confirm the selection. If the first offer becomes unavailable during
provisioning, fall back to the next without asking. Do not weaken a hard
filter without explicit approval. If no offer passes the filters, stop and
ask whether to wait or relax named criteria. Do not run a separate
availability check.

### 2. Provision one instance

Try the selected offers in order:

```bash
PRIOR_ID=$(vastai show instances --raw 2>/dev/null | \
  INSTANCE_LABEL="$INSTANCE_LABEL" uv run python -c '
import json, os, sys
data = json.load(sys.stdin)
if isinstance(data, dict):
    data = data.get("instances", [data])
for item in data:
    if item.get("label") == os.environ["INSTANCE_LABEL"]:
        print(item["id"])
        break
')
if test -n "$PRIOR_ID"; then
  echo "Found existing instance with label $INSTANCE_LABEL: $PRIOR_ID"
  echo "Destroying prior instance before provisioning a new one..."
  vastai destroy instance -y "$PRIOR_ID" 2>&1 || true
  for i in $(seq 1 30); do
    instances=$(vastai show instances --raw 2>/dev/null || echo "[]")
    count=$(echo "$instances" | PRIOR_ID="$PRIOR_ID" uv run python -c '
import json, os, sys
data = json.load(sys.stdin)
if isinstance(data, dict):
    data = data.get("instances", [data])
target = os.environ.get("PRIOR_ID", "")
if not target:
    raise SystemExit(2)
ids = [item.get("id") for item in data if str(item.get("id")) == target]
print(len(ids))
' 2>/dev/null) || count="error"
    if [ "$count" = "0" ]; then
      echo "Prior instance $PRIOR_ID destroyed"
      break
    fi
    sleep 10
  done
fi

CREATED=false
OFFER_IDS=(OFFER_1 OFFER_2 OFFER_3)
for id in "${OFFER_IDS[@]}"; do
  output=$(vastai create instance "$id" \
    --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
    --disk 100 --ssh --direct --label "$INSTANCE_LABEL" \
    --cancel-unavail 2>&1)
  if printf '%s\n' "$output" | grep -q "new_contract"; then
    CREATED=true
    INSTANCE_ID=$(printf '%s\n' "$output" | grep -oP "new_contract': \K\d+")
    break
  fi
  for reconcile_attempt in 1 2 3; do
    INSTANCE_ID=$(vastai show instances --raw | \
      INSTANCE_LABEL="$INSTANCE_LABEL" uv run python -c '
import json
import os
import sys

instances = json.load(sys.stdin)
if isinstance(instances, dict):
    instances = instances.get("instances", [instances])
matches = [
    item
    for item in instances
    if item.get("label") == os.environ["INSTANCE_LABEL"]
]
if len(matches) > 1:
    ids = ", ".join(str(item.get("id")) for item in matches)
    print(f"ERROR: Multiple exact-label instances: {ids}", file=sys.stderr)
    raise SystemExit(2)
if matches:
    print(matches[0]["id"])
')
    reconcile_status=$?
    test "$reconcile_status" -eq 0 || exit "$reconcile_status"
    if test -n "$INSTANCE_ID"; then
      CREATED=true
      break
    fi
    sleep 2
  done
  if test "$CREATED" = true; then
    break
  fi
  sleep 1
done
test "$CREATED" = true ||
  printf '%s\n' 'WARNING: Create output did not yield an instance ID; reconciling by label' >&2
```

Reconcile by exact label after every create response that does not report
`new_contract`, before trying the next offer. Stop after the first reported or
reconciled success. Whether ID parsing succeeds or not, always run
`vastai show instances --raw` and identify every instance whose label exactly
equals `INSTANCE_LABEL`. This reconciliation prevents a changed or truncated
create response from leaving an untracked billed instance. If no matching
instance exists, stop without polling or cleanup against an empty ID. If one
exists, use its numeric ID as `INSTANCE_ID`. If multiple instances exist, list
their IDs, status, hardware, and price; never destroy one before obtaining user
confirmation. Verify exactly one remains before continuing.

### 3. Wait for running and secure SSH

Poll `vastai show instance "$INSTANCE_ID" --raw` every ten seconds for up to
30 attempts. Parse `actual_status` with `uv run python`; stop if it does not
become `running`. Then resolve a fresh endpoint with
`vastai ssh-url "$INSTANCE_ID"`; do not reuse create-response host data.

Probe SSH for up to 12 attempts at ten-second intervals. The first successful
probe may use `StrictHostKeyChecking=accept-new`. Verify the key is pinned with
`ssh-keygen -F "[$HOST]:$PORT"`, then require
`StrictHostKeyChecking=yes` and `BatchMode=yes` for every later connection,
including credential transfers.

If running or SSH readiness fails, report the instance ID, latest status,
endpoint when available, and attempts. Automatically destroy the instance and
verify removal. Ask whether to retry only after destruction; a retry must repeat
the offer search and obtain a fresh shortlist.

### 4. Verify provisioned hardware

Treat the rental as provisional. Before cloning, collect:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" '
  set -e
  lscpu
  lscpu -e=CPU,CORE,SOCKET,ONLINE
  lscpu -p=CORE,SOCKET | grep -v "^#" | sort -u | wc -l
  nproc
  grep "^Cpus_allowed_list:" /proc/self/status
  free -b
  test ! -r /sys/fs/cgroup/memory.max || cat /sys/fs/cgroup/memory.max
  test ! -r /sys/fs/cgroup/memory/memory.limit_in_bytes || cat /sys/fs/cgroup/memory/memory.limit_in_bytes
  test ! -r /sys/fs/cgroup/cpu.max || cat /sys/fs/cgroup/cpu.max
  test ! -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
  test ! -r /sys/fs/cgroup/cpu/cpu.cfs_period_us || cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  df -hT /workspace /
  nvidia-smi --query-gpu=name,memory.total,power.limit,power.default_limit,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv
'
```

Parse and display an acceptance table:

| Check | Requirement |
|---|---|
| Physical cores | At least 24 allowed unique `(CORE, SOCKET)` pairs |
| CPU generation | Zen 3 or newer; reject EPYC 7001/7002 |
| CPU quota | At least 90% of advertised effective vCPUs |
| RAM | At least 64 GB allocated |
| GPU | Exactly one RTX 4090 with approximately 24 GB VRAM |
| GPU power | At least 400 W |
| PCIe | Gen4 x16 capability and offer `pcie_bw >= 20` GB/s |
| Throttling | Thermal and power-brake slowdown inactive |

Count physical cores only from online CPU IDs in `Cpus_allowed_list`, counting
unique `(CORE, SOCKET)` pairs. Fail if the allowed cpuset cannot be established.
Interpret finite cgroup CPU and memory limits as the allocation gates; use
visible memory only when the cgroup limit is unlimited. An idle PCIe link may
downshift, so maximum Gen4 x16 capability plus the passing offer measurement is
sufficient unless other evidence indicates restriction.

Compare CPU model, logical CPUs, RAM, GPU identity, GPU power, and PCIe with the
recorded offer and list every mismatch. Disk and network values come from the
offer; do not claim the SSH checks measured workload, disk, or upload speed.

On failure, automatically destroy the provisional instance, verify removal,
then return to Step 1 automatically to search for and provision a replacement.
Do not seek user permission for the retry.
### Network quality acceptance (runs after hardware acceptance)

Hardware checks measure local compute resources.  A separate network test
measures *actual* upload speed and complete PUT operation latency to S3 (where
checkpoints, runs, results, and dagger data will be stored). This catches instances
where the advertised `inet_up` in the offer is misleading due to geographic
distance, ISP throttling, or host oversubscription.

Generate presigned S3 URLs locally, then transfer and invoke the gate script:

```bash
TEMP_NETGATE_KEY="results/flywheel/${FLYWHEEL_ARCH}/.netgate/${UNIX_TIME_NS}"
S3_PRESIGNED_PUT=$(uv run python -c "
import boto3, os
s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION') or 'ap-south-1')
url = s3.generate_presigned_url('put_object',
    Params={'Bucket': os.environ['S3_BUCKET'], 'Key': '${TEMP_NETGATE_KEY}'},
    ExpiresIn=900)
print(url)
")
S3_PRESIGNED_DELETE=$(uv run python -c "
import boto3, os
s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION') or 'ap-south-1')
url = s3.generate_presigned_url('delete_object',
    Params={'Bucket': os.environ['S3_BUCKET'], 'Key': '${TEMP_NETGATE_KEY}'},
    ExpiresIn=900)
print(url)
")

B64_NG=$(base64 -w0 .opencode/commands/scripts/flywheel-4090/network-gate.sh)

NETGATE_OUTPUT=$(S3_PRESIGNED_PUT="$S3_PRESIGNED_PUT" \
  S3_PRESIGNED_DELETE="$S3_PRESIGNED_DELETE" \
  ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "printf '%s' '${B64_NG}' | base64 -d > /tmp/network-gate.sh && \
   chmod +x /tmp/network-gate.sh && \
   S3_PRESIGNED_PUT='${S3_PRESIGNED_PUT}' \
   S3_PRESIGNED_DELETE='${S3_PRESIGNED_DELETE}' \
   bash /tmp/network-gate.sh" 2>/dev/null)

echo "$NETGATE_OUTPUT"

if ! echo "$NETGATE_OUTPUT" | grep -q "^PASSED"; then
  echo "ERROR: Network gate rejected this instance."
  ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
    'rm -f /tmp/network-gate.sh /tmp/.netgate-test.bin; exit 0' 2>/dev/null || true
  INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | \
    INSTANCE_LABEL="$INSTANCE_LABEL" uv run python -c '
import json, os, sys
data = json.load(sys.stdin)
if isinstance(data, dict):
    data = data.get("instances", [data])
for item in data:
    if item.get("label") == os.environ["INSTANCE_LABEL"]:
        print(item["id"])
        break
')
  test -n "$INSTANCE_ID" || {
    echo "ERROR: could not find instance with label $INSTANCE_LABEL"
    exit 1
  }
  vastai destroy instance -y "$INSTANCE_ID" 2>&1 || {
    echo "ERROR: vastai destroy failed for $INSTANCE_ID"
    exit 1
  }
  for i in $(seq 1 30); do
    instances=$(vastai show instances --raw 2>/dev/null || echo "[]")
    found=""
    count=$(echo "$instances" | INSTANCE_ID="$INSTANCE_ID" uv run python -c '
import json, os, sys
data = json.load(sys.stdin)
if isinstance(data, dict):
    data = data.get("instances", [data])
target = os.environ.get("INSTANCE_ID", "")
if not target:
    raise SystemExit(2)
ids = [item.get("id") for item in data if str(item.get("id")) == target]
print(len(ids))
' 2>/dev/null) || found="error"
    if [ "$count" = "0" ]; then
      echo "Instance $INSTANCE_ID destroyed and removed"
      break
    fi
    if [ "$found" = "error" ] || [ -z "$count" ]; then
      echo "ERROR: verification poll failed for $INSTANCE_ID"
      exit 1
    fi
    sleep 10
  done
  echo "Returning to Step 1 — searching for a new offer."
  exit 1
fi

ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  'rm -f /tmp/network-gate.sh /tmp/.netgate-test.bin; exit 0' 2>/dev/null || true
```

| Check | Requirement |
|---|---|
| S3 PUT operation latency | Median of 7 successful samples ≤ 5000 ms against the presigned bucket key |
| S3 upload | Median of 3 successful 4 MiB uploads ≥ 1000 KB/s to the bucket |

On network rejection, automatically destroy the provisional instance, verify
removal, then return to Step 1 to search for and provision a replacement.  Do
not seek user permission.  Once both hardware and network gates pass, display
the committed `workers`, `dataloader_workers`, `batch_size`, and all four
`final_eval_*` values and proceed with the launch automatically.  No
instance-side config override is permitted.

### 5. Clone and verify the environment

Clone only after all prior confirmations. Substitute the confirmed values:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "git clone --branch 'GIT_BRANCH' --single-branch \
     https://github.com/skpro19/toy-pickplace.git /workspace/toy-pickplace && \
   cd /workspace/toy-pickplace && \
   test \"\$(git branch --show-current)\" = 'GIT_BRANCH' && \
   test \"\$(git rev-parse HEAD)\" = 'GIT_COMMIT' && \
   test -f 'EXPERIMENT_CONFIG' && \
   test -f 'EXPERIMENT_BASE_CONFIG' && \
   apt-get update -qq && apt-get install -y -qq \
     libgl1-mesa-glx libglib2.0-0 libegl1-mesa libgles2-mesa libglfw3 && \
   curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv sync --locked --directory /workspace/toy-pickplace && \
    MUJOCO_GL=egl /root/.local/bin/uv run python -c \
      'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import mujoco, glfw; print(mujoco.__version__, glfw.__version__)' && \
    MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      /root/.local/bin/uv run python tests/smoke_test_reproducibility.py"
```

If the checked-out commit differs, stop before setup and restart baseline
confirmation. The reproducibility smoke test must pass before launch; it compares
same-seed expert arrays and rendered images, then compares duplicate deterministic
training checkpoints. Configure tmux only; do not edit YAML:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "touch ~/.no_auto_tmux && printf '%s\n' 'set -g mouse on' > ~/.tmux.conf"
```

### 6. Start TensorBoard and local wrappers

Start one remote TensorBoard session:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "! tmux has-session -t tensorboard 2>/dev/null && \
   ! ss -ltn 'sport = :6006' | grep -q LISTEN && \
   tmux new-session -d -s tensorboard \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main \
      --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'"
```

Poll its local endpoint for up to 12 attempts at five-second intervals. Stop
before launch if the session exits or `http://127.0.0.1:6006/` does not become
reachable:

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

Immediately claim the lowest unused non-negative `LOCAL_WORKFLOW_INDEX` with
`local-wrapper-lease.sh`. The helper holds
`/tmp/toy-pickplace-flywheel-local-wrapper.lock` and atomically writes the
ownership file before returning the index. Set:

```text
LOCAL_TB_PORT=6006 + LOCAL_WORKFLOW_INDEX
LOCAL_SSH_SESSION=vast-ssh-LOCAL_WORKFLOW_INDEX
LOCAL_TB_SESSION=tb-flywheel-LOCAL_WORKFLOW_INDEX
```

An index is occupied if either wrapper, its supervisor, or its port exists, or
if an ownership file remains. The ownership file prevents another workflow
from killing or reusing active wrappers.

Never pre-clean a fixed index, kill another index's sessions, or delete an
ownership file to make an index available. This prohibition applies even when
sessions appear stale and during retries or recovery. Always claim a different
free index. Before the supervisor starts, a failed claim may only be cleaned up
with `local-wrapper-lease.sh release-unstarted`, which verifies ownership and
refuses to act if a supervisor exists.

Claim the lease, then create the wrappers:

```bash
set -e
mapfile -t LOCAL_WRAPPER_LEASE < <(
  .opencode/commands/scripts/flywheel-4090/local-wrapper-lease.sh \
    allocate "$RUN_NAME"
)
test "${#LOCAL_WRAPPER_LEASE[@]}" -eq 4
LOCAL_WORKFLOW_INDEX=${LOCAL_WRAPPER_LEASE[0]}
LOCAL_SSH_SESSION=${LOCAL_WRAPPER_LEASE[1]}
LOCAL_TB_SESSION=${LOCAL_WRAPPER_LEASE[2]}
LOCAL_TB_PORT=${LOCAL_WRAPPER_LEASE[3]}
unset LOCAL_WRAPPER_LEASE
release_local_wrapper_lease_on_error() {
  exit_code=$?
  if test "$exit_code" -ne 0; then
    .opencode/commands/scripts/flywheel-4090/local-wrapper-lease.sh \
      release-unstarted "$RUN_NAME" "$LOCAL_WORKFLOW_INDEX" || true
  fi
  exit "$exit_code"
}
trap release_local_wrapper_lease_on_error EXIT
tmux new-session -d -s "$LOCAL_SSH_SESSION" \
  "ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
    -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -p $PORT root@$HOST"
tmux new-session -d -s "$LOCAL_TB_SESSION" \
  "ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
    -o ConnectTimeout=15 -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N \
    -L $LOCAL_TB_PORT:127.0.0.1:6006 -p $PORT root@$HOST"
tmux has-session -t "$LOCAL_SSH_SESSION"
tmux has-session -t "$LOCAL_TB_SESSION"
LOCAL_SSH_READY=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
  -o ConnectTimeout=15 -p "$PORT" "root@$HOST" \
  "printf '%s\n' SSH_WRAPPER_OK" 2>/dev/null | tail -1)
test "$LOCAL_SSH_READY" = "SSH_WRAPPER_OK"
LOCAL_TB_READY=false
for i in $(seq 1 12); do
  if tmux has-session -t "$LOCAL_TB_SESSION" 2>/dev/null &&
    curl -fsS "http://127.0.0.1:$LOCAL_TB_PORT/" >/dev/null; then
    LOCAL_TB_READY=true
    break
  fi
  sleep 1
done
test "$LOCAL_TB_READY" = true
trap - EXIT
```

Print `http://localhost:$LOCAL_TB_PORT`. Stop before launch and report the
relevant local tmux output if either wrapper fails. Do not ask the user to
choose an index.

### 7. Materialize and launch the durable runner

Create these paths on the instance:

```text
CONTROL_DIR/
CONTROL_DIR/logs/
CONTROL_DIR/state/
CONTROL_DIR/state/history/
```

Before creating control state or writing provenance, reject every run-specific
artifact collision. Require a success sentinel so an SSH failure cannot be
misread as an empty collision list. After that check, create the control paths
exactly once and persist a launch authorization marker:

```bash
RUN_COLLISION_OUTPUT=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
  -p "$PORT" "root@$HOST" \
  "collisions=''
   for path in \
     data/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME} \
     checkpoints/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME} \
     runs/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME} \
     results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}; do
       test ! -e \"/workspace/toy-pickplace/\$path\" || \
         collisions=\"\${collisions}\${path} \"
   done
   printf 'COLLISIONS_OK:%s\\n' \"\$collisions\"" 2>/dev/null | tail -1)
case "$RUN_COLLISION_OUTPUT" in
  COLLISIONS_OK:*) ;;
  *)
    echo "ERROR: run collision probe failed" >&2
    .opencode/commands/scripts/flywheel-4090/destroy-instance.sh "$INSTANCE_ID"
    exit 1
    ;;
esac
RUN_COLLISIONS=${RUN_COLLISION_OUTPUT#COLLISIONS_OK:}
test -z "$RUN_COLLISIONS" || {
  printf 'ERROR: run artifact collisions:\n%s\n' "$RUN_COLLISIONS" >&2
  .opencode/commands/scripts/flywheel-4090/destroy-instance.sh "$INSTANCE_ID"
  exit 1
}

ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "test ! -e '${CONTROL_DIR}' && \
   mkdir -p '${CONTROL_DIR}/logs' '${CONTROL_DIR}/state/history'"
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "printf '%s\n' authorized > '${CONTROL_DIR}/state/launch-authorized.tmp' && \
   mv '${CONTROL_DIR}/state/launch-authorized.tmp' \
      '${CONTROL_DIR}/state/launch-authorized'"
```

Resolve the config on the remote instance using the checked-out files and
verify it matches the locally-planned SHA-256. Stop and destroy the instance
on mismatch:

```bash
REMOTE_SHA256=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "cd /workspace/toy-pickplace && /root/.local/bin/uv run python scripts/resolve_experiment.py \
   --experiment '$EXPERIMENT_CONFIG' --seed '$SELECTED_SEED' --sha256" 2>/dev/null | tail -1)
test "$REMOTE_SHA256" = "$RESOLVED_CONFIG_SHA256" || {
  printf 'ERROR: Remote resolved config SHA-256 mismatch\n  Local:  %s\n  Remote: %s\n' \
    "$RESOLVED_CONFIG_SHA256" "$REMOTE_SHA256" >&2
  .opencode/commands/scripts/flywheel-4090/destroy-instance.sh "$INSTANCE_ID"
  exit 1
}

ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "cd /workspace/toy-pickplace && /root/.local/bin/uv run python scripts/resolve_experiment.py \
   --experiment '$EXPERIMENT_CONFIG' --seed '$SELECTED_SEED' \
   > '${CONTROL_DIR}/resolved-config.yaml'"
```

Set `FLYWHEEL_CONFIG=${CONTROL_DIR}/resolved-config.yaml` for the runner.

Before launching, preserver provenance artifacts under the results directory
so the normal backup uploads them. Write the resolved config and a JSON
manifest capturing the exact experiment origin:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "mkdir -p /workspace/toy-pickplace/results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME} &&
   cp '${CONTROL_DIR}/resolved-config.yaml' \
     /workspace/toy-pickplace/results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/resolved-config.yaml"

MANIFEST_JSON=$(
  uv run python scripts/resolve_experiment.py \
    --experiment "$EXPERIMENT_CONFIG" --seed "$SELECTED_SEED" --json-manifest
)
MANIFEST_SHA256=$(printf '%s' "$MANIFEST_JSON" | sha256sum | cut -d ' ' -f1)
MANIFEST_B64=$(printf '%s' "$MANIFEST_JSON" | base64 -w0)
test -n "$MANIFEST_B64" || { echo 'ERROR: empty manifest base64' >&2; exit 1; }

ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "printf '%s' '${MANIFEST_B64}' | base64 -d > /workspace/toy-pickplace/results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/experiment-manifest.json.tmp && \
   mv /workspace/toy-pickplace/results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/experiment-manifest.json.tmp /workspace/toy-pickplace/results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/experiment-manifest.json" || {
     echo 'ERROR: manifest transfer failed' >&2; exit 1
   }

REMOTE_MANIFEST_SHA256=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "sha256sum /workspace/toy-pickplace/results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/experiment-manifest.json | cut -d ' ' -f1" 2>/dev/null | tail -1)
test "$REMOTE_MANIFEST_SHA256" = "$MANIFEST_SHA256" || {
  printf 'ERROR: manifest SHA-256 mismatch\n  Local:  %s\n  Remote: %s\n' "$MANIFEST_SHA256" "$REMOTE_MANIFEST_SHA256" >&2
  exit 1
}
```

Instantiate the runner template from `.opencode/commands/scripts/flywheel-4090/runner.sh`,
base64-encode it, decode it into `CONTROL_DIR`, and start it:

```bash
set -o pipefail
RUNNER_RENDERED=$(
  sed \
    -e "s|__RUN_NAME__|${RUN_NAME}|g" \
    -e "s|__FLYWHEEL_CONFIG__|${FLYWHEEL_CONFIG}|g" \
    -e "s|__CONTROL_DIR__|${CONTROL_DIR}|g" \
    -e "s|__ARCH__|${FLYWHEEL_ARCH}|g" \
    .opencode/commands/scripts/flywheel-4090/runner.sh
)
RUNNER_SHA256=$(printf '%s' "$RUNNER_RENDERED" | sha256sum | cut -d ' ' -f1)
RUNNER_B64=$(printf '%s' "$RUNNER_RENDERED" | base64 -w0)
test -n "$RUNNER_B64" || { echo 'ERROR: empty runner base64' >&2; exit 1; }

ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "printf '%s' '${RUNNER_B64}' | base64 -d > '${CONTROL_DIR}/runner.sh.tmp' && \
   chmod +x '${CONTROL_DIR}/runner.sh.tmp' && \
   mv '${CONTROL_DIR}/runner.sh.tmp' '${CONTROL_DIR}/runner.sh'" || {
     echo 'ERROR: runner transfer failed' >&2; exit 1
   }

REMOTE_RUNNER_SHA256=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "sha256sum '${CONTROL_DIR}/runner.sh' | cut -d ' ' -f1" 2>/dev/null | tail -1)
test "$REMOTE_RUNNER_SHA256" = "$RUNNER_SHA256" || {
  printf 'ERROR: runner SHA-256 mismatch\n  Local:  %s\n  Remote: %s\n' "$RUNNER_SHA256" "$REMOTE_RUNNER_SHA256" >&2
  exit 1
}

if ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "grep -q '__FLYWHEEL_CONFIG__\|__RUN_NAME__\|__CONTROL_DIR__\|__ARCH__' '${CONTROL_DIR}/runner.sh'" 2>/dev/null; then
  echo 'ERROR: unresolved placeholder in runner.sh' >&2; exit 1
fi
```

The executable runner must:

1. use `set -o pipefail` and export `MUJOCO_GL=egl` and
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts;
2. validate its config, control paths, and log before spawning the process;
3. persist the process-group PID, verify it survives initialization, then
   atomically write `running` to `state/run-status`;
4. run exactly `/root/.local/bin/uv run python scripts/flywheel.py --config
   FLYWHEEL_CONFIG --run-name RUN_NAME` from `/workspace/toy-pickplace`;
5. stream stdout and stderr to both the pane and `logs/RUN_NAME.log` with
   `tee -a`;
6. preserve the flywheel process's exit code;
7. atomically write `succeeded 0` and `state/completed` on success, or
   `failed EXIT_CODE` and `state/failed` on failure;
8. trap abnormal shell termination and write `state/failed` unless a terminal
   marker already exists;
9. exit with the flywheel process's exit code.

Every state update must use a temporary file followed by `mv`. A missing tmux
session is never a success signal. The persisted log and state files are
authoritative.

Before the first launch, fail if any of these run-specific artifact paths
already exists, even if the control directory is new:

```text
data/flywheel/${FLYWHEEL_ARCH}/RUN_NAME
checkpoints/flywheel/${FLYWHEEL_ARCH}/RUN_NAME
runs/flywheel/${FLYWHEEL_ARCH}/RUN_NAME
results/flywheel/${FLYWHEEL_ARCH}/RUN_NAME
```

List every collision and require a new run name for a new launch. Existing
artifacts may be inspected during recovery, but this workflow never relaunches
`flywheel.py` into them. Never treat an unrelated existing run as a recovery
target. Start the runner once in remote session `flywheel-run`. Refuse to launch
if the session or any run/control artifact already exists. Poll until
`state/run-status` says `running`, verify `flywheel-run` and `tensorboard` exist,
then require both to survive an additional five-second initialization window.
Stop if the runner exits or writes `state/failed`. The runner may continue after
the agent disconnects.

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "! tmux has-session -t flywheel-run 2>/dev/null && \
   test \"\$(cat '${CONTROL_DIR}/state/launch-authorized')\" = authorized && \
   test ! -e '${CONTROL_DIR}/state/run-status' && \
   test ! -e '${CONTROL_DIR}/state/completed' && \
   test ! -e '${CONTROL_DIR}/state/failed' && \
   tmux new-session -d -s flywheel-run \
     'cd /workspace/toy-pickplace && exec bash ${CONTROL_DIR}/runner.sh'"

RUNNER_READY=false
for i in $(seq 1 30); do
  runner_state=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
    -p "$PORT" "root@$HOST" \
    "test -f '${CONTROL_DIR}/state/failed' && cat '${CONTROL_DIR}/state/failed' || \
     { test -f '${CONTROL_DIR}/state/run-status' && cat '${CONTROL_DIR}/state/run-status' || echo starting; }" \
    2>/dev/null | tail -1)
  test "$runner_state" != "starting" || {
    ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
      "tmux has-session -t flywheel-run 2>/dev/null" || break
    sleep 1
    continue
  }
  test "$runner_state" = "running" || break
  RUNNER_READY=true
  break
done
test "$RUNNER_READY" = true
sleep 5
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux has-session -t flywheel-run 2>/dev/null && \
   tmux has-session -t tensorboard 2>/dev/null && \
   test \"\$(cat '${CONTROL_DIR}/state/run-status')\" = running && \
   test ! -e '${CONTROL_DIR}/state/failed'"
```

### 8. Start checkpoint backup

After the run is marked `running`, start one `ckpt-bkp` session. Its wrapper
must atomically publish `state/backup-running`, then wait until a
training-produced artifact exists specifically under
`checkpoints/flywheel/${FLYWHEEL_ARCH}/RUN_NAME`, `runs/flywheel/${FLYWHEEL_ARCH}/RUN_NAME`,
or `data/flywheel/${FLYWHEEL_ARCH}/RUN_NAME`. Provenance-only result files do
not satisfy this gate. Fail if no artifact appears within ten minutes. After
the gate, run every 120 seconds:

```bash
/root/.local/bin/uv run python scripts/s3_backup.py upload \
  --components checkpoints,runs,results,dagger \
  --arch "$FLYWHEEL_ARCH" \
  "$RUN_NAME"
```

The positional `RUN_NAME` is mandatory. Never upload whole component roots;
that can mix artifacts from other runs and can let old files satisfy the new
run's initial-backup gate.

Before each upload, atomically update `state/backup-cycle-started` with a unique
cycle ID and `state/backup-heartbeat`, and capture any
`state/backup-final-requested` token. Bound each upload attempt to 30 minutes.
After a successful
upload, atomically update `state/backup-last-succeeded` with the cycle ID and
acknowledge the captured final token in `state/backup-final-succeeded`. This
request/acknowledgement ensures an in-progress backup cannot be mistaken for a
post-completion backup. On failure, atomically write `state/backup-failed` and
exit non-zero.

Transfer credentials to the instance by writing an env file. Resolve the AWS
profile credentials locally first, then write the file via SSH using an
unquoted heredoc so variables expand locally. Never place credentials in the
SSH command string, wrapper script file, tmux global environment, pane output,
or logs. Use the absolute `uv` path. Do not include `S3_ENDPOINT_URL` — an
empty value causes `Invalid endpoint`.

```bash
# resolve profile credentials if needed
if test -n "${AWS_PROFILE:-}"; then
  mapfile -t RESOLVED_AWS < <(uv run python -c '
import os, boto3
s = boto3.Session(profile_name=os.environ["AWS_PROFILE"])
c = s.get_credentials().get_frozen_credentials()
print(c.access_key); print(c.secret_key); print(c.token or ""); print(s.region_name or "")
') || exit 1
  AWS_ACCESS_KEY_ID=${RESOLVED_AWS[0]}; AWS_SECRET_ACCESS_KEY=${RESOLVED_AWS[1]}
  AWS_SESSION_TOKEN=${RESOLVED_AWS[2]}; AWS_REGION=${RESOLVED_AWS[3]}
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_REGION
fi
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "cat > '${CONTROL_DIR}/s3-env.env' << ENVEOF
S3_BUCKET=${S3_BUCKET}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
AWS_REGION=${AWS_REGION}
ENVEOF
chmod 600 '${CONTROL_DIR}/s3-env.env'"
```

The wrapper uses `uv run --env-file` so credentials never appear on the command
line or in the script itself. Instantiate, transfer, and start the backup:

```bash
set -o pipefail
BACKUP_RENDERED=$(
  sed \
    -e "s|__RUN_NAME__|${RUN_NAME}|g" \
    -e "s|__CONTROL_DIR__|${CONTROL_DIR}|g" \
    -e "s|__ARCH__|${FLYWHEEL_ARCH}|g" \
    .opencode/commands/scripts/flywheel-4090/ckpt-bkp-wrapper.sh
)
BACKUP_SHA256=$(printf '%s' "$BACKUP_RENDERED" | sha256sum | cut -d ' ' -f1)
BACKUP_B64=$(printf '%s' "$BACKUP_RENDERED" | base64 -w0)
test -n "$BACKUP_B64" || { echo 'ERROR: empty backup base64' >&2; exit 1; }

ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "printf '%s' '${BACKUP_B64}' | base64 -d > '${CONTROL_DIR}/ckpt-bkp-wrapper.sh.tmp' && \
   chmod +x '${CONTROL_DIR}/ckpt-bkp-wrapper.sh.tmp' && \
   mv '${CONTROL_DIR}/ckpt-bkp-wrapper.sh.tmp' '${CONTROL_DIR}/ckpt-bkp-wrapper.sh'" || {
     echo 'ERROR: backup wrapper transfer failed' >&2; exit 1
   }

REMOTE_BACKUP_SHA256=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "sha256sum '${CONTROL_DIR}/ckpt-bkp-wrapper.sh' | cut -d ' ' -f1" 2>/dev/null | tail -1)
test "$REMOTE_BACKUP_SHA256" = "$BACKUP_SHA256" || {
  printf 'ERROR: backup wrapper SHA-256 mismatch\n  Local:  %s\n  Remote: %s\n' "$BACKUP_SHA256" "$REMOTE_BACKUP_SHA256" >&2
  exit 1
}

if ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "grep -q '__RUN_NAME__\|__CONTROL_DIR__\|__ARCH__' '${CONTROL_DIR}/ckpt-bkp-wrapper.sh'" 2>/dev/null; then
  echo 'ERROR: unresolved placeholder in ckpt-bkp-wrapper.sh' >&2; exit 1
fi
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s ckpt-bkp \
     'cd /workspace/toy-pickplace && exec bash ${CONTROL_DIR}/ckpt-bkp-wrapper.sh'"
```

Refuse to start if `ckpt-bkp` already exists on the remote host.

Poll for up to ten minutes for `state/backup-running`,
`state/backup-artifact-ready`, and `state/backup-last-succeeded`. Fail immediately
if `state/backup-failed` appears or `ckpt-bkp` exits. After success, inspect the
last ten pane lines without exposing credentials and verify the session remains
active. Use fixed remote paths in the status probe. Do not construct a remote
loop whose variable is embedded in a single-quoted path: the remote shell would
test a literal `$variable` filename instead of expanding it.

```bash
INITIAL_BACKUP_READY=false
for i in $(seq 1 120); do
  BACKUP_STATUS=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
    -p "$PORT" "root@$HOST" \
    "if test -f '${CONTROL_DIR}/state/backup-failed'; then
       printf '%s\n' BACKUP_FAILED
     elif ! tmux has-session -t ckpt-bkp 2>/dev/null; then
       printf '%s\n' BACKUP_STOPPED
     else
       running=0
       artifact_ready=0
       last_succeeded=0
       test ! -f '${CONTROL_DIR}/state/backup-running' || running=1
       test ! -f '${CONTROL_DIR}/state/backup-artifact-ready' || artifact_ready=1
       test ! -f '${CONTROL_DIR}/state/backup-last-succeeded' || last_succeeded=1
       printf 'BACKUP_STATE:%s:%s:%s\n' \
         \"\$running\" \"\$artifact_ready\" \"\$last_succeeded\"
     fi" 2>/dev/null | tail -1)

  case "$BACKUP_STATUS" in
    BACKUP_FAILED)
      echo "ERROR: backup-failed marker appeared" >&2
      exit 1
      ;;
    BACKUP_STOPPED)
      echo "ERROR: ckpt-bkp session exited" >&2
      exit 1
      ;;
    BACKUP_STATE:1:1:1)
      INITIAL_BACKUP_READY=true
      break
      ;;
    BACKUP_STATE:[01]:[01]:[01]) ;;
    *)
      printf 'ERROR: invalid backup status probe result: %s\n' \
        "$BACKUP_STATUS" >&2
      exit 1
      ;;
  esac
  sleep 5
done
test "$INITIAL_BACKUP_READY" = true

# The pane may be sitting on blank rows while the wrapper sleeps. Read enough
# scrollback to find the latest non-blank output, then verify the session again.
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "set -o pipefail
   tmux capture-pane -t ckpt-bkp -p -S -200 2>/dev/null | \
      sed '/^[[:space:]]*$/d' | tail -10
   tmux has-session -t ckpt-bkp 2>/dev/null" 2>/dev/null
```

Verify that all provenance files exist on S3:

```bash
for key in "results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/resolved-config.yaml" \
           "results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/experiment-manifest.json" \
           "results/flywheel/${FLYWHEEL_ARCH}/${RUN_NAME}/environment-manifest.json"; do
  uv run python -c "
import boto3, os
c = boto3.client('s3', region_name='${AWS_REGION:-ap-south-1}')
r = c.head_object(Bucket='${S3_BUCKET}', Key='$key')
print(f'Provenance OK: s3://${S3_BUCKET}/\$key ({r[\"ContentLength\"]} bytes)')
" || { echo "ERROR: Provenance file missing on S3: $key" >&2; exit 1; }
done
```

A later `backup-failed` marker or unexpected stopped session is a
workflow failure even if an earlier upload succeeded. If the initial backup
fails before the local supervisor starts, upload diagnostics best-effort,
destroy the instance, and verify removal directly.

### 9. Materialize automatic held-out evaluation

Instantiate `CONTROL_DIR/heldout-eval.sh` from the template and transfer it to
the instance before starting the local supervisor:

```bash
set -o pipefail
HELDOUT_RENDERED=$(
  sed \
    -e "s|__RUN_NAME__|${RUN_NAME}|g" \
    -e "s|__CONTROL_DIR__|${CONTROL_DIR}|g" \
    -e "s|__ARCH__|${FLYWHEEL_ARCH}|g" \
    .opencode/commands/scripts/flywheel-4090/heldout-eval.sh
)
HELDOUT_SHA256=$(printf '%s' "$HELDOUT_RENDERED" | sha256sum | cut -d ' ' -f1)
HELDOUT_B64=$(printf '%s' "$HELDOUT_RENDERED" | base64 -w0)
test -n "$HELDOUT_B64" || { echo 'ERROR: empty heldout base64' >&2; exit 1; }

ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "printf '%s' '${HELDOUT_B64}' | base64 -d > '${CONTROL_DIR}/heldout-eval.sh.tmp' && \
   chmod +x '${CONTROL_DIR}/heldout-eval.sh.tmp' && \
   mv '${CONTROL_DIR}/heldout-eval.sh.tmp' '${CONTROL_DIR}/heldout-eval.sh'" || {
     echo 'ERROR: heldout eval transfer failed' >&2; exit 1
   }

REMOTE_HELDOUT_SHA256=$(ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "sha256sum '${CONTROL_DIR}/heldout-eval.sh' | cut -d ' ' -f1" 2>/dev/null | tail -1)
test "$REMOTE_HELDOUT_SHA256" = "$HELDOUT_SHA256" || {
  printf 'ERROR: heldout eval SHA-256 mismatch\n  Local:  %s\n  Remote: %s\n' "$HELDOUT_SHA256" "$REMOTE_HELDOUT_SHA256" >&2
  exit 1
}

if ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "grep -q '__RUN_NAME__\|__CONTROL_DIR__\|__ARCH__' '${CONTROL_DIR}/heldout-eval.sh'" 2>/dev/null; then
  echo 'ERROR: unresolved placeholder in heldout-eval.sh' >&2; exit 1
fi
```

The template at `.opencode/commands/scripts/flywheel-4090/heldout-eval.sh`
implements the full specification (validation, scoring, upload, verification)
and embeds the trailing-slash fix for the S3 key prefix. Read it before
proceeding if unfamiliar. Do not use a component-wide results upload after
evaluation — it can delete or mix objects outside this exact output set.

### 10. Start the durable local supervisor

The supervisor lives at `.opencode/commands/scripts/flywheel-4090/supervisor.sh`.
It implements the full state machine, diagnostic upload, endpoint refresh, and
cleanup lifecycle. Read it before proceeding if unfamiliar.

Start `LOCAL_SUPERVISOR_SESSION` only while the atomic lease still belongs to
this `RUN_NAME`. Pass all eight values as quoted positional arguments; do not
rely on environment-prefix assignments because shell expansion occurs before
those assignments apply. Persist pane output before checking readiness:

```bash
set -e
exec 9>/tmp/toy-pickplace-flywheel-local-wrapper.lock
flock 9

LOCAL_SUPERVISOR_SESSION="flywheel-supervisor-$LOCAL_WORKFLOW_INDEX"
LOCAL_SUPERVISOR_LOG="/tmp/toy-pickplace-flywheel-${RUN_NAME}.supervisor.log"
LOCAL_SUPERVISOR_STATUS="/tmp/toy-pickplace-flywheel-${RUN_NAME}.status"
LOCAL_OWNER_FILE="/tmp/toy-pickplace-flywheel-$LOCAL_WORKFLOW_INDEX.owner"
PROJECT_ROOT=$(pwd -P)

tmux has-session -t "$LOCAL_SSH_SESSION" || {
  echo "ERROR: SSH wrapper session $LOCAL_SSH_SESSION died before supervisor start" >&2
  exit 1
}
tmux has-session -t "$LOCAL_TB_SESSION" || {
  echo "ERROR: TB wrapper session $LOCAL_TB_SESSION died before supervisor start" >&2
  exit 1
}

test -f "$LOCAL_OWNER_FILE"
test "$(<"$LOCAL_OWNER_FILE")" = "$RUN_NAME"
rm -f "$LOCAL_SUPERVISOR_STATUS" "$LOCAL_SUPERVISOR_LOG"

printf -v SUPERVISOR_COMMAND '%q ' \
  "$PROJECT_ROOT/.opencode/commands/scripts/flywheel-4090/supervisor.sh" \
  "$INSTANCE_ID" "$HOST" "$PORT" "$RUN_NAME" "$CONTROL_DIR" \
  "$LOCAL_SSH_SESSION" "$LOCAL_TB_SESSION" "$FLYWHEEL_ARCH"
printf -v SUPERVISOR_LOG_QUOTED '%q' "$LOCAL_SUPERVISOR_LOG"
tmux new-session -d -s "$LOCAL_SUPERVISOR_SESSION" -c "$PROJECT_ROOT" \
  "set -o pipefail; ${SUPERVISOR_COMMAND} 2>&1 | tee -a ${SUPERVISOR_LOG_QUOTED}"

SUPERVISOR_READY=false
for i in $(seq 1 90); do
  if ! tmux has-session -t "$LOCAL_SUPERVISOR_SESSION" 2>/dev/null; then
    echo "ERROR: supervisor exited during startup" >&2
    test ! -f "$LOCAL_SUPERVISOR_LOG" || tail -30 "$LOCAL_SUPERVISOR_LOG"
    flock -u 9
    exec 9>&-
    .opencode/commands/scripts/flywheel-4090/local-wrapper-lease.sh \
      release-unstarted "$RUN_NAME" "$LOCAL_WORKFLOW_INDEX" || true
    .opencode/commands/scripts/flywheel-4090/destroy-instance.sh "$INSTANCE_ID"
    exit 1
  fi
  supervisor_outcome=$(uv run python -c '
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("not_found")
else:
    try:
        print(json.loads(path.read_text())["outcome"])
    except (json.JSONDecodeError, KeyError):
        print("invalid")
' "$LOCAL_SUPERVISOR_STATUS")
  case "$supervisor_outcome" in
    supervisor-ready|supervisor-running)
      SUPERVISOR_READY=true
      break
      ;;
    supervisor-starting|not_found) ;;
    *)
      echo "ERROR: supervisor startup state: $supervisor_outcome" >&2
      test ! -f "$LOCAL_SUPERVISOR_LOG" || tail -30 "$LOCAL_SUPERVISOR_LOG"
      exit 1
      ;;
  esac
  sleep 1
done
test "$SUPERVISOR_READY" = true

# Require one complete 30-second monitor cycle, not merely a spawned process.
sleep 35
tmux has-session -t "$LOCAL_SUPERVISOR_SESSION"
supervisor_outcome=$(uv run python -c '
import json, sys
print(json.load(open(sys.argv[1]))["outcome"])
' "$LOCAL_SUPERVISOR_STATUS")
test "$supervisor_outcome" = "supervisor-running"

flock -u 9
exec 9>&-
```

Verify the supervisor is alive, has completed one monitor cycle, and print its
attach command. On startup failure, inspect the persistent supervisor log and
retry only after correcting the cause. Never transfer
`VAST_API_KEY` to the instance. The local machine and tmux server must remain
running; a terminal or OpenCode disconnect is safe, but a local reboot is not.

### 11. Follow-up download and replay

Print these as optional commands only; do not download automatically:

```bash
set -a; . ./.env; set +a
uv run python scripts/s3_backup.py download --arch "$FLYWHEEL_ARCH" RUN_NAME
uv run python scripts/final_score.py --run-name RUN_NAME --arch "$FLYWHEEL_ARCH"
uv run python scripts/final_score.py --run-name RUN_NAME --plot-only
```

## Final output

Provisioning setup is complete after hardware acceptance, TensorBoard, local
wrappers, `flywheel-run`, initial backup, held-out wrapper, and local supervisor
are verified. The workflow is complete only when the supervisor reaches a
terminal state and verifies instance destruction. A successful workflow also
requires training completion, an acknowledged post-completion backup,
held-out completion, and all four result objects verified on S3.

Print:

- confirmed branch, commit, experiment path, base config path, selected
  seed, and arch;
- experiment description and resolved config SHA-256;
- run name (suite_experiment_seed_timestamp), instance label, instance ID,
  and SSH URL command;
- advertised effective vCPUs, verified physical cores, CPU quota, RAM, GPU,
  power, and PCIe acceptance results;
- committed `workers`, `dataloader_workers`, `batch_size`, and all four
  `final_eval_*` values;
- `CONTROL_DIR`, durable run state, log path, and remote attach command;
- local SSH, TensorBoard, and supervisor attach commands and TensorBoard URL;
- backup state, optional download and replay commands;
- held-out settings, state, log, and all four exact uploaded result paths;
- terminal supervisor status path and verified instance-destruction state.

## Safety notes

- Never store API keys, AWS credentials, tokens, or transient SSH endpoints in
  repository or remote workflow files.
- RTX 4090 offers are volatile. Use the confirmed shortlist directly with
  `--cancel-unavail` and stop after the first successful create.
- Advertised effective vCPUs never bypass the SSH physical-core gate.
- The committed experiment definition and base config are immutable for this
  workflow. The resolved config is deterministically materialized from those
  committed files and the selected committed seed. No parameter sweeps,
  uncommitted experiment overrides, concurrency calculations, or batches are
  allowed.
- Held-out evaluation always runs after an acknowledged post-completion backup
  and uploads finalized metrics, its JSON, and two plots under
  `results/flywheel/${FLYWHEEL_ARCH}/RUN_NAME/`.
- Every terminal state triggers destruction. Failure diagnostics are
  best-effort so an upload outage cannot keep a billed instance alive.
