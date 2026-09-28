---
description: Run an on-screen rollout of a run's latest S3 checkpoint on the local machine
agent: build
---

Download the latest `epoch_*.pt` checkpoint of a training run from
`s3://toy-act/checkpoints/act_v2/<run-name>/` and evaluate it locally by
invoking `scripts/rollout.py` in `PickPlaceCan`. This runs on the local machine
and needs a working display; it does not provision a Vast.ai instance.

## Arguments

`$ARGUMENTS` is a run name followed by optional overrides:

```
<run-name> [--episodes <n>] [--steps <n>] [--version act_v2]
```

- `<run-name>` — required; must be an existing run directory under
  `s3://toy-act/checkpoints/act_v2/`.
- `--episodes <n>` — rollouts for the checkpoint (default 10).
- `--steps <n>` — maximum steps per episode, passed as `--horizon`
  (default 150).
- `--version <v>` — model version; only `act_v2` is supported and it is the
  default.

Reject unknown flags, flags missing a value, a non-positive episode/step count,
and any `--version` other than `act_v2`.

## Fixed configuration

| Setting | Value |
|---|---|
| Model version | `act_v2` |
| Rendering | On-screen (GLFW), `--on-screen` |
| Episodes | 10 (override with `--episodes`) |
| Steps per episode | 150 (override with `--steps`) |
| Seed | 0 |
| Dataset | `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` |
| AWS profile | `toy-pickplace-backup` |
| S3 region | `ap-south-1` |
| S3 checkpoint source | `s3://toy-act/checkpoints/act_v2/<run-name>/` |
| S3 config source | `s3://toy-act/runs/act_v2/<run-name>/config.json` |
| Local checkpoint dir | `checkpoints/act_v2/<run-name>/` |
| Latent `z` | from the run's `config.json` `config.use_z` (default on) |

## Workflow

1. Require `aws` and `uv`. Verify
   `aws sts get-caller-identity --profile toy-pickplace-backup` and
   `uv lock --check` succeed.
2. Parse `$ARGUMENTS`. Confirm the run name is present and the overrides are
   valid before downloading anything.
3. Require at least one `epoch_*.pt` under the run's S3 prefix and the object
   `s3://toy-act/runs/act_v2/<run-name>/config.json`; stop and report if either
   is missing.
4. Select the highest `epoch_*.pt` and download it to
   `checkpoints/act_v2/<run-name>/`, skipping the download when the local file
   already exists with the same size. `checkpoints/` is gitignored.
5. Read `config.use_z` from the run's `config.json` to choose `--use-z` or
   `--no-use-z`.
6. Run the on-screen rollout to completion, then report the checkpoint filename
   and the printed rollout summary.

## Reference implementation

```bash
set -euo pipefail

RUN_NAME=<run-name>
VERSION=act_v2
EPISODES=10
STEPS=150
# Parse "$@" into RUN_NAME / VERSION / EPISODES / STEPS here.

PREFIX="checkpoints/${VERSION}/${RUN_NAME}"
DATASET="datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5"

CHECKPOINT_NAME=$(
  AWS_PROFILE=toy-pickplace-backup aws s3 ls "s3://toy-act/${PREFIX}/" --region ap-south-1 \
    | tr -s ' ' \
    | cut -d' ' -f4 \
    | grep -E '^epoch_[0-9]+\.pt$' \
    | sort -V \
    | tail -1
)
test -n "$CHECKPOINT_NAME" || { echo "no epoch_*.pt under s3://toy-act/${PREFIX}/" >&2; exit 1; }

LOCAL_DIR="checkpoints/act_v2/${RUN_NAME}"
LOCAL_CHECKPOINT="${LOCAL_DIR}/${CHECKPOINT_NAME}"
REMOTE_SIZE=$(
  AWS_PROFILE=toy-pickplace-backup aws s3api head-object \
    --bucket toy-act --key "${PREFIX}/${CHECKPOINT_NAME}" --region ap-south-1 \
    --query 'ContentLength' --output text
)
mkdir -p "$LOCAL_DIR"
if [ ! -f "$LOCAL_CHECKPOINT" ] || [ "$(stat -c%s "$LOCAL_CHECKPOINT")" != "$REMOTE_SIZE" ]; then
  AWS_PROFILE=toy-pickplace-backup aws s3 cp \
    "s3://toy-act/${PREFIX}/${CHECKPOINT_NAME}" "$LOCAL_CHECKPOINT" --region ap-south-1
fi

USE_Z=$(
  AWS_PROFILE=toy-pickplace-backup aws s3 cp \
    "s3://toy-act/runs/${VERSION}/${RUN_NAME}/config.json" - --region ap-south-1 \
    | uv run python -c "import json, sys; print('--use-z' if json.load(sys.stdin).get('config', {}).get('use_z', True) else '--no-use-z')"
)

PYTHONPATH=. uv run python scripts/rollout.py \
  --model-version "$VERSION" \
  --checkpoint "$LOCAL_CHECKPOINT" \
  --dataset "$DATASET" \
  --n-rollouts "$EPISODES" \
  --horizon "$STEPS" \
  "$USE_Z" \
  --on-screen
```

On-screen rendering needs a reachable X display (`DISPLAY`) and GLFW; if the
viewer cannot open, report the error rather than silently switching to
off-screen.
