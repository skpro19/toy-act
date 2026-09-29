---
description: Run an on-screen rollout of a run's latest S3 checkpoint on the local machine
agent: build
---

Download the latest `step_*.pt` checkpoint of a training run from
`s3://toy-act/checkpoints/act_v2/<run-name>/` and evaluate it locally by
invoking `scripts/rollout.py` in `PickPlaceCan`. This runs on the local machine
and needs a working display; it does not provision a Vast.ai instance.

## Arguments

`$ARGUMENTS` is a run name followed by optional overrides:

```
<run-name> [--ckpt <n>] [--episodes <n>] [--steps <n>] [--seed <n>] [--version act_v2]
```

- `<run-name>` — required; must be an existing run directory under
  `s3://toy-act/checkpoints/act_v2/`.
- `--ckpt <n>` — evaluate exactly step `<n>` (for example `20000` selects
  `step_000020000.pt`); the checkpoint must exist under the run's S3 prefix or
  the command stops with an error. When omitted, the highest-numbered
  `step_*.pt` is used.
- `--episodes <n>` — rollouts for the checkpoint (default 30).
- `--steps <n>` — maximum steps per episode, passed as `--horizon`
  (default 200).
- `--seed <n>` — random seed for env resets (default 0).
- `--version <v>` — model version; only `act_v2` is supported and it is the
  default.

Reject unknown flags, flags missing a value, a non-positive checkpoint/episode/step
count, a negative seed, and any `--version` other than `act_v2`.

## Fixed configuration

| Setting | Value |
|---|---|
| Model version | `act_v2` |
| Rendering | On-screen (GLFW), `--on-screen` |
| Episodes | 30 (override with `--episodes`) |
| Steps per episode | 200 (override with `--steps`) |
| Seed | 0 (override with `--seed`) |
| Dataset | `datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5` |
| AWS profile | `toy-pickplace-backup` |
| S3 region | `ap-south-1` |
| S3 checkpoint source | `s3://toy-act/checkpoints/act_v2/<run-name>/` |
| Local checkpoint dir | `checkpoints/act_v2/<run-name>/` |
| Latent `z` and image keys | from the checkpoint's embedded `config` |

## Workflow

1. Require `aws` and `uv`. Verify
   `aws sts get-caller-identity --profile toy-pickplace-backup` and
   `uv lock --check` succeed.
2. Parse `$ARGUMENTS`. Confirm the run name is present and the overrides are
   valid before downloading anything.
3. Require at least one `step_*.pt` under the run's S3 prefix; stop and report
   if none is present.
4. Resolve the checkpoint: with `--ckpt <n>` use `step_<n>.pt` (zero-padded to
   nine digits, as written by `scripts/train_v2.py`) and stop with an error if
   it is absent from the run's S3 prefix; otherwise select the highest-numbered
   `step_*.pt`. Download it to `checkpoints/act_v2/<run-name>/`, skipping the
   download when the local file already exists with the same size.
   `checkpoints/` is gitignored.
5. Run the on-screen rollout to completion, then report the checkpoint filename
   and the printed rollout summary. `scripts/rollout.py` reads `use_z` and the
   image keys from the checkpoint's embedded `config`, so no extra flags or S3
   config download are needed.

## Reference implementation

```bash
set -euo pipefail

RUN_NAME=<run-name>
VERSION=act_v2
CKPT=""
EPISODES=30
STEPS=200
SEED=0
# Parse "$@" into RUN_NAME / VERSION / CKPT / EPISODES / STEPS / SEED here.

PREFIX="checkpoints/${VERSION}/${RUN_NAME}"
DATASET="datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5"

if [ -n "$CKPT" ]; then
  CHECKPOINT_NAME=$(printf 'step_%09d.pt' "$CKPT")
  AWS_PROFILE=toy-pickplace-backup aws s3api head-object \
    --bucket toy-act --key "${PREFIX}/${CHECKPOINT_NAME}" --region ap-south-1 \
    >/dev/null \
    || { echo "no ${CHECKPOINT_NAME} under s3://toy-act/${PREFIX}/" >&2; exit 1; }
else
  CHECKPOINT_NAME=$(
    AWS_PROFILE=toy-pickplace-backup aws s3 ls "s3://toy-act/${PREFIX}/" --region ap-south-1 \
      | tr -s ' ' \
      | cut -d' ' -f4 \
      | grep -E '^step_[0-9]+\.pt$' \
      | sort -V \
      | tail -1
  )
  test -n "$CHECKPOINT_NAME" || { echo "no step_*.pt under s3://toy-act/${PREFIX}/" >&2; exit 1; }
fi

LOCAL_DIR="checkpoints/${VERSION}/${RUN_NAME}"
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

PYTHONPATH=. uv run python scripts/rollout.py \
  --checkpoint "$LOCAL_CHECKPOINT" \
  --dataset "$DATASET" \
  --n-rollouts "$EPISODES" \
  --horizon "$STEPS" \
  --seed "$SEED" \
  --on-screen
```

On-screen rendering needs a reachable X display (`DISPLAY`) and GLFW; if the
viewer cannot open, report the error rather than silently switching to
off-screen.
