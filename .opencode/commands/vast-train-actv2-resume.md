---
description: Resume an ACT v2 Vast.ai training run from its latest S3 checkpoint
agent: build
---

Resume a completed (or interrupted) ACT v2 training run on a newly provisioned
Vast.ai RTX 4090. The run continues from its latest checkpoint on S3 up to a new
total epoch count, writes checkpoints and TensorBoard logs under a **fresh** run
name/prefix, and leaves the old run's S3 objects untouched. The old TensorBoard
events file is copied into the new run directory so the new run's curve shows the
full history from 0 to the target epoch.

The instance follows the same `flywheel-4090` pattern as
`.opencode/commands/vast-train-actv2.md`: a durable runner owns the training
process, a backup wrapper syncs the new run to S3, TensorBoard is reachable
through a local SSH port-forward, and a detached local watcher owns the run to
completion including cleanup.

## Arguments

- `$ARGUMENTS` contains the old run directory name followed by `--epochs N`:

  ```
  <old-run-name> --epochs <N>
  ```

  Example:
  `20260927-102415_bs250_lr1e-04_beta0.01_beta_start0.001_wu80_ep400_seed0_ckpt10_use_z1_l1 --epochs 1000`

- `RESUME_RUN` — the exact old run directory name (first token).
- `TARGET_EPOCHS` — the new total epoch count (`--epochs` value).

## Shared workflow

Follow `.opencode/commands/vast-train-actv2.md` steps 1 through 15 (require
tools, pin commit, provision, hardware verification, clone, install, credentials,
dataset, TensorBoard, local tmux wrappers) unchanged, except:

- Skip the "Config selection" step. The resume command derives the training
  config from the old run's `config.json` on S3 instead of selecting a repo
  config file.
- Record the resolved values below (`RESUME_CKPT`, `START_EPOCH`,
  `CHECKPOINT_EVERY`, `FIRST_SNAPSHOT`, `TARGET_EPOCHS`) in `setup.env`.

## Resume resolution (before provisioning)

1. Require `RESUME_RUN` and `TARGET_EPOCHS`; error if missing or if
   `TARGET_EPOCHS` is not a positive integer.
2. List `s3://toy-act/checkpoints/act_v2/$RESUME_RUN/` and require at least one
   `epoch_NNN.pt`. Pick the highest epoch number: `RESUME_CKPT` is its filename
   and `START_EPOCH` is the parsed number.
3. Fetch `s3://toy-act/runs/act_v2/$RESUME_RUN/config.json` and read
   `config.checkpoint_every` into `CHECKPOINT_EVERY`. Require exactly one
   `events.out.tfevents.*` under `s3://toy-act/runs/act_v2/$RESUME_RUN/`.
4. Require `START_EPOCH < TARGET_EPOCHS`.
5. Compute `FIRST_SNAPSHOT` = the smallest multiple of `CHECKPOINT_EVERY`
   strictly greater than `START_EPOCH`.

## Restore step (after base step 13, before launching the runner)

6. On the instance, restore the old run's minimal state so
   `scripts/train_v2.py` can resolve it by name, using the transferred
   `.vast-train/s3-env.env` credentials:

   ```bash
   aws s3 cp "s3://toy-act/runs/act_v2/$RESUME_RUN/config.json" \
     "runs/act_v2/$RESUME_RUN/config.json"
   aws s3 cp "s3://toy-act/runs/act_v2/$RESUME_RUN/events.out.tfevents.*" \
     "runs/act_v2/$RESUME_RUN/"
   aws s3 cp "s3://toy-act/checkpoints/act_v2/$RESUME_RUN/$RESUME_CKPT" \
     "checkpoints/act_v2/$RESUME_RUN/$RESUME_CKPT"
   ```

   Only these three artifacts are needed. Do not restore the other snapshots.

## Launch runner (replaces base step 16)

7. Start the durable runner with the resume arguments:

   ```bash
   tmux new-session -d -s train \
     -e TRAIN_MODULE=scripts.train_v2 \
     -e TRAIN_RESUME=1 \
     -e TRAIN_OLD_RUN_NAME="$RESUME_RUN" \
     -e TRAIN_EPOCHS="$TARGET_EPOCHS" \
     'bash /workspace/toy-act/.opencode/commands/scripts/vast-train/runner.sh'
   ```

   Poll `state/run-status` = `running` as in the base flow. `train_v2.py` derives
   the config, latest checkpoint, and events file from `RESUME_RUN`, generates a
   fresh run name, and copies the old events file into the new run directory.

## Backup wrapper (replaces base step 17)

8. Start the backup wrapper with the old run excluded from discovery so it syncs
   only the new run:

   ```bash
   tmux new-session -d -s ckpt-bkp \
     -e CHECKPOINT_ROOT=checkpoints/act_v2 \
     -e RUNS_ROOT=runs/act_v2 \
     -e CKPT_BKP_IGNORE_RUN="$RESUME_RUN" \
     'bash /workspace/toy-act/.opencode/commands/scripts/vast-train/ckpt-bkp-wrapper.sh'
   ```

## Watcher verification (adjusts base step 18)

9. The generated watcher verifies only the resumed snapshots: expected files are
   `epoch_%03d.pt` for every `CHECKPOINT_EVERY` from `FIRST_SNAPSHOT` through
   `TARGET_EPOCHS`. Adjust the watcher's `verify_s3` loop to
   `seq "$FIRST_SNAPSHOT" "$CHECKPOINT_EVERY" "$TARGET_EPOCHS"` instead of
   starting at `CHECKPOINT_EVERY`.

10. Hand off to the detached local watcher exactly as in the base flow (step 18),
    then report the new run name (readable from `state/run-name`), S3 URIs, offer
    price, pinned commit, TensorBoard URL, and the local run-state directory
    `.vast-train-local/toy-act-<INSTANCE_ID>/`.

## Result

The old run's S3 prefix is never modified. The new run's TensorBoard directory
contains the copied old events file plus its own events, so TensorBoard shows one
continuous curve from 0 to `TARGET_EPOCHS`, while the new run's checkpoints span
`FIRST_SNAPSHOT` through `TARGET_EPOCHS`.
