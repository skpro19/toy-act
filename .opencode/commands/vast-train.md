---
description: Train ACT on a temporary Vast.ai RTX 4090 with S3 checkpoints
agent: build
---

Run `scripts/train.py` on a newly provisioned Vast.ai instance and store every
completed checkpoint in `s3://toy-act/checkpoints/act_v1/`. Provision the latest
commit of the `dev` branch by cloning GitHub on the instance; do not transfer the
local working tree. Fetch the dataset from S3 at its project-relative path
instead of copying it from the local machine.

The instance follows the `flywheel-4090` pattern: a durable runner owns the
training process, a separate backup wrapper synchronizes run-scoped checkpoints
and TensorBoard runs to S3, and TensorBoard is reachable from the local machine
through an SSH port-forward. The interactive agent is only responsible for
provisioning and handoff; a detached local watcher owns the run to completion,
including cleanup.

## Fixed configuration

| Setting | Value |
|---|---|
| GPU | One full RTX 4090 |
| Maximum price | $0.80/hour |
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| Disk | 100 GB |
| AWS profile | `toy-pickplace-backup` |
| S3 region | `ap-south-1` |
| S3 destination | `s3://toy-act/checkpoints/act_v1/` |
| Git remote | `https://github.com/skpro19/toy-act.git` |
| Git branch | `dev` |
| Dataset | `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` |
| Dataset S3 source | `s3://toy-act/datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` |
| Remote project | `/workspace/toy-act` |
| Remote control dir | `/workspace/toy-act/.vast-train` |
| Remote TensorBoard | `127.0.0.1:6006` on the instance |
| Instance label | `toy-act-train` |

## Workflow

1. Require `vastai`, `aws`, `jq`, `ssh`, `ssh-keyscan`, `rsync`, `git`, `tmux`,
   `flock`, and `ss`. Verify:
   - `vastai show instances --raw` succeeds;
   - `aws sts get-caller-identity --profile toy-pickplace-backup` succeeds;
   - listing `s3://toy-act/checkpoints/act_v1/` succeeds;
   - `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` exists locally;
   - `uv lock --check` succeeds.
2. Pin the code to the latest `dev` commit and confirm it is anonymously
   clonable, because the instance has no GitHub credentials:
   - `git fetch origin dev` and set `GIT_COMMIT=$(git rev-parse origin/dev)`;
   - `git ls-remote https://github.com/skpro19/toy-act.git dev` must succeed
     without prompting for a username. If it prompts, stop and ask the user to
     make the repository public; never embed tokens, keys, or credentials;
   - if the local working tree is dirty or local `dev` differs from `origin/dev`,
     warn the user that the clone will not include those local changes.
3. Refuse to continue if an instance with the exact label `toy-act-train`
   already exists. Never destroy or reuse an unrelated instance.
4. Search offers with the following hard filters:

   ```bash
   vastai search offers \
     'gpu_name=RTX_4090 num_gpus=1 gpu_ram>=24 compute_cap>=890 cpu_cores_effective>=8 cpu_ram>=32 inet_down>=200 inet_up>=100 reliability>=0.98 rentable=true verification=verified gpu_display_active=false' \
     --order dph_total+ --raw
   ```

   Do not add `gpu_frac=1`. Vast.ai defines `gpu_frac` as GPUs in the offer
   divided by GPUs in the host, so it rejects a full single 4090 on a multi-GPU
   host. `num_gpus=1` together with `gpu_ram>=24` already guarantees one full
   GPU.

5. Keep offers at or below `$0.80/hour`, reject EPYC 7001/7002, and rank by
   CPU family (EPYC 9005, EPYC 9004, Threadripper 7000, EPYC 7003, modern
   Ryzen 7000/9000), then price, disk bandwidth, and reliability. Automatically
   try up to the best three offers in order. Never weaken a filter without
   asking the user.
6. Create exactly one instance using the fixed image, disk, SSH direct mode,
   and label. Reconcile the instance by exact label after every create attempt;
   do not rely only on parsing create-command output.
7. Once an instance ID exists, the detached watcher described in step 17 owns an
   EXIT trap that destroys that exact instance and verifies it no longer appears
   in `vastai show instances --raw`. The trap must run on both success and
   failure. Do not install the trap in the interactive agent shell: returning
   from the agent must not destroy the running instance.
8. Wait up to ten minutes for `vastai ssh-url INSTANCE_ID` and successful SSH.
   Use a command-specific temporary `known_hosts` file populated by
   `ssh-keyscan`; then use `StrictHostKeyChecking=yes` for all SSH, rsync, and
   port-forwarding.
9. Clone the pinned commit on the instance and verify it:

   ```bash
   git clone --branch dev --single-branch \
     https://github.com/skpro19/toy-act.git /workspace/toy-act
   test "$(git -C /workspace/toy-act rev-parse HEAD)" = "$GIT_COMMIT"
   ```

   The clone already contains `scripts/`, the project files, and the
   `.opencode/commands/scripts/vast-train/` helpers (`runner.sh`,
   `ckpt-bkp-wrapper.sh`, `local-wrapper-lease.sh`).
10. On the instance, install `uv`, `awscli`, and `tmux`, then run
    `uv sync --frozen --only-group train` and verify `torch.cuda.is_available()`,
    printing the GPU name.
11. Resolve `toy-pickplace-backup` credentials locally with
    `aws configure export-credentials`. Write them to a mode-600 temporary env
    file without printing them, append `AWS_REGION` and `S3_BUCKET`, transfer it
    as `/workspace/toy-act/.vast-train/s3-env.env` (creating
    `/workspace/toy-act/.vast-train` first), chmod it 600 remotely, and delete
    the local temporary file. Never transfer `VAST_API_KEY` or print AWS
    credentials.
12. Ensure the dataset is in the bucket, then fetch it on the instance:
    - upload `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` to
      `s3://toy-act/datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5`
      with the workload profile, skipping the upload when the object already
      exists with the same size;
    - on the instance, download it from that key to the same project-relative
      path using the transferred credentials and installed `awscli`;
    - verify the remote SHA-256 matches the local file before launching.
13. Create the remote control directories
    `/workspace/toy-act/.vast-train/{logs,state}` and start TensorBoard in a
    detached tmux session named `tensorboard`:

    ```bash
    tmux new-session -d -s tensorboard \
      'cd /workspace/toy-act && exec /root/.local/bin/uv run --frozen \
       --only-group train python -m tensorboard.main \
       --logdir runs/act_v1 --host 127.0.0.1 --port 6006'
    ```

    Poll its endpoint from the instance for up to 12 attempts at five-second
    intervals and stop if the session exits or
    `http://127.0.0.1:6006/` never becomes reachable.
14. Claim the lowest unused local workflow index with
    `.opencode/commands/scripts/vast-train/local-wrapper-lease.sh allocate
    "toy-act-$INSTANCE_ID"`. It returns `INDEX`, `SSH_SESSION`, `TB_SESSION`, and
    `TB_PORT`. Create both local tmux wrappers:

    ```bash
    tmux new-session -d -s "$SSH_SESSION" \
      "ssh -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes \
       -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
       -p $PORT root@$HOST"
    tmux new-session -d -s "$TB_SESSION" \
      "ssh -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes \
       -o ConnectTimeout=15 -o ExitOnForwardFailure=yes \
       -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N \
       -L $TB_PORT:127.0.0.1:6006 -p $PORT root@$HOST"
    ```

    Verify both sessions exist and that `http://localhost:$TB_PORT/` responds;
    print that URL. Stop before launching and report the relevant local tmux
    output if either wrapper fails.
15. Start the durable runner in a detached tmux session named `train` and wait
    until it publishes readiness:

    ```bash
    tmux new-session -d -s train \
      'bash /workspace/toy-act/.opencode/commands/scripts/vast-train/runner.sh'
    ```

    Poll for up to 30 seconds until
    `/workspace/toy-act/.vast-train/state/run-status` reads `running`; require
    the `train` session to survive an additional five-second window. A missing
    tmux session is never a success signal; the persisted state files are
    authoritative.
16. Start the backup wrapper in a detached tmux session named `ckpt-bkp`. Refuse
    to start if the session already exists. Poll for up to ten minutes for
    `state/backup-running`, `state/backup-artifact-ready`, and
    `state/backup-last-succeeded`; fail immediately if `state/backup-failed`
    appears or `ckpt-bkp` exits. The wrapper discovers the single run directory,
    then synchronizes `checkpoints/act_v1/<run>` and `runs/act_v1/<run>` to S3
    every 120 seconds using `scripts/s3_backup.py`. The rolling `last.pt` is
    excluded from synchronization because it is rewritten in place while training
    runs; only the immutable periodic snapshots are uploaded. An existing
    `last.pt` object, if present, is left untouched rather than deleted.
17. Hand off to a detached local watcher and stop babysitting. Launch the
    watcher with `setsid`/`nohup` so it survives the interactive agent returning,
    and make the watcher own the step-7 EXIT trap and the local `VAST_API_KEY`.
    The watcher must:
    - poll every 30 seconds, retrying transient SSH failures, and treat a
      stopped `train` session without a terminal state as a failure;
    - treat `state/completed` as success and `state/failed` as a reported
      failure;
    - show concise progress from `.vast-train/logs/training.log` without
      flooding;
    - on success, wait for `state/backup-final-succeeded`, read the run name from
      `state/run-name`, then verify locally through the workload profile that S3
      contains every expected periodic snapshot for the current
      `CHECKPOINT_EVERY` and `EPOCHS` values;
    - always destroy the instance through the EXIT trap and verify it no longer
      appears in `vastai show instances --raw`;
    - write a report file recording the run name, S3 URI, elapsed time, selected
      offer price, pinned commit, TensorBoard URL, and final cleanup status.
18. Report the run name, S3 URI, selected offer price, pinned commit, TensorBoard
    URL, and the watcher log/report path to the user, then return without
    blocking on the training run. Do not keep polling training progress in the
    interactive session.

If setup fails, destroy the instance through the watcher trap (or directly when
no watcher was started yet) and report the failure. If training fails, the
watcher waits briefly for the backup wrapper's best-effort final sync, preserves
already uploaded checkpoints, reports the failure and S3 prefix, and still
destroys the instance.

Because `VAST_API_KEY` is deliberately never placed on the instance, the
instance cannot clean itself up. If the local machine sleeps, reboots, or the
watcher is hard-killed, cleanup cannot run and the instance will leak; in that
case check `vastai show instances --raw` and destroy the labeled instance
manually.
