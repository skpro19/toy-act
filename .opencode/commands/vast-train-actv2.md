---
description: Train ACT v2 (CVAE) on a temporary Vast.ai RTX 4090 with S3 checkpoints
agent: build
---

Run `scripts/train_v2.py` on a newly provisioned Vast.ai instance and store every
completed checkpoint in `s3://toy-act/checkpoints/act_v2/`. Provision the latest
commit of the `act-v2` branch by cloning GitHub on the instance; do not transfer the
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
| S3 destination | `s3://toy-act/checkpoints/act_v2/` |
| Git remote | `https://github.com/skpro19/toy-act.git` |
| Git branch | `act-v2` |
| Dataset | `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` |
| Dataset S3 source | `s3://toy-act/datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` |
| Remote project | `/workspace/toy-act` |
| Remote control dir | `/workspace/toy-act/.vast-train` |
| Remote TensorBoard | `127.0.0.1:6006` on the instance |
| Instance label | `toy-act-train-actv2-<timestamp>` (generated per invocation) |
| Local run state | `.vast-train-local/toy-act-<INSTANCE_ID>/` |

## Instance label

Construct the instance label once per invocation, before provisioning, by
appending a date-time stamp to the fixed prefix:

```bash
RUN_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_LABEL="toy-act-train-actv2-${RUN_TIMESTAMP}"
```

`RUN_TIMESTAMP` uses the same `YYYYMMDD-HHMMSS` form as `make_run_name`, so the
label stays recognizable while remaining unique across repeated or concurrent
invocations. Do not regenerate `RUN_TIMESTAMP` later in the workflow. Record
`INSTANCE_LABEL` in `setup.env` so the exact label stays recoverable for
cleanup.

## Config selection

Before provisioning, ask the user which training config to use with the
interactive question tool. List the up-to-5 most-recently-modified
`configs/*.toml` files (newest first) and present each path as an option,
defaulting to `configs/act_v2_bs250_l1.toml`. Record the selected path as
`CONFIG_PATH`, verify it
exists and is git-tracked (`git ls-files --error-unmatch "$CONFIG_PATH"`), and
read its `checkpoint_every` and `epochs` values into `CHECKPOINT_EVERY` and
`EPOCHS` for use in the later verification step. The selected config is part of
the git clone on the instance, so no extra transfer is needed.

## Local run state

Create one `.vast-train-local/toy-act-<INSTANCE_ID>/` directory per instance at
provisioning time. It is the local audit and recovery record for the run and is
gitignored, so it is never committed. It holds:

| File | Contents |
|---|---|
| `setup.env` | Key/value record: instance id, instance label, SSH host/port, pinned commit, offer and actual price, local workflow index, tmux session names, TensorBoard port, run name, S3 checkpoint URI, and watcher pid/script/log/report paths |
| `instance.json` | Raw vast.ai instance record captured at provisioning (contains a `jupyter_token`, so treat it as sensitive) |
| `known_hosts` | Pinned host keys used with `StrictHostKeyChecking=yes` |
| `watcher.sh` | The exact detached watcher script that was launched |
| `watcher.pid` | PID of the detached watcher, for liveness checks and recovery |
| `watcher.log` | Timestamped watcher events and training progress |
| `report.txt` | Final outcome report written by the watcher (run name, S3 URIs, elapsed time, price, pinned commit, TensorBoard URL, S3 verification, cleanup status) |

## Workflow

1. Require `vastai`, `aws`, `jq`, `ssh`, `ssh-keyscan`, `rsync`, `git`, `tmux`,
   `flock`, and `ss`. Verify:
   - `vastai show instances --raw` succeeds;
   - `aws sts get-caller-identity --profile toy-pickplace-backup` succeeds;
   - listing `s3://toy-act/checkpoints/act_v2/` succeeds;
   - `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` exists locally;
   - `uv lock --check` succeeds.
2. Pin the code to the latest `act-v2` commit and confirm it is anonymously
   clonable, because the instance has no GitHub credentials:
   - `git fetch origin act-v2` and set `GIT_COMMIT=$(git rev-parse origin/act-v2)`;
   - `git ls-remote https://github.com/skpro19/toy-act.git act-v2` must succeed
     without prompting for a username. If it prompts, stop and ask the user to
     make the repository public; never embed tokens, keys, or credentials;
   - if the local working tree is dirty or local `act-v2` differs from `origin/act-v2`,
     warn the user that the clone will not include those local changes.
3. Refuse to continue if an instance with the exact label `$INSTANCE_LABEL`
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
   Ryzen 7000/9000), then price, disk bandwidth, and reliability. Also reject
   offers whose `machine_id` appears in
   `.vast-train-local/failed-machines.tsv` with a failure timestamp from the
   preceding 24 hours. Each quarantine record is a tab-separated Unix
   timestamp, machine id, and fixed reason (`gpu-start-error`). Ignore older
   records. Automatically try up to the best three non-quarantined offers in
   order. Never weaken a filter without asking the user.
6. Create exactly one instance using the fixed image, disk, SSH direct mode,
   and `INSTANCE_LABEL`. Reconcile the instance by exact label after every
   create attempt; do not rely only on parsing create-command output.
7. Once an instance ID exists, the detached watcher described in step 18 owns
   cleanup for that exact instance. Before training reaches `run-status=running`,
   setup failures may destroy the instance. After the watcher has observed
   `run-status=running`, it may destroy the instance only after it has read a
   remote `state/completed` or `state/failed` marker. An EXIT trap, signal,
   local suspend, SSH failure, Vast API failure, or missing tmux session must
   never bypass this terminal-state gate. Do not install cleanup in the
   interactive agent shell: returning from the agent must not destroy the
   running instance.
8. Poll the exact instance record for up to ten minutes. Require
   `actual_status=running` before accepting `vastai ssh-url INSTANCE_ID` or
   attempting SSH. If `status_msg` reports a GPU error or says the instance is
   unable to start, append the selected offer's `machine_id` to
   `.vast-train-local/failed-machines.tsv` under the same
   `/tmp/toy-act-local-wrapper.lock` `flock`, destroy the failed instance and
   verify its removal, re-run the offer search, and continue with the next
   ranked non-quarantined machine. Quarantine by `machine_id`, not offer id,
   because Vast can immediately advertise the same machine under a new offer
   id. Use a command-specific temporary `known_hosts` file populated by
   `ssh-keyscan`; then use `StrictHostKeyChecking=yes` for all SSH, rsync, and
   port-forwarding.
9. Verify the provisioned hardware against the selected offer before cloning.
   Treat the rental as provisional and collect:

   ```bash
   ssh -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes \
     -p $PORT root@$HOST '
     set -e
     lscpu
     lscpu -e=CPU,CORE,SOCKET,ONLINE
     nproc
     grep "^Cpus_allowed_list:" /proc/self/status
     grep "^MemTotal:" /proc/meminfo
     test ! -r /sys/fs/cgroup/memory.max || cat /sys/fs/cgroup/memory.max
     test ! -r /sys/fs/cgroup/cpu.max || cat /sys/fs/cgroup/cpu.max
     test ! -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
     test ! -r /sys/fs/cgroup/cpu/cpu.cfs_period_us || cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
     df -hT /workspace /
     nvidia-smi --query-gpu=name,memory.total,power.limit,pcie.link.gen.max,pcie.link.width.max,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv
   '
   ```

   Compare the results with the selected offer and require:

   | Check | Requirement |
   |---|---|
   | Physical cores | At least 8 allowed unique `(CORE, SOCKET)` pairs |
   | CPU generation | Zen 3 or newer; reject EPYC 7001/7002 |
   | CPU quota | At least 90% of advertised effective vCPUs |
   | RAM | At least 32 GB allocated and consistent with the offer |
   | GPU | Exactly one RTX 4090 with approximately 24 GB VRAM |
   | Disk | At least 100 GB available at `/workspace` |
   | Throttling | Thermal and power-brake slowdown inactive |

   Count physical cores only from online CPU IDs in `Cpus_allowed_list`, then
   count unique `(CORE, SOCKET)` pairs. Interpret finite cgroup CPU and memory
   limits as the allocation gates; use visible memory only when the cgroup limit
   is unlimited. Record the CPU model, logical CPUs, RAM, GPU identity, and disk
   and list every mismatch with the offer. The SSH checks measure local compute
   only; disk and network values still come from the offer. If a check fails,
   destroy the instance directly (the watcher has not been launched yet) and try
   the next-ranked offer; if none remain, stop and report.
10. Clone the pinned commit on the instance and verify it:

   ```bash
   git clone --branch act-v2 --single-branch \
     https://github.com/skpro19/toy-act.git /workspace/toy-act
   test "$(git -C /workspace/toy-act rev-parse HEAD)" = "$GIT_COMMIT"
   ```

   The clone already contains `scripts/`, the project files, and the
   `.opencode/commands/scripts/vast-train/` helpers (`runner.sh`,
   `ckpt-bkp-wrapper.sh`, `local-wrapper-lease.sh`).
11. On the instance, install `uv`, `awscli`, and `tmux`, then run
    `uv sync --frozen --only-group train` and verify `torch.cuda.is_available()`,
    printing the GPU name.
12. Resolve `toy-pickplace-backup` credentials locally with
    `aws configure export-credentials`. Write them to a mode-600 temporary env
    file without printing them, append `AWS_REGION` and `S3_BUCKET`, transfer it
    as `/workspace/toy-act/.vast-train/s3-env.env` (creating
    `/workspace/toy-act/.vast-train` first), chmod it 600 remotely, and delete
    the local temporary file. Never transfer `VAST_API_KEY` or print AWS
    credentials.
13. Ensure the dataset is in the bucket, then fetch it on the instance:
    - upload `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` to
      `s3://toy-act/datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5`
      with the workload profile, skipping the upload when the object already
      exists with the same size;
    - on the instance, download it from that key to the same project-relative
      path using the transferred credentials and installed `awscli`;
    - verify the remote SHA-256 matches the local file before launching.
14. Create the remote control directories
    `/workspace/toy-act/.vast-train/{logs,state}` and start TensorBoard in a
    detached tmux session named `tensorboard`:

    ```bash
    tmux new-session -d -s tensorboard \
      'cd /workspace/toy-act && exec /root/.local/bin/uv run --frozen \
       --only-group train python -m tensorboard.main \
       --logdir runs/act_v2 --host 127.0.0.1 --port 6006'
    ```

    Poll its endpoint from the instance for up to 12 attempts at five-second
    intervals and stop if the session exits or
    `http://127.0.0.1:6006/` never becomes reachable.
15. Claim the lowest unused local workflow index with
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
16. Start the durable runner in a detached tmux session named `train` and wait
    until it publishes readiness:

    ```bash
    tmux new-session -d -s train \
      -e TRAIN_MODULE=scripts.train_v2 \
      -e TRAIN_CONFIG="$CONFIG_PATH" \
      'bash /workspace/toy-act/.opencode/commands/scripts/vast-train/runner.sh'
    ```

    Poll for up to 30 seconds until
    `/workspace/toy-act/.vast-train/state/run-status` reads `running`; require
    the `train` session to survive an additional five-second window. A missing
    tmux session is never a success signal; the persisted state files are
    authoritative.
17. Start the backup wrapper in a detached tmux session named `ckpt-bkp`. Refuse
    to start if the session already exists. Launch it with
    `CHECKPOINT_ROOT=checkpoints/act_v2` and `RUNS_ROOT=runs/act_v2` in the tmux
    session environment so it discovers the v2 directories. Poll for up to ten
    minutes for
    `state/backup-running`, `state/backup-artifact-ready`, and
    `state/backup-last-succeeded`; fail immediately if `state/backup-failed`
    appears or `ckpt-bkp` exits. The wrapper discovers the single run directory,
    then synchronizes `checkpoints/act_v2/<run>` and `runs/act_v2/<run>` to S3
    every 120 seconds using `scripts/s3_backup.py`. Training only writes immutable
    periodic snapshots (`epoch_*.pt`), so every checkpoint is safe to upload.
18. Hand off to a detached local watcher and stop babysitting. Launch the
    watcher with `setsid`/`nohup` so it survives the interactive agent returning,
    and make the watcher own step-7 cleanup and the local `VAST_API_KEY`.
    The watcher must:
    - record `RUN_STARTED=yes` only after it reads remote `state/run-status` as
      `running`, and initialize `TERMINAL_CONFIRMED=no`;
    - poll every 30 seconds; if SSH fails, log the failure, sleep, and retry
      indefinitely without changing the run outcome or entering cleanup. Use a
      fixed interval, so no failure counter is needed. A suspended local machine
      freezes the watcher; after wake-up it must continue the same retry loop and
      resume normal monitoring when SSH recovers;
    - treat `vastai show instances` command failures, malformed output, and an
      unavailable API as unknown state, never as proof that the instance or
      training disappeared. Vast instance queries are diagnostic only after
      `RUN_STARTED=yes` and cannot authorize cleanup;
    - set `TERMINAL_CONFIRMED=yes` only after a successful SSH probe directly
      reads remote `state/completed` or `state/failed`. Treat `completed` as
      success and `failed` as a reported training failure;
    - if the `train` tmux session is missing without either terminal marker,
      log the inconsistency and continue polling for the runner to publish a
      terminal marker. Do not infer training failure from the missing session;
    - show concise progress from `.vast-train/logs/training.log` without
      flooding;
    - on success, wait for `state/backup-final-succeeded`, read the run name from
      `state/run-name`, then verify locally through the workload profile that S3
      contains every expected periodic snapshot for the `CHECKPOINT_EVERY` and
      `EPOCHS` values read from the selected config, using `scripts/s3_backup.py
      has-files` with `S3_CHECKPOINT_BASE=checkpoints/act_v2` and
      `--components checkpoints`; also verify the run's recorded `config.json`
      reached S3 using `has-files` with `S3_RUNS_BASE=runs/act_v2` and
      `--components runs`;
    - before every `vastai destroy`, enforce the cleanup gate again: setup may
      destroy before `RUN_STARTED=yes`; after that point require
      `TERMINAL_CONFIRMED=yes`. If the gate is closed, log that cleanup was
      refused and leave the instance untouched. Apply this gate inside the EXIT
      trap too, so normal `kill`, HUP, shell errors, and unexpected exits cannot
      destroy an unconfirmed running job. Connectivity errors must remain in the
      monitoring loop rather than reaching the EXIT trap;
    - after an authorized destroy, verify the exact instance no longer appears
      in `vastai show instances --raw`;
    - write its PID to `.vast-train-local/toy-act-<INSTANCE_ID>/watcher.pid` and
      write a report to `.vast-train-local/toy-act-<INSTANCE_ID>/report.txt`
      recording the run name, S3 URI, elapsed time, selected offer price, pinned
      commit, TensorBoard URL, and final cleanup status.
19. Report the run name, S3 URI, selected offer price, pinned commit, TensorBoard
    URL, and the local run-state directory
    `.vast-train-local/toy-act-<INSTANCE_ID>/` (log at `watcher.log`, report at
    `report.txt`) to the user, then return without blocking on the training run.
    Do not keep polling training progress in the interactive session.

If setup fails before remote `run-status=running`, destroy the instance through
the watcher cleanup (or directly when no watcher was started yet) and report the
failure. If remote `state/failed` appears after training starts, the watcher sets
`TERMINAL_CONFIRMED=yes`, waits briefly for the backup wrapper's best-effort
final sync, preserves already uploaded checkpoints, reports the failure and S3
prefix, and destroys the instance. Connectivity failures without a remote
terminal marker remain in the monitoring loop and never authorize destruction.

Because `VAST_API_KEY` is deliberately never placed on the instance, the
instance cannot clean itself up. Local suspend pauses the watcher while remote
training and backup continue; after wake-up, the watcher resumes polling and
must recover through the SSH retry path. A reboot or hard-killed watcher cannot
resume automatically and can leak the instance; in that case use `setup.env` to
restart monitoring, or check `vastai show instances --raw` and destroy the
labeled instance manually after verifying training state. The instance id and
exact `INSTANCE_LABEL` are recoverable from
`.vast-train-local/toy-act-<INSTANCE_ID>/setup.env` and `instance.json`.
