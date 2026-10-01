---
description: Train ACT v2 (CVAE) on a temporary Vast.ai RTX 4090 with S3 checkpoints
---

Run `scripts/train_v2.py` with a v4 config on a newly provisioned Vast.ai instance
and store every completed checkpoint in `s3://toy-act/checkpoints/act_v2/`.
Provision the latest
commit of the `act-v2` branch by cloning GitHub on the instance; do not transfer the
local working tree. Fetch the dataset from S3 at its project-relative path
instead of copying it from the local machine.

The instance follows the `flywheel-4090` pattern: a durable runner owns the
training process, a separate backup wrapper synchronizes run-scoped checkpoints
and TensorBoard runs to S3, and TensorBoard is reachable from the local machine
through an SSH port-forward. The pi session is only responsible for
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
| Dataset | resolved from the selected config's `dataset` key into `DATASET_PATH` |
| Dataset S3 source | `s3://toy-act/` + `DATASET_PATH` |
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

Before provisioning, ask the user which v4 training config to use with the
`question` tool. List the up-to-5 most-recently-modified v4
`configs/*.toml` files (newest first) and present each path as an option,
defaulting to `configs/act_v2_instance_bs8_beta0p01_wu80_l1_img_agentview_eyeinhand.toml`.
Record the selected path as `CONFIG_PATH`, verify it exists and is git-tracked
(`git ls-files --error-unmatch "$CONFIG_PATH"`), and load it with
`scripts.train_v2.load_config` to validate its v4 schema. Read its
`checkpoint_every`, `steps`, ordered `image_keys`, and the required `dataset`
into `CHECKPOINT_EVERY`, `STEPS`, `IMAGE_KEYS`, and `DATASET_PATH`. Derive
`DATASET_S3_URI="s3://toy-act/$DATASET_PATH"` for the step-13 upload and
download. Validate that the
selected config's cameras are supported by `scripts/rollout.py`; confirm the
downloaded dataset contains every requested `image_key` before launching.
The selected config is part of the git clone on the instance, so no extra
transfer is needed.

## Local run state

Create one `.vast-train-local/toy-act-<INSTANCE_ID>/` directory per instance at
provisioning time. It is the local audit and recovery record for the run and is
gitignored, so it is never committed. It holds:

| File | Contents |
|---|---|
| `setup.env` | Key/value record: instance id, instance label, SSH host/port, pinned commit, offer and actual price, local workflow index, tmux session names, TensorBoard port, run name, dataset path, S3 checkpoint URI, and watcher pid/script/log/report paths |
| `instance.json` | Raw vast.ai instance record captured at provisioning (contains a `jupyter_token`, so treat it as sensitive) |
| `known_hosts` | Pinned host keys used with `StrictHostKeyChecking=yes` |
| `watcher.sh` | Copy of the committed `local-watcher.sh` template that was launched |
| `watcher.pid` | PID of the detached watcher, for liveness checks and recovery |
| `watcher.log` | Timestamped watcher events and training progress |
| `report.txt` | Final outcome report written by the watcher (run name, S3 URIs, elapsed time, price, pinned commit, TensorBoard URL, S3 verification, cleanup status) |

## Workflow

1. Require `vastai`, `aws`, `jq`, `ssh`, `ssh-keyscan`, `git`, `tmux`, `flock`,
   `ss`, and `curl`. Verify:
   - `vastai show instances --raw` succeeds;
   - `aws sts get-caller-identity --profile toy-pickplace-backup` succeeds;
   - listing `s3://toy-act/checkpoints/act_v2/` succeeds;
   - the selected dataset is present locally or can be read from its S3 source;
   - `uv lock --check` succeeds.
2. Pin the code to the latest `act-v2` commit and confirm it is anonymously
   clonable, because the instance has no GitHub credentials:
   - `git fetch origin act-v2` and set `GIT_COMMIT=$(git rev-parse origin/act-v2)`;
   - `GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/skpro19/toy-act.git
     act-v2` must succeed without prompting for a username. If it fails or
     prompts, stop and ask the user to make the repository public; never embed
     tokens, keys, or credentials;
   - if the local working tree is dirty or local `act-v2` differs from `origin/act-v2`,
     warn the user that the clone will not include those local changes.
3. Refuse to continue if an instance with the exact label `$INSTANCE_LABEL`
   already exists. Never destroy or reuse an unrelated instance.
4. Search offers with the following hard filters:

   ```bash
   vastai search offers \
     'gpu_name=RTX_4090 num_gpus=1 gpu_ram>=24 gpu_max_power>=400 compute_cap>=890 cpu_cores_effective>=24 cpu_ram>=64 pci_gen>=4 pcie_bw>=20 inet_down>=500 inet_up>=200 reliability>=0.99 rentable=true verification=verified gpu_display_active=false' \
     --order dph_total+ --raw
   ```

   Do not add `gpu_frac=1`. Vast.ai defines `gpu_frac` as GPUs in the offer
   divided by GPUs in the host, so it rejects a full single 4090 on a multi-GPU
   host. `num_gpus=1` together with `gpu_ram>=24` already guarantees one full
   GPU. The `cpu_cores_effective>=24`, `cpu_ram>=64`, `gpu_max_power>=400`,
   `pci_gen>=4`, and `pcie_bw>=20` filters mirror the hardware gate enforced in
   step 9 so that instances which pass a bare capacity check but throttle under
   load are rejected up front.

5. Keep offers at or below `$0.80/hour`, reject EPYC 7001/7002, and rank by
   CPU family (EPYC 9005, EPYC 9004, Threadripper 7000, EPYC 7003, modern
   Ryzen 7000/9000), then price, disk bandwidth, and reliability. Also reject
   offers whose `machine_id` appears in
   `.vast-train-local/failed-machines.tsv` with a failure timestamp from the
   preceding 24 hours. Each quarantine record is a tab-separated Unix
   timestamp, machine id, and fixed reason (`gpu-start-error`). Ignore older
   records. Automatically try up to the best three non-quarantined offers in
   order. Never weaken a filter without asking the user.

   Vast may charge more than the offer's advertised `dph_total` (for example
   `$0.5356` offered vs `$0.5796` actual). Apply the `$0.80/hour` cap to the
   offer price and record both the offer and the actual instance `dph_total` in
   `setup.env`.
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
   pi session shell: returning from the session must not destroy the
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
   `ssh-keyscan`; then use `StrictHostKeyChecking=yes` for all SSH, file
   transfers, and port-forwarding.
9. Verify the provisioned hardware against the selected offer before cloning.
   Treat the rental as provisional and collect:

   ```bash
   ssh -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes \
     -p $PORT root@$HOST '
     set -e
     lscpu
     lscpu -e=CPU,CORE,SOCKET,ONLINE
     lscpu -p=CORE,SOCKET | grep -v "^#" | sort -u | wc -l
     nproc
     grep "^Cpus_allowed_list:" /proc/self/status
     free -b
     grep "^MemTotal:" /proc/meminfo
     test ! -r /sys/fs/cgroup/memory.max || cat /sys/fs/cgroup/memory.max
     test ! -r /sys/fs/cgroup/memory/memory.limit_in_bytes || cat /sys/fs/cgroup/memory/memory.limit_in_bytes
     test ! -r /sys/fs/cgroup/cpu.max || cat /sys/fs/cgroup/cpu.max
     test ! -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
     test ! -r /sys/fs/cgroup/cpu/cpu.cfs_period_us || cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
     df -hT /workspace /
     nvidia-smi --query-gpu=name,memory.total,power.limit,power.default_limit,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv
   '
   ```

   Compare the results with the selected offer and require:

   | Check | Requirement |
   |---|---|
   | Physical cores | At least 24 allowed unique `(CORE, SOCKET)` pairs |
   | CPU generation | Zen 3 or newer; reject EPYC 7001/7002 |
   | CPU quota | At least 90% of advertised effective vCPUs |
   | RAM | At least 64 GB allocated and consistent with the offer |
   | GPU | Exactly one RTX 4090 with approximately 24 GB VRAM |
   | GPU power | At least 400 W |
   | PCIe | Gen4 x16 capability and offer `pcie_bw >= 20` GB/s |
   | Disk | At least 100 GB available at `/workspace` |
   | Throttling | Thermal and power-brake slowdown inactive |

   Count physical cores only from online CPU IDs in `Cpus_allowed_list`, then
   count unique `(CORE, SOCKET)` pairs. Interpret finite cgroup CPU and memory
   limits as the allocation gates; use visible memory only when the cgroup limit
   is unlimited. An idle PCIe link may downshift, so maximum Gen4 x16 capability
   plus the passing offer measurement is sufficient unless other evidence
   indicates restriction. Record the CPU model, logical CPUs, RAM, GPU identity,
   GPU power, PCIe capability, and disk, and list every mismatch with the offer.
   The SSH checks measure local compute only; disk and network values still come
   from the offer. If a check fails, destroy the instance directly (the watcher
   has not been launched yet) and try the next-ranked offer; if none remain, stop
   and report.

### Network quality acceptance (runs after step 9, before cloning)

Hardware checks measure local compute. A separate gate measures actual S3
operation latency and upload speed to the bucket where checkpoints and runs are
written (`s3://toy-act/`). This catches instances whose advertised `inet_up` is
misleading due to geographic distance, ISP throttling, or host oversubscription.

Generate presigned PUT/DELETE URLs locally through the `toy-pickplace-backup`
profile, then transfer and run
`.pi/prompts/scripts/vast-train/network-gate.sh` on the instance:

```bash
TEMP_NETGATE_KEY="runs/act_v2/.netgate/$(date +%s%N)"
S3_PRESIGNED_PUT=$(AWS_PROFILE=toy-pickplace-backup uv run python -c "
import boto3
s3 = boto3.Session(profile_name='toy-pickplace-backup', region_name='ap-south-1').client('s3')
print(s3.generate_presigned_url('put_object',
    Params={'Bucket': 'toy-act', 'Key': '${TEMP_NETGATE_KEY}'}, ExpiresIn=900))
")
S3_PRESIGNED_DELETE=$(AWS_PROFILE=toy-pickplace-backup uv run python -c "
import boto3
s3 = boto3.Session(profile_name='toy-pickplace-backup', region_name='ap-south-1').client('s3')
print(s3.generate_presigned_url('delete_object',
    Params={'Bucket': 'toy-act', 'Key': '${TEMP_NETGATE_KEY}'}, ExpiresIn=900))
")

B64_NG=$(base64 -w0 .pi/prompts/scripts/vast-train/network-gate.sh)

NETGATE_OUTPUT=$(S3_PRESIGNED_PUT="$S3_PRESIGNED_PUT" \
  S3_PRESIGNED_DELETE="$S3_PRESIGNED_DELETE" \
  ssh -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes \
    -o BatchMode=yes -p $PORT root@$HOST \
    "printf '%s' '${B64_NG}' | base64 -d > /tmp/network-gate.sh && \
     chmod +x /tmp/network-gate.sh && \
     S3_PRESIGNED_PUT='${S3_PRESIGNED_PUT}' \
     S3_PRESIGNED_DELETE='${S3_PRESIGNED_DELETE}' \
     bash /tmp/network-gate.sh" 2>/dev/null)

echo "$NETGATE_OUTPUT"
```

Require:

| Check | Requirement |
|---|---|
| S3 PUT operation latency | Median of 7 successful samples ≤ 5000 ms against the presigned bucket key |
| S3 upload | Median of 3 successful 4 MiB uploads ≥ 1000 KB/s to the bucket |

The test key lives under `runs/act_v2/.netgate/`, inside the workload profile's
allowed prefix scope. If the output does not start with `PASSED`, remove
`/tmp/network-gate.sh` and `/tmp/.netgate-test.bin` on the instance, destroy the
provisional instance, verify removal, and return to step 4 to try the
next-ranked offer; if none remain, stop and report. Do not weaken these
thresholds without asking the user.

10. Clone the pinned commit on the instance and verify it:

   ```bash
   git clone --branch act-v2 --single-branch \
     https://github.com/skpro19/toy-act.git /workspace/toy-act
   test "$(git -C /workspace/toy-act rev-parse HEAD)" = "$GIT_COMMIT"
   ```

   The clone already contains `scripts/`, the project files, and the
   `.pi/prompts/scripts/vast-train/` helpers (`runner.sh`,
   `ckpt-bkp-wrapper.sh`, `local-wrapper-lease.sh`, `network-gate.sh`,
   `local-watcher.sh`).
11. On the instance, install `uv`, `awscli`, `tmux`, `build-essential`, and the
    headless GL libraries the checkpoint-evaluation renderer needs.

    `scripts/train_v2.py` runs a robosuite rollout at every checkpoint, so it
    imports `robosuite`, `robomimic`, and `mujoco`. The `train` dependency group
    therefore contains the full training stack, and its simulator packages are
    sourced from pinned upstream git revisions in `[tool.uv.sources]`
    (`robomimic` @ `d309eae`, `robosuite` @ `a071383`, v1.5.1, per the official
    robomimic install guide). `uv sync` fetches and builds them itself; no
    `third_party/` clone or transfer is needed. The compiler is required because
    robosuite pulls in `pynput` -> `evdev`, which builds from source and
    otherwise fails with `No such file or directory: 'cc'`.

    Install the tooling and sync the `train` group, which is self-sufficient:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh          # -> /root/.local/bin/uv
    /opt/conda/bin/pip install --quiet awscli
    apt-get update -qq && apt-get install -y -qq \
      tmux build-essential libgl1 libglib2.0-0 libegl1 libgles2 libglfw3
    cd /workspace/toy-act
    /root/.local/bin/uv sync --frozen --only-group train
    ```

    `runner.sh` and the TensorBoard session both run `uv run --frozen
    --only-group train`, so they reuse this same self-sufficient environment.

    Verify `torch.cuda.is_available()` (printing the GPU name), that the rollout
    imports load, and that off-screen EGL rendering works, because a broken
    renderer would otherwise surface only at the first checkpoint:

    ```bash
    cd /workspace/toy-act && MUJOCO_GL=egl /root/.local/bin/uv run --frozen \
      --only-group train python - <<'PY'
    import torch
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0))
    import robosuite, robomimic, mujoco
    print("robosuite", robosuite.__version__, "robomimic", robomimic.__version__)
    xml = '<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=84, width=84)
    renderer.update_scene(data)
    frame = renderer.render()
    renderer.close()
    print("EGL render", frame.shape, frame.dtype)
    PY
    ```

    Require the GPU name, successful `robosuite`/`robomimic` imports, and
    `EGL render (84, 84, 3) uint8`. Stop and report if it fails rather than
    launching.
12. Resolve `toy-pickplace-backup` credentials locally with
    `aws configure export-credentials`. Write them to a mode-600 temporary env
    file without printing them, append `AWS_REGION` and `S3_BUCKET`, transfer it
    as `/workspace/toy-act/.vast-train/s3-env.env` (creating
    `/workspace/toy-act/.vast-train` first), chmod it 600 remotely, and delete
    the local temporary file. Never transfer `VAST_API_KEY` or print AWS
    credentials.
13. Ensure the selected dataset (`DATASET_PATH`, S3 `DATASET_S3_URI`) is in
    the bucket, then fetch it on the instance:
    - if the dataset exists locally, upload it to `DATASET_S3_URI` with the
      workload profile, skipping upload when the object already exists with the
      same size. The multi-gigabyte multipart upload can hit transient endpoint
      errors, so set `AWS_MAX_ATTEMPTS=10 AWS_RETRY_MODE=adaptive` and retry up
      to three times; if it is not local, require that the S3 object already
      exists;
    - the workload profile can `PutObject` under `datasets/` but is **not**
      authorized to `DeleteObject` there, so never attempt to remove dataset
      objects with it;
    - on the instance, download it from `DATASET_S3_URI` to `DATASET_PATH` using
      the transferred credentials and installed `awscli`;
    - verify the remote SHA-256 matches the local file when present; otherwise
      verify its byte size against S3 `head-object` and open it with `h5py` to
      check the requested camera keys before launching.
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
    `.pi/prompts/scripts/vast-train/local-wrapper-lease.sh allocate
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
      'bash /workspace/toy-act/.pi/prompts/scripts/vast-train/runner.sh'
    ```

    The dataset path is read from `CONFIG_PATH` on the instance, so `TRAIN_DATASET`
    is not passed here; `runner.sh` still forwards it when set to override the
    config. Poll for up to 30 seconds until
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
    every 120 seconds using `scripts/s3_backup.py`. Training writes immutable
    periodic and final snapshots (`step_*.pt`), so every checkpoint is safe to
    upload.
    The same run directory holds TensorBoard training, rollout success, and
    training/rollout throughput scalars.
18. Hand off to a detached local watcher and stop babysitting. Copy the
    committed `.pi/prompts/scripts/vast-train/local-watcher.sh` to
    `$RUN_DIR/watcher.sh` and launch it with `setsid`/`nohup` so it survives the
    pi session returning:

    ```bash
    cp .pi/prompts/scripts/vast-train/local-watcher.sh "$RUN_DIR/watcher.sh"
    chmod +x "$RUN_DIR/watcher.sh"
    setsid nohup bash "$RUN_DIR/watcher.sh" "$RUN_DIR" \
      >"$RUN_DIR/watcher.out" 2>&1 </dev/null &
    ```

    The template sources `$RUN_DIR/setup.env` for the run-specific values and
    owns step-7 cleanup plus the local `VAST_API_KEY`. Set `WATCHER_DRY_RUN=yes`
    to rehearse the gate without destroying. It writes its PID to `watcher.pid`.
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
      contains every expected `step_*.pt` snapshot (multiples of
      `CHECKPOINT_EVERY` up to `STEPS`, plus `STEPS` when it is not a multiple)
      using `scripts/s3_backup.py has-files` with
      `S3_CHECKPOINT_BASE=checkpoints/act_v2` and `--components checkpoints`;
      also verify the run's recorded `config.json` reached S3 using `has-files`
      with `S3_RUNS_BASE=runs/act_v2` and `--components runs`. Verify that the
      uploaded TensorBoard event files contain the final rollout success and
      throughput scalars;
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
    Do not keep polling training progress in the pi session.

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

## Operational notes for the pi session

Keep provisioning commands bounded so the pi session stays responsive. Wrap
remote `ssh`/`scp` calls and polling loops in `timeout`, and never leave an
unbounded foreground loop in the session; the detached watcher owns the long
run, not the pi session.

Do **not** wrap `local-wrapper-lease.sh allocate` in an outer `flock`. The
script locks `/tmp/toy-act-local-wrapper.lock` internally, so an outer lock on
the same file deadlocks the inner `flock` forever. Call it directly:

```bash
LEASE=$(timeout 30 .pi/prompts/scripts/vast-train/local-wrapper-lease.sh \
  allocate "toy-act-$INSTANCE_ID")
```

Parse `vastai show instances --raw` with `jq`, not by splitting a
pipe-joined string on whitespace. Several fields (notably `status_msg`, e.g.
`success, status_msg=running`) contain spaces, so a `read` on a `|`-joined
record silently misaligns fields and the poll never sees `running`. Emit one
field at a time instead:

```bash
instance_field() {
  local id="$1" field="$2"
  vastai show instances --raw \
    | jq -r --argjson id "$id" --arg field "$field" \
        '.[] | select(.id==$id) | .[$field] // "null"'
}

for _ in $(seq 1 60); do
  [ "$(instance_field "$INSTANCE_ID" actual_status)" = running ] && break
  sleep 10
done
```
