---
description: Train ACT on a temporary Vast.ai RTX 4090 with S3 checkpoints
agent: build
---

Run `scripts/train.py` on a newly provisioned Vast.ai instance and store every
completed checkpoint in `s3://toy-act/checkpoints/act_v1/`. Provision the latest
commit of the `dev` branch by cloning GitHub on the instance; do not transfer the
local working tree. Fetch the dataset from S3 at its project-relative path
instead of copying it from the local machine.

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
| Instance label | `toy-act-train` |

## Workflow

1. Require `vastai`, `aws`, `jq`, `ssh`, `ssh-keyscan`, `rsync`, and `git`.
   Verify:
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
7. Once an instance ID exists, install an EXIT trap that destroys that exact
   instance and verifies it no longer appears in `vastai show instances --raw`.
   The trap must run on both success and failure.
8. Wait up to ten minutes for `vastai ssh-url INSTANCE_ID` and successful SSH.
   Use a command-specific temporary `known_hosts` file populated by
   `ssh-keyscan`; then use `StrictHostKeyChecking=yes` for all SSH and rsync.
9. Clone the pinned commit on the instance and verify it:

   ```bash
   git clone --branch dev --single-branch \
     https://github.com/skpro19/toy-act.git /workspace/toy-act
   test "$(git -C /workspace/toy-act rev-parse HEAD)" = "$GIT_COMMIT"
   ```

   The clone already contains `scripts/`, `pyproject.toml`, `uv.lock`,
   `.python-version`, and `.opencode/commands/scripts/vast-train/remote-run.sh`.
10. Resolve `toy-pickplace-backup` credentials locally with
    `aws configure export-credentials`. Write them to a mode-600 temporary env
    file without printing them, transfer it as `/workspace/toy-act/.aws.env`,
    chmod it 600 remotely, and delete the local temporary file. Never transfer
    `VAST_API_KEY` or print AWS credentials.
11. Ensure the dataset is in the bucket, then fetch it on the instance:
    - upload `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` to
      `s3://toy-act/datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5`
      with the workload profile, skipping the upload when the object already
      exists with the same size;
    - on the instance, download it from that key to the same project-relative
      path using the transferred credentials;
    - verify the remote SHA-256 matches the local file before launching.
12. On the instance:
    - install `uv`, `awscli`, `tmux`, and `inotify-tools`;
    - run `uv sync --frozen --only-group train`;
    - verify `torch.cuda.is_available()` and print the GPU name;
    - start `remote-run.sh` in a detached tmux session named `train`.
13. Monitor `training.status` every 30 seconds. Show concise progress from
    `training.log` without flooding the user. Temporary SSH failures should be
    retried; a stopped tmux session without a terminal status is a failure.
14. On success, discover the single new run directory under
    `checkpoints/act_v1`, then verify locally through the workload profile that
    S3 contains `last.pt` and every expected periodic snapshot for the current
    `CHECKPOINT_EVERY` and `EPOCHS` values.
15. Always destroy the instance through the EXIT trap. Report the run name,
    S3 URI, elapsed time, selected offer price, pinned commit, and final cleanup
    status.

If setup or training fails, preserve already uploaded checkpoints, report the
failure and S3 prefix, and still destroy the instance.
