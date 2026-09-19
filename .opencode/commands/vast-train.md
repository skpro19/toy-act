---
description: Train ACT on a temporary Vast.ai RTX 4090 with S3 checkpoints
agent: build
---

Run `scripts/train.py` on a newly provisioned Vast.ai instance and store every
completed checkpoint in `s3://toy-act/checkpoints/act_v1/`. Use the current
working tree, including uncommitted files. Do not clone the Git repository.

## Fixed configuration

| Setting | Value |
|---|---|
| GPU | One full RTX 4090 |
| Maximum price | $0.60/hour |
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| Disk | 100 GB |
| AWS profile | `toy-pickplace-backup` |
| S3 region | `ap-south-1` |
| S3 destination | `s3://toy-act/checkpoints/act_v1/` |
| Remote project | `/workspace/toy-act` |
| Instance label | `toy-act-train` |

## Workflow

1. Require `vastai`, `aws`, `jq`, `ssh`, `ssh-keyscan`, and `rsync`. Verify:
   - `vastai show instances --raw` succeeds;
   - `aws sts get-caller-identity --profile toy-pickplace-backup` succeeds;
   - listing `s3://toy-act/checkpoints/act_v1/` succeeds;
   - `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` exists;
   - `uv lock --check` succeeds.
2. Refuse to continue if an instance with the exact label `toy-act-train`
   already exists. Never destroy or reuse an unrelated instance.
3. Search offers with the following hard filters:

   ```bash
   vastai search offers \
     'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 gpu_ram>=24 compute_cap>=890 cpu_cores_effective>=8 cpu_ram>=32 inet_down>=200 inet_up>=100 reliability>=0.98 rentable=true verification=verified gpu_display_active=false' \
     --order dph_total+ --raw
   ```

4. Keep offers at or below `$0.60/hour`, reject EPYC 7001/7002, and rank by
   CPU family (EPYC 9005, EPYC 9004, Threadripper 7000, EPYC 7003, modern
   Ryzen 7000/9000), then price, disk bandwidth, and reliability. Automatically
   try up to the best three offers in order. Never weaken a filter without
   asking the user.
5. Create exactly one instance using the fixed image, disk, SSH direct mode,
   and label. Reconcile the instance by exact label after every create attempt;
   do not rely only on parsing create-command output.
6. Once an instance ID exists, install an EXIT trap that destroys that exact
   instance and verifies it no longer appears in `vastai show instances --raw`.
   The trap must run on both success and failure.
7. Wait up to ten minutes for `vastai ssh-url INSTANCE_ID` and successful SSH.
   Use a command-specific temporary `known_hosts` file populated by
   `ssh-keyscan`; then use `StrictHostKeyChecking=yes` for all SSH and rsync.
8. Create `/workspace/toy-act`, then transfer only:
   - `scripts/`;
   - `pyproject.toml`, `uv.lock`, and `.python-version`;
   - the fixed HDF5 dataset, preserving its project-relative path;
   - `.opencode/commands/scripts/vast-train/remote-run.sh`.
9. Resolve `toy-pickplace-backup` credentials locally with
   `aws configure export-credentials`. Write them to a mode-600 temporary env
   file without printing them, transfer it as `/workspace/toy-act/.aws.env`,
   chmod it 600 remotely, and delete the local temporary file. Never transfer
   `VAST_API_KEY` or print AWS credentials.
10. On the instance:
    - install `uv`, `awscli`, `tmux`, and `inotify-tools`;
    - run `uv sync --frozen --only-group train`;
    - verify `torch.cuda.is_available()` and print the GPU name;
    - start `remote-run.sh` in a detached tmux session named `train`.
11. Monitor `training.status` every 30 seconds. Show concise progress from
    `training.log` without flooding the user. Temporary SSH failures should be
    retried; a stopped tmux session without a terminal status is a failure.
12. On success, discover the single new run directory under
    `checkpoints/act_v1`, then verify locally through the workload profile that
    S3 contains `last.pt` and every expected periodic snapshot for the current
    `CHECKPOINT_EVERY` and `EPOCHS` values.
13. Always destroy the instance through the EXIT trap. Report the run name,
    S3 URI, elapsed time, selected offer price, and final cleanup status.

If setup or training fails, preserve already uploaded checkpoints, report the
failure and S3 prefix, and still destroy the instance.
