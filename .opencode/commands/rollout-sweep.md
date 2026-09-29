---
description: Run off-screen rollouts over every S3 checkpoint of one or more ACT v2 runs on a temporary RTX 4090
agent: build
---

Provision a temporary Vast.ai RTX 4090 and run `scripts/rollout_sweep_s3.py` over
every `epoch_*.pt` checkpoint of the named training runs under
`s3://toy-act/checkpoints/act_v2/`. The instance clones the latest `act-v2`
commit, streams each checkpoint from S3, evaluates it off-screen, and
incrementally uploads each run's artifacts to
`s3://toy-act/rollouts/act_v2/<run-name>/` (`results.json`, `results.npz`,
`success_rate.png`, and the run's `config.json`).

The instance follows the `flywheel-4090` pattern from
`.opencode/commands/vast-train-actv2.md`: a durable runner owns the sweep and a
detached local watcher owns the run to completion, including cleanup. The
interactive agent only provisions and hands off; do not keep polling progress
in the interactive session.

## Arguments

`$ARGUMENTS` is one or more bucket run names followed by an optional version:

```
<run-name> [<run-name> ...] [--version act_v2]
```

- `RUN_NAMES` — every non-flag token; each must be an existing run directory
  name under `s3://toy-act/checkpoints/act_v2/`.
- `VERSION` — the `--version` value. Only `act_v2` is supported and it is the
  default. Reject `act_v1` with a clear message.
- At least one run name is required. Reject duplicates.

## Fixed configuration

| Setting | Value |
|---|---|
| GPU | One full RTX 4090 |
| Maximum price | $0.80/hour |
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| Disk | 100 GB |
| AWS profile | `toy-pickplace-backup` |
| S3 region | `ap-south-1` |
| S3 bucket | `toy-act` |
| S3 checkpoint source | `s3://toy-act/checkpoints/act_v2/<run-name>/` |
| S3 config source | `s3://toy-act/runs/act_v2/<run-name>/config.json` |
| S3 rollout destination | `s3://toy-act/rollouts/act_v2/<run-name>/` |
| Git remote | `https://github.com/skpro19/toy-act.git` |
| Git branch | `act-v2` |
| Dataset | `datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` |
| Dataset S3 source | `s3://toy-act/datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5` |
| Episodes per checkpoint | 30 |
| Steps per episode | 250 |
| Rendering | Off-screen (EGL, `--no-on-screen`) |
| Seed | 0 |
| Remote project | `/workspace/toy-act` |
| Remote control dir | `/workspace/toy-act/.vast-train` |
| Instance label | `toy-act-rollout-actv2-<timestamp>` (generated per invocation) |
| Local run state | `.vast-train-local/toy-act-<INSTANCE_ID>/` |

## Instance label

Construct the instance label once per invocation, before provisioning:

```bash
RUN_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_LABEL="toy-act-rollout-actv2-${RUN_TIMESTAMP}"
```

Do not regenerate `RUN_TIMESTAMP` later. Record `INSTANCE_LABEL` in `setup.env`.

## Local run state

Create one `.vast-train-local/toy-act-<INSTANCE_ID>/` directory per instance at
provisioning time (gitignored, never committed). Alongside the `vast-train-actv2`
files (`setup.env`, `instance.json`, `known_hosts`, `watcher.sh`, `watcher.pid`,
`watcher.log`, `report.txt`), store:

| File | Contents |
|---|---|
| `rollout-runs.txt` | The exact ordered run names passed to the sweep |
| `report.txt` | Adds each run's S3 rollout prefix and its verified artifact count to the base run report |

## S3 write access

The `toy-pickplace-backup` profile is prefix-scoped. It currently reads
`checkpoints/act_v2`, `runs/act_v2`, and `datasets`, and must also allow writes
under `rollouts/`. Preflight the scope with a throwaway probe:

```bash
PROBE_KEY="rollouts/act_v2/.probe/$(date +%s%N)"
PROBE_FILE=$(mktemp)
printf 'probe\n' > "$PROBE_FILE"
if AWS_PROFILE=toy-pickplace-backup aws s3api put-object \
    --bucket toy-act --key "$PROBE_KEY" --body "$PROBE_FILE"; then
  AWS_PROFILE=toy-pickplace-backup aws s3api delete-object \
    --bucket toy-act --key "$PROBE_KEY" || exit 1
else
  rm -f "$PROBE_FILE"
  exit 1
fi
rm -f "$PROBE_FILE"
```

If either command is denied, follow the AGENTS.md "Updating an existing inline
IAM policy" procedure to add a statement granting `s3:GetObject`,
`s3:PutObject`, and `s3:DeleteObject` on `arn:aws:s3:::toy-act/rollouts/*` (and
`s3:ListBucket` for the `rollouts/` prefix condition) to the workload user's
inline policy. This requires `aws login` with an admin profile and the user's
explicit authorization for browser-based authentication; do not run it without
that authorization. Respect the 2048-byte inline-policy limit and verify the
write probe succeeds with the workload profile before continuing.

## Workflow

1. Require `vastai`, `aws`, `jq`, `ssh`, `ssh-keyscan`, `rsync`, `git`, `tmux`,
   `flock`, `ss`, and `base64`. Verify `vastai show instances --raw` succeeds,
   `aws sts get-caller-identity --profile toy-pickplace-backup` succeeds, and
   `uv lock --check` succeeds.
2. Parse `RUN_NAMES` and `VERSION`; reject an empty list, duplicates, and
   `VERSION` other than `act_v2`.
3. For every run name, require at least one `epoch_*.pt` under
   `s3://toy-act/checkpoints/act_v2/<run>/` and the object
   `s3://toy-act/runs/act_v2/<run>/config.json`; stop and report every missing
   run before provisioning.
4. Pass the S3 write probe in "S3 write access" above.
5. Pin the code exactly as in `vast-train-actv2.md` step 2 (fetch `origin/act-v2`,
   set `GIT_COMMIT`, confirm anonymous cloning, warn on a dirty or diverged
   working tree).
6. Provision one RTX 4090 by following `vast-train-actv2.md` steps 3 through 9
   and the "Network quality acceptance" subsection unchanged, substituting the
   `toy-act-rollout-actv2-<timestamp>` label. Use a command-specific temporary
   `known_hosts` file and `StrictHostKeyChecking=yes` for all SSH.
7. Clone the pinned commit (`vast-train-actv2.md` step 10). The clone already
   contains `scripts/rollout_sweep_s3.py` and
   `.opencode/commands/scripts/vast-train/rollout-runner.sh`; do not transfer a
   local working tree.
8. Install dependencies exactly as in `vast-train-actv2.md` step 11. That step
   already installs the headless GL libraries and verifies off-screen EGL
   rendering, which the rollout sweep renderer also needs.
9. Transfer S3 credentials exactly as in `vast-train-actv2.md` step 12.
10. Ensure the dataset is in the bucket and fetch it on the instance exactly as
    in `vast-train-actv2.md` step 13, using the fixed dataset path above.
11. Create `/workspace/toy-act/.vast-train/{logs,state}` and write the run plan
    (`rollout-runs.txt`), one run name per line, over SSH standard input:

    ```bash
    printf '%s\n' "${RUN_NAMES[@]}" | \
      ssh -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes \
        -p $PORT root@$HOST \
        'cat > /workspace/toy-act/.vast-train/rollout-runs.txt'
    ```

12. Claim the lowest unused local workflow index with
    `.opencode/commands/scripts/vast-train/local-wrapper-lease.sh allocate
    "toy-act-$INSTANCE_ID"` and create only the SSH wrapper (no TensorBoard):

    ```bash
    tmux new-session -d -s "$SSH_SESSION" \
      "ssh -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes \
       -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
       -p $PORT root@$HOST"
    ```

13. Start the durable runner in a detached tmux session named `rollout` and wait
    until it publishes readiness:

    ```bash
    tmux new-session -d -s rollout \
      'bash /workspace/toy-act/.opencode/commands/scripts/vast-train/rollout-runner.sh'
    ```

    Poll for up to 30 seconds until
    `/workspace/toy-act/.vast-train/state/run-status` reads `running`; require
    the `rollout` session to survive an additional five-second window. A missing
    tmux session is never a success signal; the persisted state files are
    authoritative.
14. Hand off to a detached local watcher (`setsid`/`nohup`) and stop babysitting.
    The watcher owns cleanup and the local `VAST_API_KEY`. It must:
    - record `RUN_STARTED=yes` only after it reads remote `state/run-status` as
      `running`, and initialize `TERMINAL_CONFIRMED=no`;
    - poll every 30 seconds, retrying SSH failures indefinitely and never
      treating a Vast API or SSH failure as proof that the sweep disappeared;
    - set `TERMINAL_CONFIRMED=yes` only after a successful SSH probe directly
      reads remote `state/completed` or `state/failed`;
    - show concise progress from `.vast-train/logs/rollout.log` without
      flooding; note that `success_rate.png` and the results files are re-uploaded
      after every checkpoint, so the bucket reflects progress as it runs;
    - on success, verify through the workload profile that every run has all four
      objects under `rollouts/act_v2/<run>/` (`results.json`, `results.npz`,
      `success_rate.png`, `config.json`) before reporting success; on `failed`,
      still verify and record whichever artifacts exist;
    - before every `vastai destroy`, enforce the cleanup gate again: setup may
      destroy before `RUN_STARTED=yes`; after that point require
      `TERMINAL_CONFIRMED=yes`. Never let an EXIT trap, signal, missing tmux
      session, or connectivity error bypass the gate;
    - after an authorized destroy, verify the exact instance no longer appears in
      `vastai show instances --raw`, then write `report.txt` recording the pin
      commit, offer price, TensorBoard-free instance details, each run's rollout
      S3 prefix, and the verified artifact count.
15. Report the run names, each `s3://toy-act/rollouts/act_v2/<run>/` prefix, the
    pinned commit, the selected offer price, and the local run-state directory
    `.vast-train-local/toy-act-<INSTANCE_ID>/` (`watcher.log`, `report.txt`) to
    the user, then return without blocking. Do not keep polling in the
    interactive session.

## Checking progress

Because outputs are uploaded incrementally, progress is visible without SSH:

```bash
AWS_PROFILE=toy-pickplace-backup aws s3 ls \
  s3://toy-act/rollouts/act_v2/<run-name>/
tail -f .vast-train-local/toy-act-<INSTANCE_ID>/watcher.log
```

The `success_rate.png` in the bucket is refreshed after every checkpoint; its
latest `epoch_*` point is the current progress.

## Failure handling

Setup failures before remote `run-status=running` destroy the instance and
report. A per-checkpoint failure is recorded in the `failures` array of
`results.json` and the sweep continues with the next checkpoint; after all runs
finish, the orchestrator exits nonzero if any checkpoint failed, so the watcher
marks `state/failed`, still verifies and preserves the uploaded partial results,
reports the failures, and destroys the instance. The old run's checkpoints and
`runs/` prefix are never modified.
