---
description: Run a fresh, logged Vast.ai sweep; explicitly resume an iteration
argument-hint: "<sweep-spec> [--plan] [--resume <iteration-id>]"
---

Run an ACT v2 sweep using the committed driver:
`.pi/prompts/scripts/vast-train/workflow.py`.
Read this file and `vast-train-actv2.md` before execution. Do not reconstruct
provisioning, state updates, or cleanup through ad hoc shell commands.

## Inputs and invocation

User arguments: `$ARGUMENTS`.

- Require an explicit sweep spec path unless `--resume <iteration-id-or-dir>`
  is supplied. Ask only if a required input is missing; never guess a config.
- Parse the spec path, optional `--plan`, and optional `--resume`; reject other
  arguments. Pass paths as quoted arguments, not interpolated shell programs.
- A normal invocation **always creates a new iteration**, even for identical
  configs. Never search for old `.state`/`.commit` files, reuse an older
  iteration implicitly, or ask about completed/interrupted historical runs.
- `--resume` continues only the specified iteration, using its saved configs
  and pinned SHA. The source spec may have changed; it is not resolved again.
- `--plan` creates immutable configs/state but performs no Vast/AWS operations.
  Use the printed iteration ID with `--resume` to launch that saved plan.

```bash
uv run --frozen python .pi/prompts/scripts/vast-train/workflow.py \
  --spec "$SWEEP_SPEC"
# Add --plan for a local plan, or --resume "$ITERATION_ID" for explicit recovery.
# Resume without an original spec: workflow.py --resume "$ITERATION_ID"
```

The driver performs Git preflight before resolving project configs, then
rechecks the pinned revision before provisioning/setup/launch. Require a clean,
anonymously synchronized `act-v2` checkout. Never automatically stash, commit,
push, pull, reset, switch branches, or bypass a failed check.

## Execution contract

The driver, not the agent, owns:

- unique iteration IDs, immutable input/config snapshots and their hashes,
  an ordered manifest (no directory glob), and iteration-specific run names;
- a nonblocking iteration lock and atomic/fsynced state updates;
- durable create intent and exact instance labels **before** create requests;
  ambiguous requests reconcile on resume, never blindly create replacements;
- one provisional rental at a time, up to three supported offers per combo,
  hardware/network gates, bounded remote setup and dataset validation;
- run-directory recording, local wrapper leases, conservative detached watcher
  startup, service launch and handoff;
- a fresh shared `check-handoff.py` exit-0 check, followed by an independent
  fresh check, before advancing to the next combo;
- reconciliation of this iteration only. Training outcome, artifact verification,
  and verified removal are recorded separately. Historical iterations cannot
  block a fresh invocation.

Training overlaps across combos; each instance has its own watcher. The driver
returns after handoff rather than waiting for training completion. Live watchers
finish reports independently; `summary.txt` is a driver-time snapshot, refreshed
on explicit resume, not a continuously updated dashboard.

## Shared TensorBoard dashboard

Each fresh ablation iteration gets **one local TensorBoard server**, shared by
all its combos. Separate `/ablate` invocations get separate servers and URLs;
explicit resume reuses or recovers only that iteration's recorded services.
`--plan` starts no services and makes no S3 requests.

The driver starts iteration-owned tmux services and records their allocated
port/URL in `tensorboard/service.json`. A detached synchronizer reads only this
iteration's recorded run prefixes from `s3://toy-act/runs/act_v2/<run>/`, using
profile `toy-pickplace-backup`, and downloads only event files into distinct
`tensorboard/logs/<combo>/<run>/` directories. It retries every 30 seconds,
including after handoff and completion, so final uploads remain discoverable.
Downloads are staged outside the watched log directory. Successful snapshots
append to the cached event file after verifying its existing bytes match;
shrinking or rewritten snapshots fail closed. This lets TensorBoard keep reading
the same file instead of freezing on an older replaced file.
Metrics lag the remote backup cadence (120-second pauses plus upload time),
local synchronization and TensorBoard reload. This is not a real-time SSH feed.

The server and synchronizer remain available after training finishes; completed
runs stay cached for comparison. They are not reboot-supervised: explicit resume
recovers them against the same cache and recorded port. Port/session conflicts
block recovery; never steal an occupied port or kill an unrelated service.
Diagnostics live in `tensorboard/server.log`, `tensorboard/driver.log`,
`tensorboard/events.jsonl`, and `tensorboard/sync-status.json` beneath the iteration.
To refresh a completed iteration's dashboard without running training stages:

```bash
uv run --frozen --only-group train python \
  .pi/prompts/scripts/vast-train/ablation_tensorboard.py refresh "$ITERATION_DIR"
```

This verifies ownership before restarting only that iteration's dashboard
sessions, downloads its recorded event files, and reuses its saved port/cache.
It never provisions, launches training, reconciles outcomes, or removes instances.
An unrelated occupied port blocks recovery; no unrelated service is stopped.

Report the **shared iteration URL** for all combos, not their internal forwarding
URLs. Existing per-instance TensorBoard servers and forwarding remain internal
handoff checks; the shared dashboard never authorizes instance cleanup.

## Safety and failures

- Preserve `vast-train-actv2.md`'s hardware, price, Git and cleanup policies.
  Never weaken filters without asking.
- Provisioning failures may remove only exact, provisional instances **before**
  training launch intent. Removal must be verified by a successful, valid API
  response before a replacement is rented.
- Once launch intent is recorded, only the detached watcher owns cleanup.
  It requires a directly observed remote terminal marker, including on restart
  and in its EXIT trap. SSH/API failures, interruption, or missing tmux sessions
  never authorize destruction.
- A resume may restart a dead saved watcher and restore local forwarding. It
  does not rerun terminal training or rent replacements for uncertain attempts.
- Checker exit 1/2 blocks new launches; exit 3 requires a watcher report plus
  confirmed removal before terminal reconciliation. Do not mark terminal runs
  `running`, erase state, or bypass the gate to make the loop advance.
- Driver exit 0 means handed off/reconciled (or planned), not training success;
  1 means blocked, 2 invalid CLI input, 130 interrupted. A blocked/interrupted
  iteration is recoverable through explicit resume. A new invocation remains
  independent. Never perform global old-run cleanup.

## Logging and response

Keep driver stdout/stderr visible; do not redirect or suppress them. The driver
prints the iteration ID, log directory, resume command, combo count, elapsed
stage updates, and command start/finish messages. It streams redacted output
from configured diagnostic logs to stderr and emits waiting heartbeats every
30 seconds during commands. Sensitive and unlogged raw command output stays
hidden. Each verified handoff prints a current summary, including TensorBoard
URLs; handoff is not training completion. Detached watcher output remains in
its local logs after the driver returns. These updates apply to new driver
processes, not an already-running process.

All records are ignored and local:
`.vast-train-local/ablations/<spec-name>/<iteration-id>/` contains
`manifest.json`, `state.json`, `inputs/`, `configs/`, `events.jsonl`,
`driver.log`, `preflight.log`, `resolution.log`, `combos/`, and `summary.txt`.
Per-instance `.vast-train-local/toy-act-<ID>/` contains setup logs, pinned host
keys, sensitive mode-600 `instance.json`, `dataset.json`, hashed setup/watcher
snapshots, watcher output/log, `handoff.json`, and final `report.txt`. Interrupted
nonterminal watcher reports are preserved in `report-history/` before recovery.

Keep the foreground driver bounded with a timeout appropriate to setup/dataset
size; TERM records an interruption and never cleans up live runs. If interrupted,
report the iteration ID and resume command; do not restart the whole command as
an implicit retry. No reboot supervision is installed: resume explicitly after
a reboot to recover this iteration's watchers.

Report one concise table with combo status, instance ID and TensorBoard URL,
then the iteration directory, logs and explicit resume command if needed. For
failures, include the relevant local diagnostic path. Do not repeatedly ask
about old runs or poll training after handoff. Never print credentials, presigned
URLs or raw sensitive instance records; never commit generated state/configs.
