---
description: Train ACT v2 on a fresh Vast.ai RTX 4090 with durable local logging
argument-hint: "<config-path> [--plan] [--resume <iteration-id>]"
---

Train ACT v2 using `.pi/prompts/scripts/vast-train/workflow.py`. This same
script is the ablation driver; do not reproduce its implementation with ad hoc
commands. The agent owns invocation and reporting, not background monitoring.

## Inputs and invocation

User arguments: `$ARGUMENTS`.

Use an explicitly supplied config path, or a caller-provided `CONFIG_PATH`.
Without either, ask for the path; no interactive picker or default config.
An explicit `--resume <iteration-id-or-dir>` needs no original config. Support
`--plan` for config resolution/snapshotting without Vast/AWS operations; reject
other arguments. Quote paths when constructing tool commands.

```bash
uv run --frozen python .pi/prompts/scripts/vast-train/workflow.py \
  --config "$CONFIG_PATH"
# Optional: --plan, or --resume "$ITERATION_ID" (not both).
# If supplied by a caller: --expected-commit "$SWEEP_GIT_COMMIT"
```

Each ordinary invocation starts fresh. Repeated use of a config never inherits
old completion, interruption, or pinned-revision state. Resume is explicit and
uses immutable saved configs; source files are not resolved again. Ablations
invoke the shared driver once with `--spec`, not nested single-config workflows.

## Non-negotiable policy

| Setting | Requirement |
|---|---|
| Git | Clean, synchronized `act-v2`; anonymous clone of `https://github.com/skpro19/toy-act.git`; pinned SHA verified remotely |
| GPU / price | Exactly one full RTX 4090, approximately 24 GiB; offer at most $0.80/hour; record actual price separately |
| Image / disk | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`; 100 GB rental; at least 100 GB available at `/workspace` |
| CPU / RAM | At least 24 allowed online physical cores; supported Zen 3+ family; quota/logical allocation at least 90% of advertised effective vCPUs; at least 64 GB allocated RAM, consistent with offer |
| GPU power / PCIe | At least 400 W; Gen4 x16 capability; advertised PCIe bandwidth at least 20 GB/s; thermal and power-brake slowdown inactive |
| Offer connectivity | Advertised download ≥500 Mbps, upload ≥200 Mbps, reliability ≥0.99; rentable/verified, no active display |
| Measured S3 network | Median of 7 successful PUTs ≤5000 ms; median of 3 successful 4 MiB uploads ≥1000 KB/s |
| Offer ordering | EPYC 9005, EPYC 9004, Threadripper 7000, EPYC 7003, Ryzen 7000/9000; then price, disk bandwidth, reliability |
| Retry / quarantine | At most three machines per combo; GPU-start failures quarantine machine ID for 24 hours; search again before replacement |
| AWS | Profile `toy-pickplace-backup`, region `ap-south-1`; do not request administrative authentication |
| Data | Config's project-relative `datasets/` path; S3 URI `s3://toy-act/<path>`; verify size, local SHA when available, and requested cameras |
| Artifacts | `s3://toy-act/checkpoints/act_v2/<run>/` and `s3://toy-act/runs/act_v2/<run>/` |
| Remote project | `/workspace/toy-act`; control directory `.vast-train` |

Never add `gpu_frac=1` (it excludes full single GPUs on multi-GPU hosts), weaken
filters, substitute another revision, or automatically repair Git. Ignored local
configs/state are allowed; other dirty or unsynchronized checkouts stop launches.
Unknown or missing hardware/API data fails closed.

## Script-owned stages

| Stage | Canonical implementation |
|---|---|
| Iteration identity, snapshots, state, locks, summaries | `iteration.py`, `workflow.py`, `workflow_common.py` |
| Git revision gate | `git-preflight.sh` |
| Local tools, AWS access, lock/config/dataset-source checks | `setup_run.py` |
| Search, ranking, durable create intent, exact-label reconciliation, provisional cleanup | `provision.py` |
| Allowed CPU/cgroup/GPU/disk measurements and validation | `hardware-probe.sh`, `hardware_gate.py` |
| S3 latency/upload measurements | `network-gate.sh` (signed URLs sent via SSH stdin, never logged) |
| Clone/pin, dependencies, CUDA/import/EGL checks | `remote-setup.sh` |
| Config/credential transfer, dataset sync and integrity/camera checks | `setup_run.py`, `verify-dataset.sh` |
| Idempotent TensorBoard/training/backup launch | `start-services.sh`, existing `runner.sh`, `ckpt-bkp-wrapper.sh` |
| Local wrapper allocation and forwarding | `local-wrapper-lease.sh`, `setup_run.py` |
| Per-ablation shared S3-backed dashboard and event synchronization | `ablation_tensorboard.py` |
| Detached monitoring, terminal verification and cleanup | Saved copy of `local-watcher.sh` |
| Authoritative read-only handoff gate | `check-handoff.py` |

All implementation lives under `.pi/prompts/scripts/vast-train/`, apart from
project config/training modules in `scripts/`. `.vast-train-local/` contains
runtime records, not canonical helpers. Per-run watcher copies are intentional
historical snapshots and remain unchanged on recovery.

Dependencies are installed with `uv sync --frozen --only-group train`; setup
verifies CUDA, rollout imports and offscreen EGL before launching. Every remote
command has a timeout. No AWS/Vast secret is embedded in the clone or repository.
Temporary credential files are mode 600 and removed locally after transfer;
`VAST_API_KEY` is never sent to the instance. Dataset upload retries are bounded;
never delete datasets with the workload profile.

## Cleanup and handoff contract

- Before training launch intent, provisioning may remove only its exact
  provisional rental. API failures or malformed output never prove removal;
  unverified removal stops retries.
- The driver persists launch intent and starts a conservative detached watcher
  **before** contacting the remote launch helper. From that moment, the driver
  must not destroy the instance—even if SSH fails before returning a result.
- A conservative/restarted watcher requires a directly read remote `completed`
  or `failed` marker before destruction. Interruption, suspend, signals, shell
  errors, API/SSH failures, or a missing train session cannot bypass this gate.
- The watcher holds an exclusive per-run lock. It retries SSH indefinitely,
  observes training, publishes a PID/start-ticks-bound readiness acknowledgement,
  and owns backup verification, exact-instance cleanup and the final report.
- Successful training triggers final-backup waiting, expected checkpoint/config
  verification and TensorBoard scalar verification. Failed training preserves
  diagnostics and best-effort backup. Outcome, verification and cleanup remain
  separate; a successful training outcome alone is not `done`.
- Handoff requires live nonterminal training; a working recorded TensorBoard
  forward; live backup with artifact/last-success markers and no failure marker;
  and a matching live watcher acknowledgement. Only a fresh checker exit 0
  permits handoff. Exit 1/2 blocks launches; exit 3 waits for watcher report and
  confirmed removal. No checker result authorizes cleanup.
- An explicit resume may recover the same instance, restart a dead saved watcher
  and restore the same local forwarding lease. Interrupted nonterminal watcher
  reports are archived before recovery; missing/modified saved setup or watcher
  snapshots block restart. It never reruns terminal training or blindly replaces
  an uncertain create request. Ambiguous setup/report states
  remain blocked with diagnostics rather than being destroyed automatically.

The detached watcher survives session return/suspend, not host reboot. No reboot
supervisor is installed. Resume the recorded iteration after reboot; do not
silently create replacements. Historical unrelated runs remain untouched.

## Local records and response

Iteration record:
`.vast-train-local/ablations/<config-stem>/<iteration-id>/` with manifest/state,
immutable inputs/configs, timestamped `driver.log`, structured `events.jsonl`,
stage diagnostics, and driver-time `summary.txt`.

Instance record:
`.vast-train-local/toy-act-<ID>/` with `setup.env`, sensitive `instance.json`,
`dataset.json`, `known_hosts`, streaming `setup.log`, hashed `watcher.sh`,
`watcher.lock`, `watcher.pid`, `watcher.ready`, `watcher.out`, `watcher.log`,
`handoff.json`, `report.txt`, and archived interrupted reports in `report-history/`.
Local wrapper indices/ports are allocated per instance; use the recorded
TensorBoard URL, never assume port 6006. Resume restores the recorded lease after
reboot only if the slot is free or still owned by this instance. Occupied slots
are never stolen, and old watcher cleanup cannot release a newer run's lease.
A lost forward does not stop training.

For sweep invocations only, `ablation_tensorboard.py` also manages one shared
local dashboard per iteration. All its combos report that URL; a fresh sweep
gets a separate server, while resume recovers the same service/cache. Event
files are downloaded from the iteration's registered S3 run prefixes every
30 seconds, so display updates lag remote backups. Services persist after
handoff/completion for comparison and final uploads. Records, cached event
files, server logs and sync status live under the iteration's `tensorboard/`.
Occupied recorded ports or unrelated sessions are never stolen. Downloads are
staged outside the watched logdir; successful snapshots append after checking
the cached prefix matches, preserving the file TensorBoard is already reading.
A completed sweep's dashboard can be refreshed without training or instance
operations using:

```bash
uv run --frozen --only-group train python \
  .pi/prompts/scripts/vast-train/ablation_tensorboard.py refresh "$ITERATION_DIR"
```

Refresh checks session ownership, restarts only that iteration's dashboard
services, and retains its recorded port/cache. It does not reconcile outcomes.
The `/tb-s3` prompt wraps this recovery; pass an iteration ID or directory, or run
it with no argument to choose from the saved dashboards.
Per-instance forwarding remains an internal handoff requirement; single-config
invocations continue reporting their original forwarded URL. `--plan` starts no services.

Run the foreground driver with a bounded timeout appropriate to the dataset.
On interruption, report the saved iteration ID/resume command; no implicit fresh
retry. Exit 0 means handoff/reconciliation or plan creation, not training success;
1 blocked, 2 invalid CLI, 130 interrupted.

After handoff, report config/run identity, instance ID, price, pinned SHA,
TensorBoard URL, expected S3 prefixes and local log paths; return without polling
training. On failure, report the relevant stage diagnostic and explicit resume
command. Do not ask about unrelated historical runs, print secrets/presigned URLs,
commit generated configs/state, or install pi-session cleanup traps.
