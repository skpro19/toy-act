---
description: Run a config sweep on Vast.ai, one instance per combo
argument-hint: "<sweep-spec>"
---

Run every combination in a sweep spec, each in its own freshly provisioned
Vast.ai instance, by delegating to `.pi/prompts/vast-train-actv2.md`. Combos
are provisioned one at a time, but their training runs overlap: the next combo
starts as soon as the previous has been handed off to its detached watcher (see
"Run gate" below). Generated configs are local only and are never committed.

## Inputs

- `SWEEP_SPEC` = `${1:-}` — required. If it is empty, stop and ask the user for a
  sweep spec path before doing anything else.

## Mandatory Git preflight and sweep revision

Before resolving the sweep or running local project Python, run the shared
preflight used by `vast-train-actv2.md`:

```bash
SWEEP_NAME="$(basename "$SWEEP_SPEC" .toml)"
SWEEP_COMMIT_FILE=".vast-train-local/ablate-${SWEEP_NAME}.commit"
SWEEP_STATE_FILE=".vast-train-local/ablate-${SWEEP_NAME}.state"
PREFLIGHT_ARGS=()
if [ -f "$SWEEP_COMMIT_FILE" ]; then
  SWEEP_GIT_COMMIT=$(<"$SWEEP_COMMIT_FILE")
  [[ "$SWEEP_GIT_COMMIT" =~ ^[0-9a-f]{40,64}$ ]] \
    || { echo 'Invalid persisted sweep SHA; stop.' >&2; exit 1; }
  PREFLIGHT_ARGS+=(--expected-commit "$SWEEP_GIT_COMMIT")
elif [ -e "$SWEEP_STATE_FILE" ]; then
  echo 'Existing sweep state has no pinned SHA; stop rather than guess.' >&2
  exit 1
fi
SWEEP_GIT_COMMIT=$(timeout 120 bash .pi/prompts/scripts/vast-train/git-preflight.sh \
  "${PREFLIGHT_ARGS[@]}") \
  || { echo 'Git preflight failed; stop the sweep.' >&2; exit 1; }
if [ ! -f "$SWEEP_COMMIT_FILE" ]; then
  mkdir -p .vast-train-local
  (set -o noclobber; printf '%s\n' "$SWEEP_GIT_COMMIT" > "$SWEEP_COMMIT_FILE") \
    || { echo 'Sweep revision already claimed; stop this invocation.' >&2; exit 1; }
fi
```

Require a clean `act-v2` checkout matching the current anonymous GitHub branch.
Ignored local configs and run state are allowed; staged changes, unstaged
tracked changes, non-ignored untracked files, wrong branches, detached HEAD,
and ahead/behind/diverged histories are errors. Remote failures are fatal.
Do not automatically stash, commit, push, pull, reset, or switch branches.

Persist `SWEEP_GIT_COMMIT` in
`.vast-train-local/ablate-${SWEEP_NAME}.commit` alongside the state file before
launching any combos. On resume, load that existing SHA instead of replacing
it with the current branch tip, and require the preflight to succeed with
`--expected-commit "$SWEEP_GIT_COMMIT"` before resolving configs. If an existing
state file has no recorded SHA, stop and report; do not guess its revision.
Use a new sweep name/state for an intentionally different code revision.

Keep this SHA unchanged for the entire sweep. Every combo's training workflow
must receive `SWEEP_GIT_COMMIT` and recheck it before config resolution and
provisioning. A changed checkout or remote branch stops additional launches;
existing runs and watchers remain untouched.

## Run gate

A combo is **ready to hand off**, and therefore releases the loop to start the
next combo, only when all of the following are true for that combo:

- remote `state/run-status` reads `running`, the `train` session exists, and
  neither `completed` nor `failed` exists;
- the recorded local TensorBoard URL responds and its forwarding session exists;
- the `ckpt-bkp` session exists, `backup-running`, `backup-artifact-ready`, and
  `backup-last-succeeded` exist, and `backup-failed` does not;
- the watcher is alive and its atomic `watcher.ready` acknowledgement matches
  this instance, PID, and Linux process start ticks. It publishes this only
  after observing training and arming its cleanup gate.

The authoritative read-only checker is
`.pi/prompts/scripts/vast-train/check-handoff.py`. After the delegated workflow
returns, perform a fresh check for its recorded run directory:

```bash
uv run --frozen python .pi/prompts/scripts/vast-train/check-handoff.py "$RUN_DIR"
HANDOFF_RC=$?
```

Capture the exit status explicitly (use an `if` or temporarily disable `errexit`
if needed). Only exit **0** permits marking the combo `running` and moving on.
Exit **1** means not ready/unknown, **2** means invalid configuration, and **3**
means a remote terminal marker was observed. Output is JSON with per-condition
`passed`, `pending`, `failed`, or `unknown` results. No check can authorize
cleanup. On 1/2, stop new launches and report; do not re-provision the same
combo. On 3, reconcile via the watcher's report and confirmed instance removal
before marking `done`/`failed`; stop and report if reconciliation is incomplete.
Never mark a terminal run `running` or bypass the watcher.

## Workflow

1. Set `SWEEP_NAME="$(basename "$SWEEP_SPEC" .toml)"`, perform the mandatory Git
   preflight (honoring any persisted sweep revision), then resolve the sweep
   into self-contained configs:

   ```bash
   SWEEP_NAME="$(basename "$SWEEP_SPEC" .toml)"
   RESOLVE_DIR=".vast-train-local/ablate-${SWEEP_NAME}"
   uv run python -m scripts.ablate --spec "$SWEEP_SPEC" --resolve-dir "$RESOLVE_DIR"
   ```

   This writes one `<slug>.toml` per combo, each containing the full effective
   config and `name = <slug>`.

2. Build the ordered config list:

   ```bash
   mapfile -t CONFIGS < <(ls -1 "$RESOLVE_DIR"/*.toml | sort)
   ```

3. Maintain a state file `.vast-train-local/ablate-${SWEEP_NAME}.state` with one
   line per config: `STATUS <path> [run_dir]` where `STATUS` is one of:

   - `pending` — not started yet;
   - `awaiting_handoff <run_dir>` — instance provisioned, but handoff has not
     passed; record its run directory as soon as it exists, before checking the
     gate. Never provision a duplicate instance for this combo on resume;
   - `running <run_dir>` — handed off to its watcher (satisfies the run gate);
     `<run_dir>` is the `.vast-train-local/toy-act-<INSTANCE_ID>/` directory
     recorded at handoff, and the instance is still active;
   - `done` — watcher reported a terminal state and the instance is gone;
   - `failed` — watcher reported a training or setup failure;
   - `stalled` — watcher died with no terminal marker (report; never destroy).

4. Reconcile the state file before starting each combo: for every `running`
   or `awaiting_handoff` line, inspect its recorded `run_dir`. If `report.txt` exists and the
   instance is gone, mark the line `done` or `failed`. If the watcher is dead
   with no terminal marker, mark it `stalled` and stop to report — never
   destroy an instance here. For a nonterminal `awaiting_handoff` run, recheck
   the gate on that same instance; promote it to `running` only on exit 0.
   Otherwise stop new launches and report, rather than creating a replacement.

5. For each `pending` config, in order:

   - Set `CONFIG_PATH=<path>` and pass the unchanged `SWEEP_GIT_COMMIT`.
   - Require the delegated Git preflight to succeed with that expected SHA.
     If it fails, stop the sweep, leave this and later combos `pending`, report
     the reason, and leave all previously launched runs and watchers untouched.
   - Follow `.pi/prompts/vast-train-actv2.md` for that single config. It will
     use the pre-set `CONFIG_PATH`, skip its interactive picker, and provision
     a fresh instance.
   - Record `awaiting_handoff <run_dir>` as soon as the instance run directory
     exists. After the workflow returns, invoke the shared checker above. Only
     exit 0 permits marking the line `running <run_dir>`; otherwise follow the
     failure/terminal handling above and stop new launches.
   - Start the next `pending` config immediately; the current combo's watcher
     owns its run and cleanup from here on.
   - Never destroy an instance here: each run's watcher owns cleanup.

6. When the loop finishes or is interrupted, report per-combo status from the
   state file and any remaining `pending` or `stalled` configs.

## Constraints

- Provision one instance at a time; training runs may overlap across combos,
  each with its own instance and watcher.
- Reuse the vast-train cleanup gates unchanged; never bypass the watcher.
- Do not commit or push generated configs.
- Never mix code revisions within a sweep or bypass a failed Git preflight.
