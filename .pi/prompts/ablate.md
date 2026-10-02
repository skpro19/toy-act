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

## Run gate

A combo is **ready to hand off**, and therefore releases the loop to start the
next combo, only when all of the following are true for that combo:

- remote `state/run-status` reads `running` (training started);
- the local TensorBoard URL `http://localhost:$TB_PORT/` responds (SSH
  forwarding works);
- `state/backup-running` has been observed (backup wrapper running);
- the detached watcher has been launched and recorded `RUN_STARTED=yes`.

`vast-train-actv2.md` returns immediately after handoff, which is after all four
conditions above. Verify the gate before marking a combo `running`; never start
the next combo until the current combo has been handed off.

## Workflow

1. Resolve the sweep into self-contained configs:

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
   - `running <run_dir>` — handed off to its watcher (satisfies the run gate);
     `<run_dir>` is the `.vast-train-local/toy-act-<INSTANCE_ID>/` directory
     recorded at handoff, and the instance is still active;
   - `done` — watcher reported a terminal state and the instance is gone;
   - `failed` — watcher reported a training or setup failure;
   - `stalled` — watcher died with no terminal marker (report; never destroy).

4. Reconcile the state file before starting each combo: for every `running`
   line, inspect its recorded `run_dir`. If `report.txt` exists and the
   instance is gone, mark the line `done` or `failed`. If the watcher is dead
   with no terminal marker, mark it `stalled` and stop to report — never
   destroy an instance here.

5. For each `pending` config, in order:

   - Set `CONFIG_PATH=<path>`.
   - Follow `.pi/prompts/vast-train-actv2.md` for that single config. It will
     use the pre-set `CONFIG_PATH`, skip its interactive picker, and provision
     a fresh instance.
   - After that workflow returns, verify the run gate for this combo, then mark
     the line `running <run_dir>`.
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
