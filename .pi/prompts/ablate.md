---
description: Run a config sweep on Vast.ai, one instance per combo
argument-hint: "[sweep-spec]"
---

Run every combination in a sweep spec, each in its own freshly provisioned
Vast.ai instance, by delegating to `.pi/prompts/vast-train-actv2.md`. Combos
run sequentially (one instance at a time). Generated configs are local only and
are never committed.

## Inputs

- `SWEEP_SPEC` = `${1:-configs/sweeps/bs32-seed.toml}`

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
   line per config: `pending|running|done <path>`.

4. For each `pending` config, in order:

   - Set `CONFIG_PATH=<path>`.
   - Follow `.pi/prompts/vast-train-actv2.md` for that single config. It will
     use the pre-set `CONFIG_PATH`, skip its interactive picker, and provision
     a fresh instance.
   - After that workflow returns (the run has been handed off to its detached
     watcher), mark the line `running` and move to the next config.
   - Never destroy an instance here: each run's watcher owns cleanup.

5. When the loop finishes or is interrupted, report per-combo status from the
   state file and any remaining `pending` configs.

## Constraints

- Sequential only: never provision more than one instance at a time.
- Reuse the vast-train cleanup gates unchanged; never bypass the watcher.
- Do not commit or push generated configs.
