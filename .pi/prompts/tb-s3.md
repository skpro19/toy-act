---
description: Restart or refresh a saved sweep's shared S3-backed TensorBoard dashboard
argument-hint: "[iteration-id-or-dir]"
---

Restart a saved ablation iteration's shared TensorBoard server and S3
synchronizer without running any training stage. The optional selector is
`$ARGUMENTS`; it may be an iteration ID or a local iteration directory. This
command never provisions, launches, reconciles, or destroys anything, and never
steals an occupied port or stops an unrelated service.

## Scope

This command only:

- verifies the selected iteration owns its recorded `ablate-tb-*` and
  `ablate-sync-*` tmux sessions;
- re-downloads that iteration's recorded event files from S3 into its cache;
- restarts only those sessions on the iteration's recorded port.

It must never call `vastai create`, `vastai stop`, or `vastai destroy`, and must
never change a combo status or remove an instance. A clean Git checkout is not
required; this command does not run Git preflight.

## Select the iteration

1. Require `uv`, `tmux`, `curl`, `find`, and S3 access (`.env` or
   `AWS_PROFILE=toy-pickplace-backup`).
2. List candidate iterations: every `<spec>/<iteration>/` under
   `.vast-train-local/ablations/` whose `manifest.json` has `"kind": "sweep"`
   and that already contains `tensorboard/service.json`. `--plan` iterations
   have no `service.json`, so they are not candidates.
3. If `$ARGUMENTS` is non-empty, select the one candidate whose iteration ID or
   resolved directory path exactly matches it. If it matches zero or several
   candidates, report the valid iteration IDs and stop.
4. If `$ARGUMENTS` is empty, **always ask** with the `question` tool before doing
   anything else — even when only one candidate exists. Build one option per
   candidate, ordered newest first by iteration ID (`YYYYMMDDTHHMMSSZ-...` sorts
   lexicographically), and read fields with read-only tools:

   - `<iteration>/manifest.json`: `created_at`;
   - `<iteration>/state.json`: `driver_status` and each combo `status`;
   - `<iteration>/tensorboard/service.json`: `port` and `url`;
   - dashboard ownership: `tmux has-session -t ablate-tb-<iteration-id>`;
   - recorded-port state: `own` when that session is live, otherwise `free`
     when a bind test to `127.0.0.1:<port>` succeeds, or `held` when it does
     not.

   Use these option fields:

   - label: `<spec-stem> · <iteration-id>`;
   - description: `created <created_at> · driver <driver_status> · <combo
     summary> · <recorded URL> · dashboard <own|free|held>`, where the combo
     summary is counts such as `6 combos: 5 done, 1 verification_failed`.

   Ports are reused across iterations, so a stopped iteration can record a port
   that another live dashboard now holds. Mark it `held` instead of implying the
   URL currently serves this iteration, and warn that a `held` port makes
   `refresh` fail closed. Do not print `state.json` offer details or any
   sensitive record.
5. If there are no candidates, report that no saved sweep dashboard exists and
   stop.

## Refresh

Run exactly one refresh for the selected directory, with the path quoted:

```bash
uv run --frozen --only-group train python \
  .pi/prompts/scripts/vast-train/ablation_tensorboard.py refresh "$ITERATION_DIR"
```

Keep stdout and stderr visible; do not redirect or suppress them. The helper
checks session ownership before stopping anything, downloads event files into
`tensorboard/staging/`, appends only matching snapshot growth to the cached
watched files, and restarts the iteration's two tmux sessions on the saved port.
It makes no Vast or training calls.

## Report and failures

- On success, report the iteration ID, the recorded URL from
  `tensorboard/service.json`, and the diagnostics directory `.../tensorboard/`
  (`server.log`, `driver.log`, `sync-status.json`).
- If the recorded port is occupied or a download fails, report the relevant local
  diagnostic path and stop. Never kill an unrelated process and never steal an
  occupied port.
- Tell the operator to refresh the browser tab if the charts look stale.

No reboot supervision is installed. Rerun `/tb-s3` after a restart to recover the
iteration's dashboard against the same cache and recorded port.
