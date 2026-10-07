---
description: List ablation run names, bucket links, and fixed/ablated params
argument-hint: "[iteration-id-or-dir]"
---

Print a read-only list of an ablation iteration's runs. The optional selector is
`$ARGUMENTS`: an iteration ID or a local iteration directory. This command never
writes files, downloads artifacts, mutates state, or calls Vast, and never prints
sensitive instance records.

## Select the iteration

1. Parse `$ARGUMENTS` as an optional iteration selector (the first non-flag
   token).
2. List candidate iterations: every `<spec>/<iteration>/` under
   `.vast-train-local/ablations/` whose `manifest.json` has `"kind": "sweep"`.
   Include `--plan` iterations; they simply have no recorded run names yet.
3. If a selector is present, select the one candidate whose iteration ID or
   resolved directory path exactly matches it. If it matches zero or several
   candidates, report the 5 most recent valid iteration IDs and stop.
4. If no selector is present, **always ask** with the `question` tool before
   doing anything else — even when only one candidate exists. Consider only the
   **5 most recent** candidates, ordered newest first by iteration ID
   (`YYYYMMDDTHHMMSSZ-...` sorts lexicographically), and read fields with
   read-only tools:
   - `<iteration>/manifest.json`: `created_at`;
   - `<iteration>/state.json`: `driver_status` and each combo `status`.
   Use these option fields:
   - label: `<spec-stem> · <iteration-id>`;
   - description: `created <created_at> · driver <driver_status> · <combo
     summary>`, where the combo summary is counts such as `6 combos: 5 running,
     1 pending`.
5. If there are no candidates, report that no saved sweep iteration exists and
   stop.

## Report

Run the read-only helper once for the selected directory, with the path quoted:

```bash
uv run --frozen python \
  .pi/prompts/scripts/vast-train/ablation_runs.py runs "$ITERATION_DIR"
```

Keep stdout and stderr visible. The helper prints the run details for the
selected iteration as exactly three Markdown tables, in this order — they are
the entire result, so do not reformat them as prose or lists:

- **Fixed params** — the `[fixed]` rows from the sweep spec snapshot recorded
  in the iteration's `manifest.json`.
- **Ablated params** — the `[grid]` rows from the same snapshot.
- **Runs** — one row per run started by this iteration, newest first, with its
  `s3://toy-act/runs/act_v2/<run>/` folder location. Runs are collected from
  `state.json`, each combo's local `handoff.json`/`report.txt` (for runs that
  started but failed before handoff), and the two S3 prefixes filtered by the
  iteration's `-i<iteration-hash>` run suffix.

## Limits

- Read-only: it reads saved manifest, state, and sweep spec snapshots, and runs
  non-mutating `aws s3 ls` on the two iteration prefixes. It never downloads,
  launches, reconciles, cleans up, or stops a service.
- The S3 listings are non-recursive folder listings; use the `report --files`
  mode for a full object inventory. A failed S3 listing falls back to the
  locally recorded run names without failing the command.
- Never print `instance.json`, `known_hosts`, `setup.env` values, credentials,
  or presigned URLs. The helper reports run names and bucket prefixes only.
