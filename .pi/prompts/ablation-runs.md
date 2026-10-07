---
description: Report all recorded information for a saved ablation iteration in tables
argument-hint: "[iteration-id-or-dir] [--combo <id>] [--files] [--no-s3]"
---

Print a read-only, all-tabular report for one saved ablation iteration. The
optional selector is `$ARGUMENTS`; it may be an iteration ID or a local
iteration directory, followed by any of `--combo <id>`, `--files`, or
`--no-s3`. This command never writes files, downloads artifacts, mutates state,
or calls Vast, and never prints sensitive instance records.

## Select the iteration

1. Parse `$ARGUMENTS` into an optional iteration selector (the first non-flag
   token) and the optional flags `--combo <id>`, `--files`, `--no-s3`.
2. List candidate iterations: every `<spec>/<iteration>/` under
   `.vast-train-local/ablations/` whose `manifest.json` has `"kind": "sweep"`.
   Include `--plan` iterations; they simply have no recorded TensorBoard URL.
3. If a selector is present, select the one candidate whose iteration ID or
   resolved directory path exactly matches it. If it matches zero or several
   candidates, report the valid iteration IDs and stop.
4. If no selector is present, **always ask** with the `question` tool before
   doing anything else — even when only one candidate exists. Build one option
   per candidate, ordered newest first by iteration ID (`YYYYMMDDTHHMMSSZ-...`
   sorts lexicographically), and read fields with read-only tools:
   - `<iteration>/manifest.json`: `created_at`;
   - `<iteration>/state.json`: `driver_status` and each combo `status`;
   - `<iteration>/tensorboard/service.json`: `port` and `url` when present.
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
  .pi/prompts/scripts/vast-train/ablation_runs.py report "$ITERATION_DIR" \
  [--combo "$COMBO"] [--files] [--no-s3]
```

Keep stdout and stderr visible. The helper prints these Markdown tables: the
iteration summary, configs, execution, verification, local records, S3
artifacts, S3 prefixes, TensorBoard cache, live progress for nonterminal combos,
and — only with `--files` — a full local and S3 file inventory.

## Limits

- Read-only: it lists files and runs non-mutating `aws s3 ls` only. It never
  downloads, launches, reconciles, cleans up, or stops a service.
- `--no-s3` skips the S3 artifact and inventory listings; `--files` expands full
  object and file inventories.
- Never print `instance.json`, `known_hosts`, `setup.env` values, credentials,
  or presigned URLs. The helper reports names, sizes, and times only.
