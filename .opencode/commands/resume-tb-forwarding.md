---
description: Resume TensorBoard SSH forwarding for an active ACT v2 Vast run
agent: build
---

Restore the local TensorBoard SSH port-forward for an existing Vast.ai training
run. The optional selector is `$ARGUMENTS`; it may be an instance ID, instance
label, or local run-state directory name. Do not provision, stop, or destroy an
instance, and do not start, stop, or signal the training or backup sessions.

## Discovery

1. Require `vastai`, `jq`, `ssh`, `ssh-keyscan`, `ssh-keygen`, `tmux`, `curl`,
   `flock`, and `ss`. Require `.vast-train-local/` to exist.
2. Read every `.vast-train-local/toy-act-*/setup.env` as plain `KEY=value`
   records. Do not `source` or `eval` these files because values such as
   `OFFER_CPU` can contain spaces. Use an exact-key reader that returns the text
   after the first `=`. Require `INSTANCE_ID`, `INSTANCE_LABEL`, `GIT_BRANCH`,
   `HOST`, `PORT`, `RUN_DIR`, `INDEX`, `TB_SESSION`, `TB_PORT`, and `TB_URL` for
   a candidate, and retain only candidates with `GIT_BRANCH=act-v2`.
3. Run `vastai show instances --raw` once and require successful, valid JSON
   array output. Match candidates by both numeric instance ID and exact label,
   and retain only records whose `actual_status` is `running`.
4. If `$ARGUMENTS` is non-empty, select the one active candidate whose instance
   ID, instance label, or run-state directory basename exactly matches it. If it
   matches zero or multiple candidates, stop with a concise diagnostic.
5. Without a selector, automatically select the only active candidate. If none
   exists, report that there is no active recorded training instance. If several
   exist, use the interactive question tool to ask which exact instance to use;
   show the instance ID, label, run name when recorded, and TensorBoard port.

Treat `.vast-train-local/*/instance.json` as sensitive and never print it. Never
print `VAST_API_KEY`, AWS credentials, or SSH private-key material.

## SSH Endpoint

1. Resolve the current endpoint with `vastai ssh-url "$INSTANCE_ID"`; do not
   assume the recorded host and port are still current. Parse and validate the
   returned `root@HOST:PORT` endpoint.
2. Use `$RUN_DIR/known_hosts` with `StrictHostKeyChecking=yes` for every SSH
   command. If the current endpoint already has an entry, keep it unchanged. If
   the endpoint changed and has no entry, collect its keys into a mode-600
   temporary file with `ssh-keyscan`, require non-empty output, then append those
   keys to the run-specific `known_hosts` under
   `/tmp/toy-act-local-wrapper.lock` `flock`, set the file mode to 600, and
   remove the temporary file. Never replace an existing entry after a host-key
   mismatch; stop and report the mismatch instead.
3. Use `BatchMode=yes`, `ConnectTimeout=15`, `ServerAliveInterval=30`, and
   `ServerAliveCountMax=3`. Confirm SSH succeeds before changing either
   TensorBoard session.

## Remote TensorBoard

Check the exact remote tmux session `tensorboard` and
`http://127.0.0.1:6006/`:

- If both the session and endpoint are healthy, leave them untouched.
- If the session exists but the endpoint remains unavailable for three checks
  five seconds apart, capture concise tmux output for diagnostics, kill only the
  `tensorboard` session, and recreate it.
- If the session is absent, create it.

Use the same command as the training workflow:

```bash
tmux new-session -d -s tensorboard \
  'cd /workspace/toy-act && exec /root/.local/bin/uv run --frozen \
   --only-group train python -m tensorboard.main \
   --logdir runs/act_v2 --host 127.0.0.1 --port 6006'
```

After creating it, poll remotely up to 12 times at five-second intervals. Stop
and report captured tmux output if the session exits or the endpoint never
responds.

## Local Forward

1. Reuse the recorded `INDEX`, `TB_SESSION`, `TB_PORT`, and `TB_URL`; do not
   allocate a new workflow index or port.
2. Under `/tmp/toy-act-local-wrapper.lock` `flock`, verify
   `/tmp/toy-act-local-wrapper-$INDEX.owner`. If present, it must contain exactly
   `toy-act-$INSTANCE_ID`. If absent, recreate it atomically with that value. A
   conflicting owner is an error.
3. If the exact local tmux session exists and `$TB_URL` responds, make no
   changes and report that forwarding is already healthy.
4. If that session exists but is unhealthy, capture concise pane output and kill
   only that exact session. Wait briefly for its socket to close.
5. Before starting the forward, use `ss` to require that `TB_PORT` is free. If
   another process owns it, stop and report the owner; never kill an unrelated
   process.
6. Create the recorded tmux session with a one-shot SSH forward:

```bash
tmux new-session -d -s "$TB_SESSION" \
  "exec ssh -o UserKnownHostsFile=$RUN_DIR/known_hosts \
   -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15 \
   -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
   -o ServerAliveCountMax=3 -N \
   -L $TB_PORT:127.0.0.1:6006 -p $PORT root@$HOST"
```

Substitute the freshly resolved `HOST` and `PORT`. Poll `$TB_URL` up to 12 times
at five-second intervals and require the tmux session to remain alive. On
failure, report concise pane output and leave training untouched.

## Result

Report the selected instance ID and label, whether remote TensorBoard was reused
or restarted, whether the local forward was reused or restarted, the tmux
session name, and the verified TensorBoard URL.

The forwarding session is intentionally one-shot. It exits when SSH disconnects
or the instance is destroyed; rerun `/resume-tb-forwarding` after a later
suspend or network interruption. This command must never call `vastai create`,
`vastai stop`, or `vastai destroy`.
