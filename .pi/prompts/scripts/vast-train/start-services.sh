#!/usr/bin/env bash
# Idempotent launch: never overwrite markers or restart a terminal runner.
set -euo pipefail
cd /workspace/toy-act
CONTROL=/workspace/toy-act/.vast-train
HELPERS=/workspace/toy-act/.pi/prompts/scripts/vast-train
if [ ! -e "$CONTROL/state/completed" ] && [ ! -e "$CONTROL/state/failed" ]; then
  if ! tmux has-session -t tensorboard 2>/dev/null; then
    tmux new-session -d -s tensorboard \
      'cd /workspace/toy-act && exec /root/.local/bin/uv run --frozen --only-group train python -m tensorboard.main --logdir runs/act_v2 --host 127.0.0.1 --port 6006'
  fi
  ready=no
  for _ in $(seq 1 12); do
    if curl --silent --fail --max-time 3 --output /dev/null http://127.0.0.1:6006/; then ready=yes; break; fi
    tmux has-session -t tensorboard
    sleep 5
  done
  test "$ready" = yes
  if ! tmux has-session -t train 2>/dev/null; then
    test ! -e "$CONTROL/state/run-status" || { echo 'Missing train session without terminal marker; refusing restart' >&2; exit 2; }
    tmux new-session -d -s train -e TRAIN_MODULE=scripts.train_v2 \
      -e TRAIN_CONFIG="$CONTROL/train-config.toml" "bash $HELPERS/runner.sh"
  fi
fi
# Backup also needs to run when training terminated quickly.
if [ ! -e "$CONTROL/state/backup-final-succeeded" ] && ! tmux has-session -t ckpt-bkp 2>/dev/null; then
  test ! -e "$CONTROL/state/backup-running" || { echo 'Backup exited; refusing silent restart' >&2; exit 2; }
  tmux new-session -d -s ckpt-bkp -e CHECKPOINT_ROOT=checkpoints/act_v2 \
    -e RUNS_ROOT=runs/act_v2 "bash $HELPERS/ckpt-bkp-wrapper.sh"
fi
