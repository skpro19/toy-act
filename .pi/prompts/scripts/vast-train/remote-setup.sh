#!/usr/bin/env bash
# Idempotent setup for a provisional instance. GIT_COMMIT is provided on stdin.
# No cleanup traps; local workflow state owns recovery after interruption.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0
export DEBIAN_FRONTEND=noninteractive
PROJECT=/workspace/toy-act
if [ ! -d "$PROJECT/.git" ]; then
  test ! -e "$PROJECT" || { echo 'Existing non-clone project directory; refusing replacement' >&2; exit 2; }
  git clone --branch act-v2 --single-branch https://github.com/skpro19/toy-act.git "$PROJECT"
fi
# Never change code after a training launch.
if [ -e "$PROJECT/.vast-train/state/run-status" ] || tmux has-session -t train 2>/dev/null; then
  echo 'Training already launched; setup cannot replace its inputs' >&2
  exit 2
fi
git -C "$PROJECT" fetch --no-tags origin "$GIT_COMMIT"
git -C "$PROJECT" checkout --detach "$GIT_COMMIT"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$GIT_COMMIT"
mkdir -p "$PROJECT/.vast-train/"{logs,state}
if [ ! -x /root/.local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
apt-get update -qq
apt-get install -y -qq awscli tmux build-essential libgl1 libglib2.0-0 libegl1 libgles2 libglfw3
cd "$PROJECT"
/root/.local/bin/uv sync --frozen --only-group train
MUJOCO_GL=egl /root/.local/bin/uv run --frozen --only-group train python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'RTX 4090' in torch.cuda.get_device_name(0)
import robosuite, robomimic, mujoco
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=84, width=84)
renderer.update_scene(data)
frame = renderer.render()
renderer.close()
assert frame.shape == (84, 84, 3) and str(frame.dtype) == 'uint8'
print('CUDA/imports/EGL verified')
PY
