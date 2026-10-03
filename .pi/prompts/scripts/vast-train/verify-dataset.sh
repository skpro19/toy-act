#!/usr/bin/env bash
set -euo pipefail
cd /workspace/toy-act
/root/.local/bin/uv run --frozen --only-group train python - <<'PY'
import hashlib
import os
from pathlib import Path
import h5py
from scripts.train_v2 import load_config
from scripts.rollout import camera_names_from_image_keys
config = load_config(path=Path('.vast-train/train-config.toml'))
camera_names_from_image_keys(image_keys=tuple(config['image_keys']))
path = Path(config['dataset'])
assert path.stat().st_size == int(os.environ['EXPECTED_BYTES']), 'Dataset byte size mismatch'
if os.environ.get('EXPECTED_SHA'):
    checksum = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            checksum.update(block)
    assert checksum.hexdigest() == os.environ['EXPECTED_SHA'], 'Dataset SHA-256 mismatch'
with h5py.File(path, 'r') as dataset:
    demos = list(dataset['data'].keys())
    assert demos, 'Empty dataset'
    for demo in demos:
        for key in config['image_keys']:
            assert key in dataset['data'][demo]['obs'], f'Missing {key} in {demo}'
print('Dataset size/hash/cameras verified')
PY
