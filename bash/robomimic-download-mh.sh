#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# Download the robomimic CAN multi-human (MH) datasets needed by
# bash/robomimic-quality-subsets.sh.
#
# The MH dataset is the only CAN source with the per-operator "better" / "okay" /
# "worse" proficiency labels, stored as filter keys in the hdf5 mask group.
#
# Usage:
#   bash/robomimic-download-mh.sh            # raw + low_dim
#   HDF5_TYPES=raw bash/robomimic-download-mh.sh
#
# Notes:
#   * "raw" (demo_v15.hdf5) has no observations and is what the image variants are
#     rendered from; "low_dim" is what the quality-subset script consumes.
#   * The "image" MH variant is not published and is generated locally by rendering
#     the raw dataset's simulator states.

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TASK="${TASK:-can}"
readonly DATASET_TYPE="${DATASET_TYPE:-mh}"
readonly HDF5_TYPES="${HDF5_TYPES:-raw low_dim}"

echo "Task:         ${TASK}"
echo "Dataset type: ${DATASET_TYPE}"
echo "Hdf5 types:   ${HDF5_TYPES}"
echo ""

cd "$PROJECT_DIR"
uv run python - "$PROJECT_DIR" "$TASK" "$DATASET_TYPE" $HDF5_TYPES <<'PY'
import os
import sys

import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY, HF_REPO_ID

project_dir = sys.argv[1]
task = sys.argv[2]
dataset_type = sys.argv[3]
hdf5_types = sys.argv[4:]

# Always download into the project's datasets directory. Deriving this from
# robomimic.__path__ would land inside site-packages when robomimic is installed
# into the venv rather than used from third_party/.
download_dir = os.path.join(project_dir, "datasets", task, dataset_type)
os.makedirs(download_dir, exist_ok=True)

for hdf5_type in hdf5_types:
    entry = DATASET_REGISTRY[task][dataset_type][hdf5_type]
    url = entry["url"]
    if url is None:
        print(
            f"Skipping {task}-{dataset_type}-{hdf5_type}: no published download URL. "
            "Generate it locally by rendering the raw dataset."
        )
        continue

    print(f"Downloading {task}/{dataset_type}/{hdf5_type} -> {download_dir}")
    # @download_file_from_hf prompts on overwrite, so decide here instead to keep the
    # script non-interactive and to avoid re-downloading a complete file.
    destination = os.path.join(download_dir, os.path.basename(url))
    if os.path.exists(destination):
        print(f"Already present, skipping: {destination}")
        print("")
        continue

    FileUtils.download_file_from_hf(
        repo_id=HF_REPO_ID,
        filename=url,
        download_dir=download_dir,
        check_overwrite=False,
    )
    print("")
PY

echo "Done. Datasets in ${PROJECT_DIR}/datasets/${TASK}/${DATASET_TYPE}/"
