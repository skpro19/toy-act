import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import glfw
import mujoco
import torch

from common import write_json


def output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    manifest = json.loads(args.manifest.read_text())
    contract["dataset_digest"] = manifest["aggregate_sha256"]
    contract["dataset_frames"] = manifest["total_frames"]
    contract["software"] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "mujoco": mujoco.__version__,
        "glfw": glfw.__version__,
        "kernel": platform.release(),
        "nvidia_driver": output(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "uv": output("/root/.local/bin/uv", "--version"),
        "uv_lock_sha256": hashlib.sha256(
            (args.output.parents[2] / "uv.lock").read_bytes()
        ).hexdigest(),
        "resolved_packages": output(
            "/root/.local/bin/uv",
            "--directory",
            str(args.output.parents[2]),
            "pip",
            "freeze",
        ).splitlines(),
        "torch_backends": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }
    governors = {
        path.read_text().strip()
        for path in Path("/sys/devices/system/cpu").glob(
            "cpu[0-9]*/cpufreq/scaling_governor"
        )
        if path.is_file()
    }
    contract["software"]["cpu_governors"] = sorted(governors)
    contract["plans"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((args.output.parent / "plans").glob("*.plan"))
    }
    write_json(
        path=args.output,
        value=contract,
    )


if __name__ == "__main__":
    main()
