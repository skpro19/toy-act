"""Resolve a training config (with inheritance) into a self-contained TOML.

Usage:
    uv run python -m scripts.resolve_config \
        --config configs/train/act_v2/BS-32/bs32_z0_s0.toml \
        --out .vast-train-local/resolved/bs32_z0_s0.toml \
        --name bs32_z0_s0
"""

import argparse
from pathlib import Path

import tomli_w

from scripts.train_v2 import read_config_file, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="compact run slug injected into the resolved config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = read_config_file(path=args.config)
    config = validate_config(config=config)
    config["name"] = args.name or args.config.stem

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tomli_w.dumps(config))
    print(args.out)


if __name__ == "__main__":
    main()
