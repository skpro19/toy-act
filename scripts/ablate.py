"""Generate ablation configs from a grid spec.

Each ablation group lives in a folder with a ``BASE.toml``; a spec expands a
grid of overrides against that base. Each combination's run name is the full
compact encoding of its effective config, so the name reflects every training
parameter used.

Spec example:

    description = "batch size and seed sweep on ph"
    group = "configs/train/act_v2/BS-32"

    [fixed]
    dataset = "datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand_trimmed.hdf5"

    [grid]
    batch_size = [8, 32, 64]
    seed = [0, 420]

Usage:
    uv run python -m scripts.ablate --spec configs/sweep/my-sweep.toml
    uv run python -m scripts.ablate --spec configs/sweep/my-sweep.toml --resolve-dir DIR
    uv run python -m scripts.ablate --spec configs/sweep/my-sweep.toml --materialize
    uv run python -m scripts.ablate --spec configs/sweep/my-sweep.toml --exec --steps 10
"""

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import tomli_w
import tomllib

from scripts.train_v2 import (
    deep_merge,
    make_run_slug,
    read_config_file,
    validate_config,
)

SPEC_KEYS = frozenset({"description", "group", "fixed", "grid"})
BASE_FILENAME = "BASE.toml"
TRAIN_COMMAND = ("uv", "run", "python", "-m", "scripts.train_v2")


def load_spec(*, path: Path) -> dict:
    with path.open("rb") as file:
        spec = tomllib.load(file)

    unknown = sorted(set(spec) - SPEC_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: unknown spec keys {unknown}; allowed keys: {sorted(SPEC_KEYS)}"
        )

    group = spec.get("group")
    if not isinstance(group, str) or not group:
        raise ValueError(f"{path}: group must be a non-empty string")

    grid = spec.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError(f"{path}: grid must be a non-empty table of lists")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: grid.{key} must be a non-empty list")

    fixed = spec.get("fixed", {})
    if not isinstance(fixed, dict):
        raise ValueError(f"{path}: fixed must be a table")

    return spec


def expand_grid(*, grid: dict) -> list[dict]:
    keys = sorted(grid)
    combinations = itertools.product(*(grid[key] for key in keys))
    return [dict(zip(keys, values)) for values in combinations]


def build_plan(*, spec: dict, group_dir: Path, base_config: dict) -> list[dict]:
    fixed = spec.get("fixed", {})
    plan: list[dict] = []
    seen: dict[str, dict] = {}
    for combo in expand_grid(grid=spec["grid"]):
        overrides = deep_merge(base=dict(fixed), overrides=combo)
        effective = deep_merge(base=base_config, overrides=overrides)
        try:
            validate_config(config=effective)
        except (KeyError, ValueError) as error:
            raise ValueError(f"config for {combo} is invalid: {error}") from error
        slug = make_run_slug(config=effective)
        if slug in seen:
            raise ValueError(
                f"duplicate run name {slug!r} for combos {seen[slug]} and {combo}"
            )
        seen[slug] = combo
        plan.append(
            {
                "slug": slug,
                "overrides": overrides,
                "effective": effective,
                "path": group_dir / f"{slug}.toml",
            }
        )
    return plan


def render_delta(*, spec: dict, slug: str, overrides: dict) -> str:
    description = spec.get("description") or spec["group"]
    payload = {
        "description": f"{description} | {slug}",
        "base_config": BASE_FILENAME,
        "overrides": overrides,
    }
    return tomli_w.dumps(payload)


def print_plan(*, plan: list[dict]) -> None:
    for entry in plan:
        overrides = " ".join(
            f"{key}={value!r}" for key, value in sorted(entry["overrides"].items())
        )
        print(f"{entry['path'].name}  {overrides}")


def materialize(*, plan: list[dict], spec: dict, force: bool) -> list[Path]:
    written: list[Path] = []
    for entry in plan:
        path = entry["path"]
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass --force to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_delta(spec=spec, slug=entry["slug"], overrides=entry["overrides"])
        )
        written.append(path)
    return written


def write_resolved(*, plan: list[dict], resolve_dir: Path) -> list[Path]:
    resolve_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for entry in plan:
        config = dict(entry["effective"])
        config["name"] = entry["slug"]
        path = resolve_dir / f"{entry['slug']}.toml"
        path.write_text(tomli_w.dumps(config))
        written.append(path)
    return written


def run_deltas(*, plan: list[dict], steps: int | None) -> None:
    for entry in plan:
        command = [*TRAIN_COMMAND, "--config", str(entry["path"])]
        if steps is not None:
            command += ["--steps", str(steps)]
        print("+ " + " ".join(command))
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="path to the ablation spec TOML",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="write one delta TOML per grid combination",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing delta files",
    )
    parser.add_argument(
        "--exec",
        action="store_true",
        help="run each delta with scripts.train_v2 after writing",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="training-step override passed to scripts.train_v2 with --exec",
    )
    parser.add_argument(
        "--resolve-dir",
        type=Path,
        help="write one resolved self-contained TOML per combo into this directory",
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="write the ordered resolved config paths as JSON (requires --resolve-dir)",
    )
    args = parser.parse_args()
    if args.manifest is not None and args.resolve_dir is None:
        parser.error("--manifest requires --resolve-dir")
    return args


def main() -> None:
    args = parse_args()

    spec = load_spec(path=args.spec)
    group_dir = Path(spec["group"])
    base_config = read_config_file(path=group_dir / BASE_FILENAME)
    plan = build_plan(spec=spec, group_dir=group_dir, base_config=base_config)

    if args.materialize or args.exec:
        for path in materialize(plan=plan, spec=spec, force=args.force):
            print(f"wrote {path}")

    if args.resolve_dir is not None:
        resolved = write_resolved(plan=plan, resolve_dir=args.resolve_dir)
        for path in resolved:
            print(f"resolved {path}")
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps([str(path.resolve()) for path in resolved], indent=2) + "\n")

    if not (args.materialize or args.exec or args.resolve_dir):
        print_plan(plan=plan)

    if args.exec:
        run_deltas(plan=plan, steps=args.steps)


if __name__ == "__main__":
    main()
