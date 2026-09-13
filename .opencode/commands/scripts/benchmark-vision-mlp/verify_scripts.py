import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from common import write_json


def verify_scripts(*, tools_dir: Path, manifest: Path) -> None:
    expected = json.loads(manifest.read_text())
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(tools_dir.iterdir())
        if path.is_file()
    }
    if actual != expected:
        raise ValueError(f"script digest mismatch: {actual!r} != {expected!r}")


def git_manifest(*, project_root: Path, commit: str, source_prefix: str) -> dict[str, str]:
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, source_prefix],
        cwd=project_root,
        text=True,
    ).splitlines()
    if not paths:
        raise ValueError("selected commit contains no benchmark scripts")
    names = [Path(path).name for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("benchmark script basenames are not unique")
    return {
        name: hashlib.sha256(
            subprocess.check_output(
                ["git", "show", f"{commit}:{path}"], cwd=project_root
            )
        ).hexdigest()
        for name, path in zip(names, paths, strict=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--source-prefix")
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()

    git_arguments = (
        args.project_root,
        args.commit,
        args.source_prefix,
        args.write_manifest,
    )
    if all(value is not None for value in git_arguments):
        expected = git_manifest(
            project_root=args.project_root,
            commit=args.commit,
            source_prefix=args.source_prefix,
        )
        write_json(path=args.write_manifest, value=expected)
        verify_scripts(tools_dir=args.tools_dir, manifest=args.write_manifest)
    elif args.manifest is not None and all(value is None for value in git_arguments):
        verify_scripts(tools_dir=args.tools_dir, manifest=args.manifest)
    else:
        parser.error("use --manifest or all Git-manifest arguments")


if __name__ == "__main__":
    main()
