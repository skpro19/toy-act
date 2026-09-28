---
description: Derive a new ACT v2 config from a base config with key=value overrides
agent: build
---

Create a new ACT v2 training config by loading an existing config under
`configs/`, applying `key=value` overrides, and writing a newly named config
next to the source. The source config is never modified.

## Arguments

- `$1` is the path to an existing `.toml` config (e.g.
  `configs/act_v2_bs250_l1.toml`).
- `$2 $3 ... $N` are one or more `key=value` overrides (e.g. `beta=0.5`,
  `epochs=200`, `use_z=false`, `action_loss=l2`). At least one override is
  required.

Supported keys and their value types:

| Key | Type | Notes |
|---|---|---|
| `action_loss` | string | `l1` or `l2` |
| `batch_size` | int | |
| `epochs` | int | |
| `lr` | float | |
| `seed` | int | |
| `beta` | float | |
| `beta_start` | float | may be added when absent |
| `beta_warmup_epochs` | int | may be added when absent |
| `checkpoint_every` | int | |
| `use_z` | bool | `true`/`false` |

An override key that is not already in the source config is added; a key that
is present is updated. Every other key keeps its exact source value and literal
formatting.

## Validation

Before writing anything, validate and stop (without touching the source file)
on any failure:

1. `$1` exists and is a `.toml` file.
2. Every override is `key=value`; a missing `=` is an error.
3. Every override key is in the supported-key table above; an unknown key is an
   error.
4. Each override value casts to the key's type; a failed cast is an error.
5. The merged config satisfies the `scripts/train_v2.py` `load_config` rules:
   `action_loss` is `l1` or `l2`, `beta_warmup_epochs >= 0`,
   `beta_start >= 0.0`.
6. The generated target path does not already exist; if it does, stop and tell
   the user rather than overwriting.

## Naming

The new file is named `act_v2[_<mode>]_...toml` by regenerating the name from
the merged config. It lives in the same directory as the source.

- `<mode>` is preserved from the source filename: the first `_`-separated
  segment after `act_v2` when that segment does not start with `bs` (for
  example `instance`, `local`, `smoke`). Otherwise there is no mode segment.

Fields are appended in this fixed order; every field is always included and no
field is omitted for matching its default. `seed`, `checkpoint_every`,
`beta_warmup_epochs`, and `beta_start` are intentionally omitted, matching the
run-name scheme in `scripts/train_v2.py` `RUN_NAME_FIELDS`:

| Segment | Format | Inclusion |
|---|---|---|
| `bs<batch_size>` | integer | always included |
| `beta<beta>` | `{:g}` with `.` -> `p` | always included |
| `ep<epochs>` | integer | always included |
| `noz` | literal | included only when `use_z` is false |
| `<action_loss>` | literal | always included |
| `lr<lr>` | scientific, `.` -> `p`, normalized exponent | always included |

Examples:

| Source | Overrides | New name |
|---|---|---|
| `act_v2_bs250_l1.toml` | `beta=0.5` | `act_v2_bs250_beta0p5_ep1000_l1_lr1e-4.toml` |
| `act_v2_instance_bs250_beta0p1_ep400_wu80_l1.toml` | `beta=0.5` | `act_v2_instance_bs250_beta0p5_ep400_l1_lr1e-4.toml` |
| `act_v2_bs250_l1.toml` | `epochs=200 use_z=false action_loss=l2` | `act_v2_bs250_beta10_ep200_noz_l2_lr1e-4.toml` |
| `act_v2_bs250_l1.toml` | `lr=1e-5` | `act_v2_bs250_beta10_ep1000_l1_lr1e-5.toml` |

## Workflow

1. Read the source config and confirm the requested overrides and the derived
   new name with the user before writing, unless the intent is unambiguous.
2. Run the reference implementation below with `$1` as the config path and
   every override as a following argument. It loads the source with `tomllib`,
   applies the overrides, validates, derives the name, and writes the new file.

```bash
uv run python - "$@" <<'PY'
import sys
import tomllib
from pathlib import Path

KEY_ORDER = (
    "action_loss",
    "batch_size",
    "epochs",
    "lr",
    "seed",
    "beta",
    "beta_start",
    "beta_warmup_epochs",
    "checkpoint_every",
    "use_z",
)

TYPES = {
    "action_loss": "str",
    "batch_size": "int",
    "epochs": "int",
    "lr": "float",
    "seed": "int",
    "beta": "float",
    "beta_start": "float",
    "beta_warmup_epochs": "int",
    "checkpoint_every": "int",
    "use_z": "bool",
}

ACTION_LOSS_CHOICES = {"l1", "l2"}


def read_literals(path):
    literals = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        literals[key.strip()] = value.strip()
    return literals


def cast_value(key, raw):
    kind = TYPES[key]
    if kind == "str":
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            return raw[1:-1]
        return raw
    if kind == "bool":
        lowered = raw.lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
        raise ValueError(f"{key} expects a boolean, got {raw!r}")
    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{key} expects an integer, got {raw!r}")
    if kind == "float":
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"{key} expects a float, got {raw!r}")
    raise AssertionError(f"unhandled type {kind}")


def slug_number(value):
    return f"{value:g}".replace(".", "p")


def slug_lr(value):
    mantissa, exponent = f"{value:.0e}".split("e")
    return mantissa.replace(".", "p") + "e" + str(int(exponent))


def make_config_name(base, config):
    segments = base.stem.split("_")
    mode = segments[2] if len(segments) > 2 and not segments[2].startswith("bs") else ""
    batch = int(config["batch_size"])
    beta = float(config["beta"])
    epochs = int(config["epochs"])
    use_z = bool(config.get("use_z", True))
    lr = float(config["lr"])
    action_loss = config["action_loss"]

    parts = ["act", "v2"]
    if mode:
        parts.append(mode)
    parts.append(f"bs{batch}")
    parts.append(f"beta{slug_number(beta)}")
    parts.append(f"ep{epochs}")
    if not use_z:
        parts.append("noz")
    parts.append(action_loss)
    parts.append(f"lr{slug_lr(lr)}")
    return "_".join(parts) + ".toml"


def fmt_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return f"{value:.1f}"
        return repr(value)
    if isinstance(value, str):
        return f'"{value}"'
    raise TypeError(f"unsupported value type {type(value)!r}")


def serialize(config, literals, overrides):
    lines = []
    for key in KEY_ORDER:
        if key not in config:
            continue
        if key in overrides:
            lines.append(f"{key} = {fmt_value(config[key])}")
        else:
            lines.append(f"{key} = {literals.get(key, fmt_value(config[key]))}")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 3:
        print("usage: new-config <base.toml> key=value [...]", file=sys.stderr)
        return 2
    base = Path(sys.argv[1])
    if not base.is_file():
        print(f"ERROR: config not found: {base}", file=sys.stderr)
        return 1
    with base.open("rb") as file:
        config = tomllib.load(file)
    literals = read_literals(base)

    overrides = {}
    for pair in sys.argv[2:]:
        if "=" not in pair:
            print(f"ERROR: expected key=value, got {pair!r}", file=sys.stderr)
            return 1
        key, _, raw = pair.partition("=")
        if key not in TYPES:
            print(f"ERROR: unknown config key {key!r}", file=sys.stderr)
            return 1
        try:
            overrides[key] = cast_value(key, raw)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    before = {key: config.get(key) for key in overrides}
    config.update(overrides)

    if config["action_loss"] not in ACTION_LOSS_CHOICES:
        print(f"ERROR: action_loss must be one of l1, l2, got {config['action_loss']!r}", file=sys.stderr)
        return 1
    if config.get("beta_warmup_epochs", 0) < 0:
        print("ERROR: beta_warmup_epochs must be >= 0", file=sys.stderr)
        return 1
    if config.get("beta_start", 0.0) < 0.0:
        print("ERROR: beta_start must be >= 0.0", file=sys.stderr)
        return 1

    new_name = make_config_name(base, config)
    out = base.with_name(new_name)
    if out.exists():
        print(f"ERROR: target already exists: {out}", file=sys.stderr)
        return 1
    out.write_text(serialize(config, literals, overrides))
    print(f"created {out}")
    for key in KEY_ORDER:
        if key in overrides:
            print(f"  {key}: {before[key]!r} -> {config[key]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
```

3. After the script succeeds, show the user the created path and a summary of
   each changed key (`old -> new`). Do not modify the source config.
