"""Training script for ACT v2 (CVAE + action chunking)."""

import argparse
import faulthandler
import json
import random
import re
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scripts.dataset import CanPhDataset, NormalizationStats
from scripts.models.act_v2.config import (
    ACTION_CHUNK_SIZE,
    D_MODEL,
    IMG_DIMS,
    JOINT_DIMS,
    N_HEAD,
    NUM_LAYERS,
    PROPRIO_DIMS,
    Z_DIMS,
)
from scripts.models.act_v2.model import ACTV2
from scripts.rollout import (
    camera_names_from_image_keys,
    close_env,
    configure_renderer,
    create_rollout_env,
    make_rollout_env_meta,
    run_rollout,
    summarize_rollouts,
)
from scripts.train_v1 import (
    compute_adam_moment_norms,
    compute_global_l2_norm,
    compute_global_update_norm,
    make_dataloader_generator,
    make_dataloader_worker_init_fn,
    seed_everything,
    snapshot_parameters,
)

RUNS_ROOT = Path("runs/act_v2")
CHECKPOINTS_ROOT = Path("checkpoints/act_v2")
SENSITIVITY_PROBE_SIZE = 16
CONFIG_VERSION = "v4"
EVAL_EPISODES = 30
EVAL_HORIZON = 250
EVAL_SEED = 0
DATALOADER_NUM_WORKERS = 4
DATALOADER_TIMEOUT_SECONDS = 120

CONFIG_KEYS = (
    "action_loss",
    "batch_size",
    "steps",
    "image_keys",
    "dataset",
    "lr",
    "seed",
    "beta",
    "checkpoint_every",
    "tensorboard",
    "version",
)

TENSORBOARD_NAMESPACES = (
    "activations",
    "batch_metrics",
    "denorm_l1",
    "epoch_metrics",
    "eval",
    "hyperparams",
    "latent",
    "lr",
    "optimizer",
    "ranges",
    "sensitivity",
    "throughput",
    "timing",
)

ACTION_LOSS_CHOICES = frozenset({"l1", "l2"})

ACTIVATION_HOOK_TAGS = (
    "image_encoder",
    "cvae_encoder",
    "transformer_encoder",
    "transformer_decoder",
    "action_head",
)

RUN_NAME_FIELDS = (
    ("batch_size", "bs", "{:d}"),
    ("lr", "lr", "{:.0e}"),
    ("beta", "beta", "{:g}"),
    ("beta_warmup_steps", "wu", "{:d}"),
    ("steps", "st", "{:d}"),
    ("image_keys", "img_", "{:s}"),
    ("use_z", "z", "{:d}"),
    ("dataset", "ds_", "{:s}"),
    ("seed", "s", "{:d}"),
    ("action_loss", "", "{:s}"),
)

RUN_NAME_OMIT_KEYS = frozenset({
    "checkpoint_every",
    "beta_start",
    "tensorboard",
    "version",
    "base_config",
    "description",
    "name",
})

CAMERA_TAGS = {
    "agentview_image": "av",
    "robot0_eye_in_hand_image": "eih",
}

DATASET_CAMERA_TAGS = {
    "agentview": "av",
    "robot0_eye_in_hand": "eih",
}

DATASET_TIMESTAMP_PREFIX = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_"
)


def validate_config(*, config: dict) -> dict:
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"config missing required keys: {missing}")

    version = config["version"]
    if version != CONFIG_VERSION:
        raise ValueError(
            f"config version must be {CONFIG_VERSION!r}, got {version!r}"
        )

    config.setdefault("beta_start", 0.0)
    config.setdefault("beta_warmup_steps", 0)
    config.setdefault("use_z", True)

    dataset = config["dataset"]
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset must be a non-empty string")

    if config["steps"] <= 0:
        raise ValueError(f"steps must be > 0, got {config['steps']}")
    if config["checkpoint_every"] <= 0:
        raise ValueError(
            f"checkpoint_every must be > 0, got {config['checkpoint_every']}"
        )
    if config["beta_warmup_steps"] < 0:
        raise ValueError(
            f"beta_warmup_steps must be >= 0, got {config['beta_warmup_steps']}"
        )
    if config["beta_start"] < 0.0:
        raise ValueError(f"beta_start must be >= 0, got {config['beta_start']}")

    action_loss = config["action_loss"]
    if action_loss not in ACTION_LOSS_CHOICES:
        allowed = ", ".join(sorted(ACTION_LOSS_CHOICES))
        raise ValueError(f"action_loss must be one of {{{allowed}}}, got {action_loss!r}")

    image_keys = config["image_keys"]
    if not isinstance(image_keys, list) or len(image_keys) == 0:
        raise ValueError("image_keys must be a non-empty list of HDF5 obs key names")
    if not all(isinstance(key, str) and key for key in image_keys):
        raise ValueError("image_keys must contain only non-empty strings")

    config["tensorboard"] = validate_tensorboard_flags(
        tensorboard=config["tensorboard"],
    )

    return config


def validate_tensorboard_flags(*, tensorboard: dict) -> dict:
    if not isinstance(tensorboard, dict):
        raise ValueError("tensorboard must be a table of boolean namespace flags")

    unknown = sorted(set(tensorboard) - set(TENSORBOARD_NAMESPACES))
    if unknown:
        allowed = ", ".join(TENSORBOARD_NAMESPACES)
        raise ValueError(
            f"unknown tensorboard namespaces {unknown}; allowed namespaces: {allowed}"
        )

    flags = {}
    for namespace in TENSORBOARD_NAMESPACES:
        value = tensorboard.get(namespace, True)
        if not isinstance(value, bool):
            raise ValueError(
                f"tensorboard.{namespace} must be a boolean, got {value!r}"
            )
        flags[namespace] = value
    return flags


def deep_merge(*, base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(base=merged[key], overrides=value)
        else:
            merged[key] = value
    return merged


def read_config_file(*, path: Path) -> dict:
    return _read_config_file(path=path.resolve(), visited=frozenset())


def _read_config_file(*, path: Path, visited: frozenset[Path]) -> dict:
    if path in visited:
        raise ValueError(f"config inheritance cycle detected at {path}")
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("rb") as file:
        raw = tomllib.load(file)

    base_config = raw.pop("base_config", None)
    raw.pop("description", None)
    overrides = raw.pop("overrides", None)
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError(f"{path}: overrides must be a table")

    if base_config is None:
        return deep_merge(base=raw, overrides=overrides or {})

    if not isinstance(base_config, str) or not base_config:
        raise ValueError(f"{path}: base_config must be a non-empty string")
    base_path = (path.parent / base_config).resolve()
    base = _read_config_file(path=base_path, visited=visited | {path})
    merged = deep_merge(base=base, overrides=overrides or {})
    return deep_merge(base=merged, overrides=raw)


def load_config(*, path: Path) -> dict:
    return validate_config(config=read_config_file(path=path))


def config_uses_inheritance(*, path: Path) -> bool:
    with path.open("rb") as file:
        return "base_config" in tomllib.load(file)


def resolve_dataset(*, cli_dataset: Path | None, config: dict) -> Path:
    if cli_dataset is not None:
        return cli_dataset
    return Path(config["dataset"])


def make_action_loss_fn(*, action_loss: str) -> nn.Module:
    if action_loss == "l1":
        return nn.L1Loss(reduction="mean")
    if action_loss == "l2":
        return nn.MSELoss(reduction="mean")
    allowed = ", ".join(sorted(ACTION_LOSS_CHOICES))
    raise ValueError(f"action_loss must be one of {{{allowed}}}, got {action_loss!r}")


def beta_at_step(
    *,
    step: int,
    beta_start: float,
    beta: float,
    beta_warmup_steps: int) -> float:
    if beta_warmup_steps <= 0 or step >= beta_warmup_steps:
        return beta
    progress = step / beta_warmup_steps
    return beta_start + (beta - beta_start) * progress


def save_run_config(
    *,
    run_dir: Path,
    run_name: str,
    config: dict) -> None:
    payload = {"run_name": run_name, "config": config}
    path = run_dir / "config.json"
    with path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def sanitize_slug(*, text: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in text
    )


def format_config_value(
    *,
    value: object,
    template: str) -> str:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (list, tuple)):
        value = "-".join(str(item) for item in value)
    return sanitize_slug(text=template.format(value))


def format_image_keys(*, image_keys: list[str]) -> str:
    return "-".join(CAMERA_TAGS.get(key, key) for key in image_keys)


def format_dataset(*, dataset: str) -> str:
    path = Path(dataset)
    stem = DATASET_TIMESTAMP_PREFIX.sub("", path.stem)
    for token, tag in DATASET_CAMERA_TAGS.items():
        stem = stem.replace(token, tag)
    return f"{path.parent.name}_{stem}"


def render_config_fields(*, config: dict) -> list[str]:
    parts: list[str] = []
    handled_keys: set[str] = set(RUN_NAME_OMIT_KEYS)
    for key, label, template in RUN_NAME_FIELDS:
        if key not in config:
            continue
        handled_keys.add(key)
        if key == "image_keys":
            rendered = format_image_keys(image_keys=config[key])
        elif key == "dataset":
            rendered = format_dataset(dataset=config[key])
        else:
            rendered = format_config_value(value=config[key], template=template)
        parts.append(f"{label}{rendered}" if label else rendered)
    for key in sorted(key for key in config if key not in handled_keys):
        rendered = format_config_value(value=config[key], template="{}")
        parts.append(f"{key}-{rendered}")
    return parts


def make_run_slug(*, config: dict) -> str:
    return "_".join(render_config_fields(config=config))


def make_run_name(*, config: dict) -> str:
    slug = config.get("name") or make_run_slug(config=config)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{slug}"


def register_activation_norm_hooks(*, model: ACTV2) -> tuple[dict[str, float], list]:
    norms: dict[str, float] = {}
    hook_targets = {
        "image_encoder": model.image_encoder,
        "cvae_encoder": model.cvae_encoder,
        "transformer_encoder": model.transformer_encoder,
        "transformer_decoder": model.transformer_decoder,
        "action_head": model.action_head,
    }
    handles = []
    for tag, module in hook_targets.items():
        def make_hook(name: str):
            def hook(_module, _inputs, output) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                norms[name] = tensor.detach().float().norm().item()

            return hook

        handles.append(module.register_forward_hook(make_hook(tag)))
    return norms, handles


def get_kl_loss(
    *,
    mu: torch.Tensor,
    log_sigma_x2: torch.Tensor) -> torch.Tensor:
    _, _, d = mu.shape

    sum_mu_x2 = torch.sum(mu.pow(2), dim=2)
    sum_sigma_x2 = torch.sum(log_sigma_x2.exp(), dim=2)
    sum_log_sigma_x2 = torch.sum(log_sigma_x2, dim=2)

    kl_loss = 0.5 * (sum_mu_x2 + sum_sigma_x2 - d - sum_log_sigma_x2)

    return kl_loss.mean()


def measure_observation_sensitivity(
    *, model: ACTV2, images: torch.Tensor, proprio: torch.Tensor) -> tuple[float, float]:
    """Measure normalized action changes from swapping one observation modality."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            baseline = model.infer(proprio=proprio, img=images)
            swapped_images = model.infer(proprio=proprio, img=images.roll(1, dims=0))
            swapped_proprio = model.infer(proprio=proprio.roll(1, dims=0), img=images)
            image_l1 = (baseline - swapped_images).abs().mean().item()
            proprio_l1 = (baseline - swapped_proprio).abs().mean().item()
    finally:
        model.train(was_training)

    return image_l1, proprio_l1


def checkpoint_path_for_step(*, checkpoint_dir: Path, global_step: int) -> Path:
    return checkpoint_dir / f"step_{global_step:09d}.pt"


def save_checkpoint(
    *,
    path: Path,
    run_name: str,
    config: dict,
    dataset: str,
    epoch: int,
    global_step: int,
    model: ACTV2,
    loss_running_avg: float,
    normalization: NormalizationStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_name": run_name,
            "config": dict(config),
            "dataset": dataset,
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "loss_running_avg": loss_running_avg,
            "normalization": normalization.as_checkpoint_dict(),
        },
        path,
    )


def maybe_save_step_checkpoint(
    *,
    checkpoint_dir: Path,
    checkpoint_every: int,
    run_name: str,
    config: dict,
    dataset: str,
    epoch: int,
    global_step: int,
    model: ACTV2,
    loss_running_avg: float,
    normalization: NormalizationStats) -> Path | None:
    if global_step <= 0 or global_step % checkpoint_every != 0:
        return None
    snapshot_path = checkpoint_path_for_step(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
    )
    save_checkpoint(
        path=snapshot_path,
        run_name=run_name,
        config=config,
        dataset=dataset,
        epoch=epoch,
        global_step=global_step,
        model=model,
        loss_running_avg=loss_running_avg,
        normalization=normalization,
    )
    return snapshot_path


def evaluate_and_log_checkpoint(
    *,
    model: ACTV2,
    env: Any | None,
    dataset: Path,
    image_keys: tuple[str, ...],
    normalization: NormalizationStats,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    run_started: float,
    train_seconds: float,
    train_steps: int,
    train_samples: int,
    checkpoint_seconds: float,
    tensorboard: dict) -> Any:
    """Evaluate a saved model without changing the training RNG or model mode."""
    run_eval = tensorboard["eval"]
    summary = None
    rollout_seconds = 0.0
    env_steps = 0

    if run_eval:
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        was_training = model.training
        eval_started = time.monotonic()
        try:
            if env is None:
                configure_renderer(on_screen=False)
                camera_names = camera_names_from_image_keys(image_keys=image_keys)
                env_meta = make_rollout_env_meta(
                    dataset_path=dataset,
                    camera_names=camera_names,
                    camera_height=IMG_DIMS[0],
                    camera_width=IMG_DIMS[1],
                )
                env = create_rollout_env(env_meta=env_meta, on_screen=False, write_video=False)

            model.eval()
            rollouts = []
            with torch.inference_mode():
                for episode_idx in range(EVAL_EPISODES):
                    np.random.seed(EVAL_SEED + episode_idx)
                    rollouts.append(run_rollout(
                        model=model,
                        env=env,
                        device=device,
                        normalization=normalization,
                        image_keys=image_keys,
                        horizon=EVAL_HORIZON,
                        terminate_on_success=True,
                        render=False,
                        video_writer=None,
                        video_skip=1,
                    ))
        finally:
            model.train(was_training)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_states)

        rollout_seconds = time.monotonic() - eval_started
        summary = summarize_rollouts(rollouts=rollouts)
        env_steps = sum(int(rollout["horizon"]) for rollout in rollouts)

    if run_eval:
        writer.add_scalar("eval/success_rate", summary["success_rate"], global_step)
        writer.add_scalar("eval/return_mean", summary["return_mean"], global_step)
        writer.add_scalar("eval/horizon_mean", summary["horizon_mean"], global_step)

    if tensorboard["throughput"]:
        writer.add_scalar("throughput/train_steps_per_sec", train_steps / train_seconds, global_step)
        writer.add_scalar("throughput/train_samples_per_sec", train_samples / train_seconds, global_step)
        if run_eval:
            writer.add_scalar(
                "throughput/rollout_env_steps_per_sec", env_steps / rollout_seconds, global_step,
            )
            writer.add_scalar(
                "throughput/rollout_episodes_per_min", EVAL_EPISODES * 60 / rollout_seconds,
                global_step,
            )

    if tensorboard["timing"]:
        if run_eval:
            writer.add_scalar("timing/rollout_seconds", rollout_seconds, global_step)
        writer.add_scalar("timing/checkpoint_seconds", checkpoint_seconds, global_step)
        writer.add_scalar("timing/elapsed_hours", (time.monotonic() - run_started) / 3600, global_step)

    writer.flush()
    if run_eval:
        print(
            f"step {global_step}: rollout success={summary['num_success']}/{EVAL_EPISODES} "
            f"return={summary['return_mean']:.3f} elapsed={rollout_seconds:.1f}s"
        )
    return env


def train(
    *,
    config: dict,
    dataset: Path) -> None:
    seed = config["seed"]
    batch_size = config["batch_size"]
    lr = config["lr"]

    tensorboard = config["tensorboard"]
    log_activations = tensorboard["activations"]
    log_batch_metrics = tensorboard["batch_metrics"]
    log_denorm_l1 = tensorboard["denorm_l1"]
    log_epoch_metrics = tensorboard["epoch_metrics"]
    log_hyperparams = tensorboard["hyperparams"]
    log_latent = tensorboard["latent"]
    log_lr = tensorboard["lr"]
    log_optimizer = tensorboard["optimizer"]
    log_ranges = tensorboard["ranges"]
    log_sensitivity = tensorboard["sensitivity"]
    track_epoch_losses = log_batch_metrics or log_epoch_metrics

    seed_everything(seed=seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training but PyTorch could not initialize it. "
            "Verify the NVIDIA driver and CUDA_VISIBLE_DEVICES before rerunning."
        )
    device = torch.device("cuda")

    can_ph_dataset = CanPhDataset(
        file=str(dataset),
        image_keys=tuple(config["image_keys"]),
        k=ACTION_CHUNK_SIZE,
    )
    dataloader_generator = make_dataloader_generator(seed=seed)
    can_ph_dataloader = DataLoader(
        dataset=can_ph_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        timeout=DATALOADER_TIMEOUT_SECONDS,
        generator=dataloader_generator,
        worker_init_fn=make_dataloader_worker_init_fn(base_seed=seed),
    )

    model = ACTV2(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        z_dims=Z_DIMS,
        proprio_dims=PROPRIO_DIMS,
        action_chunk_size=ACTION_CHUNK_SIZE,
        use_z=config["use_z"],
    ).to(device=device)

    action_loss_kind = config["action_loss"]
    action_loss_fn = make_action_loss_fn(action_loss=action_loss_kind)
    denorm_l1_fn = nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    model_parameters = list(model.parameters())
    if log_activations:
        activation_norms, activation_hook_handles = register_activation_norm_hooks(model=model)
    else:
        activation_norms, activation_hook_handles = {}, []
    normalization = can_ph_dataset.normalization
    action_mean = torch.from_numpy(normalization.action_mean).to(device=device).view(1, 1, -1)
    action_std = torch.from_numpy(normalization.action_std).to(device=device).view(1, 1, -1)

    target_steps = config["steps"]
    global_step = 0

    beta = config["beta"]
    beta_start = config["beta_start"]
    beta_warmup_steps = config["beta_warmup_steps"]
    checkpoint_every = config["checkpoint_every"]

    run_name = make_run_name(config=config)
    dataset_path = str(dataset.resolve())
    run_dir = RUNS_ROOT / run_name
    checkpoint_dir = CHECKPOINTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir=run_dir, run_name=run_name, config=config)
    writer = SummaryWriter(log_dir=str(run_dir))
    run_started = time.monotonic()
    train_interval_started = run_started
    last_evaluated_step = 0
    train_samples = 0
    eval_env = None
    print(f"seed => {seed}")
    print(f"device => {device}")
    print(f"gpu => {torch.cuda.get_device_name(device)}")
    print(f"tensorboard logs => {run_dir.resolve()}")
    print(f"checkpoints => {checkpoint_dir.resolve()}")
    if beta_warmup_steps > 0:
        print(
            f"beta schedule => linear warmup from {beta_start:g} to {beta:g} "
            f"over steps 0–{beta_warmup_steps - 1}"
        )
    else:
        print(f"beta => {beta:g}")
    print(f"target_steps => {target_steps}")
    print(f"checkpoint_every => {checkpoint_every} steps")
    print(f"rollout evaluation => {EVAL_EPISODES} episodes, horizon {EVAL_HORIZON}")
    print(f"action_loss => {action_loss_kind}")
    print(f"use_z => {config['use_z']}")
    enabled_namespaces = [
        namespace for namespace in TENSORBOARD_NAMESPACES if tensorboard[namespace]
    ]
    print(f"tensorboard namespaces => {', '.join(enabled_namespaces)}")

    probe_images = None
    probe_proprio = None
    last_loss_running_avg = 0.0
    epoch = 0
    while global_step < target_steps:
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_weighted_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            can_ph_dataloader,
            desc=f"epoch {epoch + 1} step {global_step}/{target_steps}",
        )
        for batch_dict in pbar:
            beta_t = beta_at_step(
                step=global_step,
                beta_start=beta_start,
                beta=beta,
                beta_warmup_steps=beta_warmup_steps,
            )
            optimizer.zero_grad()

            img = batch_dict["images"].to(device, non_blocking=True)
            proprio = batch_dict["proprio"].to(device, non_blocking=True)
            actions = batch_dict["target_actions"].to(device, non_blocking=True)
            if log_sensitivity and probe_images is None:
                # Fix real observations from the first batch for every epoch.
                probe_images = img[:SENSITIVITY_PROBE_SIZE].detach().clone()
                probe_proprio = proprio[:SENSITIVITY_PROBE_SIZE].detach().clone()

            pred_actions, mu, log_sigma_x2 = model(proprio=proprio, actions=actions, img=img)

            action_loss = action_loss_fn(pred_actions, actions)
            kl_loss = get_kl_loss(mu=mu, log_sigma_x2=log_sigma_x2)
            loss = action_loss + beta_t * kl_loss

            loss.backward()
            if log_optimizer:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float("inf"),
                ).item()
                params_before = snapshot_parameters(parameters=model_parameters)
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            if track_epoch_losses:
                batch_action_loss = action_loss.item()
                batch_kl_loss = kl_loss.item()
                batch_weighted_kl_loss = beta_t * batch_kl_loss
                epoch_action_loss += batch_action_loss
                epoch_kl_loss += batch_kl_loss
                epoch_weighted_kl_loss += batch_weighted_kl_loss
            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

            with torch.no_grad():
                if log_lr:
                    learning_rate = optimizer.param_groups[0]["lr"]
                if log_optimizer:
                    adam_exp_avg_norm, adam_exp_avg_sq_norm = compute_adam_moment_norms(
                        optimizer=optimizer,
                    )
                    param_norm = compute_global_l2_norm(tensors=model_parameters)
                    update_norm = compute_global_update_norm(
                        parameters=model_parameters,
                        before=params_before,
                    )

                if log_denorm_l1 or log_ranges:
                    pred_physical = pred_actions.detach() * action_std + action_mean
                    target_physical = actions * action_std + action_mean
                    pred_joint = pred_physical[..., :JOINT_DIMS]
                    pred_gripper = pred_physical[..., JOINT_DIMS:]
                    target_joint = target_physical[..., :JOINT_DIMS]
                    target_gripper = target_physical[..., JOINT_DIMS:]

                if log_denorm_l1:
                    batch_l1_joint = denorm_l1_fn(pred_joint, target_joint).item()
                    batch_l1_gripper = denorm_l1_fn(pred_gripper, target_gripper).item()

                if log_ranges:
                    pred_joint_min = pred_joint.min().item()
                    pred_joint_max = pred_joint.max().item()
                    pred_gripper_min = pred_gripper.min().item()
                    pred_gripper_max = pred_gripper.max().item()
                    target_joint_min = target_joint.min().item()
                    target_joint_max = target_joint.max().item()
                    target_gripper_min = target_gripper.min().item()
                    target_gripper_max = target_gripper.max().item()

                if log_latent:
                    mu_norm = mu.detach().float().norm().item()
                    log_sigma_x2_mean = log_sigma_x2.detach().mean().item()
                    sigma_mean = log_sigma_x2.detach().exp().mean().item()

                if log_batch_metrics and batch_loss > 0.0:
                    batch_kl_fraction = batch_weighted_kl_loss / batch_loss
                else:
                    batch_kl_fraction = 0.0

            if log_batch_metrics:
                writer.add_scalar("batch_metrics/loss", batch_loss, global_step)
                writer.add_scalar("batch_metrics/action_loss", batch_action_loss, global_step)
                writer.add_scalar("batch_metrics/kl_loss", batch_kl_loss, global_step)
                writer.add_scalar("batch_metrics/weighted_kl_loss", batch_weighted_kl_loss, global_step)
                writer.add_scalar("batch_metrics/kl_fraction", batch_kl_fraction, global_step)
            if log_hyperparams:
                writer.add_scalar("hyperparams/beta", beta_t, global_step)
            if log_optimizer:
                writer.add_scalar("optimizer/grad_norm_global", grad_norm, global_step)
                writer.add_scalar("optimizer/param_norm_global", param_norm, global_step)
                writer.add_scalar("optimizer/update_norm_global", update_norm, global_step)
                writer.add_scalar("optimizer/adam_exp_avg_norm", adam_exp_avg_norm, global_step)
                writer.add_scalar("optimizer/adam_exp_avg_sq_norm", adam_exp_avg_sq_norm, global_step)
            if log_lr:
                writer.add_scalar("optimizer/lr", learning_rate, global_step)
            if log_denorm_l1:
                writer.add_scalar("denorm_l1/joint", batch_l1_joint, global_step)
                writer.add_scalar("denorm_l1/gripper", batch_l1_gripper, global_step)
            if log_ranges:
                writer.add_scalar("ranges/pred_min_joint", pred_joint_min, global_step)
                writer.add_scalar("ranges/pred_max_joint", pred_joint_max, global_step)
                writer.add_scalar("ranges/pred_min_gripper", pred_gripper_min, global_step)
                writer.add_scalar("ranges/pred_max_gripper", pred_gripper_max, global_step)
                writer.add_scalar("ranges/target_min_joint", target_joint_min, global_step)
                writer.add_scalar("ranges/target_max_joint", target_joint_max, global_step)
                writer.add_scalar("ranges/target_min_gripper", target_gripper_min, global_step)
                writer.add_scalar("ranges/target_max_gripper", target_gripper_max, global_step)
            if log_latent:
                writer.add_scalar("latent/mu_norm", mu_norm, global_step)
                writer.add_scalar("latent/log_sigma_x2_mean", log_sigma_x2_mean, global_step)
                writer.add_scalar("latent/sigma_mean", sigma_mean, global_step)
            if log_activations:
                for tag in ACTIVATION_HOOK_TAGS:
                    writer.add_scalar(
                        f"activations/{tag}",
                        activation_norms[tag],
                        global_step,
                    )
            global_step += 1
            train_samples += img.shape[0]

            running_avg_loss = epoch_loss / num_batches
            last_loss_running_avg = running_avg_loss
            checkpoint_started = time.monotonic()
            snapshot_path = maybe_save_step_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                run_name=run_name,
                config=config,
                dataset=dataset_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                loss_running_avg=running_avg_loss,
                normalization=normalization,
            )
            if snapshot_path is not None:
                checkpoint_seconds = time.monotonic() - checkpoint_started
                train_seconds = max(checkpoint_started - train_interval_started, 1e-9)
                print(f"saved snapshot => {snapshot_path.resolve()}")
                eval_env = evaluate_and_log_checkpoint(
                    model=model,
                    env=eval_env,
                    dataset=dataset,
                    image_keys=tuple(config["image_keys"]),
                    normalization=normalization,
                    device=device,
                    writer=writer,
                    global_step=global_step,
                    run_started=run_started,
                    train_seconds=train_seconds,
                    train_steps=global_step - last_evaluated_step,
                    train_samples=train_samples,
                    checkpoint_seconds=checkpoint_seconds,
                    tensorboard=tensorboard,
                )
                last_evaluated_step = global_step
                train_samples = 0
                train_interval_started = time.monotonic()

            if global_step >= target_steps:
                break

        if num_batches == 0:
            break

        if log_epoch_metrics:
            avg_loss = epoch_loss / num_batches
            avg_action_loss = epoch_action_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            avg_weighted_kl_loss = epoch_weighted_kl_loss / num_batches
            epoch_beta = beta_at_step(
                step=max(global_step - 1, 0),
                beta_start=beta_start,
                beta=beta,
                beta_warmup_steps=beta_warmup_steps,
            )
            if avg_loss > 0.0:
                kl_fraction = avg_weighted_kl_loss / avg_loss
            else:
                kl_fraction = 0.0

            writer.add_scalar("epoch_metrics/loss", avg_loss, epoch)
            writer.add_scalar("epoch_metrics/action_loss", avg_action_loss, epoch)
            writer.add_scalar("epoch_metrics/kl_loss", avg_kl_loss, epoch)
            writer.add_scalar("epoch_metrics/weighted_kl_loss", avg_weighted_kl_loss, epoch)
            writer.add_scalar("epoch_metrics/kl_fraction", kl_fraction, epoch)
            writer.add_scalar("epoch_metrics/beta", epoch_beta, epoch)

        if log_sensitivity:
            image_l1, proprio_l1 = measure_observation_sensitivity(
                model=model, images=probe_images, proprio=probe_proprio,
            )
            writer.add_scalar("sensitivity/image_swap_l1", image_l1, epoch)
            writer.add_scalar("sensitivity/proprio_swap_l1", proprio_l1, epoch)

        epoch += 1

    if global_step > 0 and global_step % checkpoint_every != 0:
        checkpoint_started = time.monotonic()
        snapshot_path = checkpoint_path_for_step(
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
        )
        save_checkpoint(
            path=snapshot_path,
            run_name=run_name,
            config=config,
            dataset=dataset_path,
            epoch=epoch - 1,
            global_step=global_step,
            model=model,
            loss_running_avg=last_loss_running_avg,
            normalization=normalization,
        )
        print(f"saved final snapshot => {snapshot_path.resolve()}")
        eval_env = evaluate_and_log_checkpoint(
            model=model,
            env=eval_env,
            dataset=dataset,
            image_keys=tuple(config["image_keys"]),
            normalization=normalization,
            device=device,
            writer=writer,
            global_step=global_step,
            run_started=run_started,
            train_seconds=max(checkpoint_started - train_interval_started, 1e-9),
            train_steps=global_step - last_evaluated_step,
            train_samples=train_samples,
            checkpoint_seconds=time.monotonic() - checkpoint_started,
            tensorboard=tensorboard,
        )

    if eval_env is not None:
        close_env(eval_env)
    for handle in activation_hook_handles:
        handle.remove()
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT v2 (CVAE + action chunking).")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the training hyperparameter config file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "path to the CAN PH HDF5 training dataset; overrides the config's "
            "required 'dataset' key"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="override total training steps",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    # Record Python stacks in training.log if native HDF5/CUDA code faults.
    faulthandler.enable()
    args = parse_args()

    config = load_config(path=args.config)

    if config_uses_inheritance(path=args.config):
        config["name"] = args.config.stem

    if args.steps is not None:
        config["steps"] = args.steps

    dataset = resolve_dataset(cli_dataset=args.dataset, config=config)
    train(config=config, dataset=dataset)


if __name__ == "__main__":
    main()
