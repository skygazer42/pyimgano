from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import numpy as np
import pyimgano.services.workbench_service as workbench_service

from pyimgano.recipes.registry import register_recipe
from pyimgano.reporting.environment import collect_environment
from pyimgano.reporting.report import save_run_report, stamp_report_payload
from pyimgano.reporting.runs import (
    build_workbench_run_dir_name,
    build_workbench_run_paths,
    ensure_run_dir,
)
from pyimgano.training.checkpointing import save_checkpoint
from pyimgano.training.runner import micro_finetune
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.training_runtime import restore_training_checkpoint_if_requested


def _load_train_paths(
    *,
    dataset: str,
    root: str,
    category: str,
    resize: tuple[int, int],
) -> list[str]:
    from pyimgano.pipelines.mvtec_visa import load_benchmark_split

    split = load_benchmark_split(
        dataset=dataset,  # type: ignore[arg-type]
        root=root,
        category=category,
        resize=tuple(resize),
        load_masks=False,
    )
    return list(split.train_paths)


def _load_train_numpy(
    *,
    dataset: str,
    root: str,
    category: str,
    resize: tuple[int, int],
) -> list[np.ndarray]:
    from pyimgano.datasets import load_dataset

    ds = load_dataset(  # nosec B615 - pyimgano.datasets.load_dataset, not Hugging Face Hub
        dataset,
        root,
        category=category,
        resize=tuple(resize),
        load_masks=False,
    )
    train_data = np.asarray(ds.get_train_data())
    return [np.asarray(train_data[i]) for i in range(int(train_data.shape[0]))]


def _create_detector(config: WorkbenchConfig) -> Any:
    return workbench_service.create_workbench_detector(config=config)


def _build_training_kwargs(config: WorkbenchConfig) -> dict[str, Any]:
    training_kwargs: dict[str, Any] = {}
    if config.training.epochs is not None:
        training_kwargs["epochs"] = int(config.training.epochs)
    if config.training.lr is not None:
        training_kwargs["lr"] = float(config.training.lr)
    if config.training.validation_fraction is not None:
        training_kwargs["validation_fraction"] = float(config.training.validation_fraction)
    if config.training.early_stopping_patience is not None:
        training_kwargs["early_stopping_patience"] = int(config.training.early_stopping_patience)
    if config.training.early_stopping_min_delta is not None:
        training_kwargs["early_stopping_min_delta"] = float(
            config.training.early_stopping_min_delta
        )
    if config.training.max_steps is not None:
        training_kwargs["max_steps"] = int(config.training.max_steps)
    if config.training.max_train_samples is not None:
        training_kwargs["max_train_samples"] = int(config.training.max_train_samples)
    if config.training.batch_size is not None:
        training_kwargs["batch_size"] = int(config.training.batch_size)
    if config.training.num_workers is not None:
        training_kwargs["num_workers"] = int(config.training.num_workers)
    if config.training.weight_decay is not None:
        training_kwargs["weight_decay"] = float(config.training.weight_decay)
    if config.training.optimizer_name is not None:
        training_kwargs["optimizer_name"] = str(config.training.optimizer_name)
    if config.training.optimizer_momentum is not None:
        training_kwargs["optimizer_momentum"] = float(config.training.optimizer_momentum)
    if config.training.optimizer_nesterov is not None:
        training_kwargs["optimizer_nesterov"] = bool(config.training.optimizer_nesterov)
    if config.training.optimizer_dampening is not None:
        training_kwargs["optimizer_dampening"] = float(config.training.optimizer_dampening)
    if config.training.adam_beta1 is not None:
        training_kwargs["adam_beta1"] = float(config.training.adam_beta1)
    if config.training.adam_beta2 is not None:
        training_kwargs["adam_beta2"] = float(config.training.adam_beta2)
    if config.training.adam_amsgrad is not None:
        training_kwargs["adam_amsgrad"] = bool(config.training.adam_amsgrad)
    if config.training.optimizer_eps is not None:
        training_kwargs["optimizer_eps"] = float(config.training.optimizer_eps)
    if config.training.rmsprop_alpha is not None:
        training_kwargs["rmsprop_alpha"] = float(config.training.rmsprop_alpha)
    if config.training.rmsprop_centered is not None:
        training_kwargs["rmsprop_centered"] = bool(config.training.rmsprop_centered)
    if config.training.scheduler_name is not None:
        training_kwargs["scheduler_name"] = str(config.training.scheduler_name)
    if config.training.scheduler_milestones is not None:
        training_kwargs["scheduler_milestones"] = [
            int(v) for v in config.training.scheduler_milestones
        ]
    if config.training.scheduler_step_size is not None:
        training_kwargs["scheduler_step_size"] = int(config.training.scheduler_step_size)
    if config.training.scheduler_gamma is not None:
        training_kwargs["scheduler_gamma"] = float(config.training.scheduler_gamma)
    if config.training.scheduler_t_max is not None:
        training_kwargs["scheduler_t_max"] = int(config.training.scheduler_t_max)
    if config.training.scheduler_eta_min is not None:
        training_kwargs["scheduler_eta_min"] = float(config.training.scheduler_eta_min)
    if config.training.scheduler_patience is not None:
        training_kwargs["scheduler_patience"] = int(config.training.scheduler_patience)
    if config.training.scheduler_factor is not None:
        training_kwargs["scheduler_factor"] = float(config.training.scheduler_factor)
    if config.training.scheduler_min_lr is not None:
        training_kwargs["scheduler_min_lr"] = float(config.training.scheduler_min_lr)
    if config.training.scheduler_cooldown is not None:
        training_kwargs["scheduler_cooldown"] = int(config.training.scheduler_cooldown)
    if config.training.scheduler_threshold is not None:
        training_kwargs["scheduler_threshold"] = float(config.training.scheduler_threshold)
    if config.training.scheduler_threshold_mode is not None:
        training_kwargs["scheduler_threshold_mode"] = str(
            config.training.scheduler_threshold_mode
        )
    if config.training.scheduler_eps is not None:
        training_kwargs["scheduler_eps"] = float(config.training.scheduler_eps)
    if config.training.criterion_name is not None:
        training_kwargs["criterion_name"] = str(config.training.criterion_name)
    if config.training.shuffle_train is not None:
        training_kwargs["shuffle_train"] = bool(config.training.shuffle_train)
    if config.training.drop_last is not None:
        training_kwargs["drop_last"] = bool(config.training.drop_last)
    if config.training.pin_memory is not None:
        training_kwargs["pin_memory"] = bool(config.training.pin_memory)
    if config.training.persistent_workers is not None:
        training_kwargs["persistent_workers"] = bool(config.training.persistent_workers)
    if config.training.validation_split_seed is not None:
        training_kwargs["validation_split_seed"] = int(config.training.validation_split_seed)
    if config.training.warmup_epochs is not None:
        training_kwargs["warmup_epochs"] = int(config.training.warmup_epochs)
    if config.training.warmup_start_factor is not None:
        training_kwargs["warmup_start_factor"] = float(config.training.warmup_start_factor)
    if config.training.ema_enabled is not None:
        training_kwargs["ema_enabled"] = bool(config.training.ema_enabled)
    if config.training.ema_decay is not None:
        training_kwargs["ema_decay"] = float(config.training.ema_decay)
    if config.training.ema_start_epoch is not None:
        training_kwargs["ema_start_epoch"] = int(config.training.ema_start_epoch)
    return training_kwargs


@register_recipe(
    "micro-finetune-autoencoder",
    tags=("builtin", "training"),
    metadata={
        "description": "Micro-finetune recipe intended for small autoencoder-style models (best-effort).",
    },
)
def micro_finetune_autoencoder(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "micro-finetune-autoencoder"

    if not bool(config.output.save_run):
        raise ValueError(
            f"{recipe_name} requires output.save_run=true (needs checkpoint artifacts)."
        )

    category_for_name = (
        None if str(config.dataset.category).lower() == "all" else str(config.dataset.category)
    )
    run_name = build_workbench_run_dir_name(
        dataset=str(config.dataset.name),
        recipe=recipe_name,
        model=str(config.model.name),
        category=category_for_name,
    )
    run_dir = ensure_run_dir(output_dir=config.output.output_dir, name=run_name)
    paths = build_workbench_run_paths(run_dir)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    save_run_report(paths.environment_json, collect_environment())
    save_run_report(paths.config_json, {"config": asdict(config)})

    if config.dataset.input_mode == "paths":
        train_inputs: Sequence[Any] = _load_train_paths(
            dataset=str(config.dataset.name),
            root=str(config.dataset.root),
            category=str(config.dataset.category),
            resize=tuple(config.dataset.resize),
        )
    elif config.dataset.input_mode == "numpy":
        train_inputs = _load_train_numpy(
            dataset=str(config.dataset.name),
            root=str(config.dataset.root),
            category=str(config.dataset.category),
            resize=tuple(config.dataset.resize),
        )
    else:
        raise ValueError(
            f"Unknown input_mode: {config.dataset.input_mode!r}. Choose from: paths, numpy."
        )

    if config.dataset.limit_train is not None:
        train_inputs = list(train_inputs)[: int(config.dataset.limit_train)]

    detector = _create_detector(config)
    checkpoint_restore = restore_training_checkpoint_if_requested(
        detector=detector,
        config=config,
    )
    training = micro_finetune(
        detector,
        train_inputs,
        seed=config.seed,
        fit_kwargs=_build_training_kwargs(config),
    )
    if checkpoint_restore is not None:
        training = dict(training or {})
        training["checkpoint_restore"] = checkpoint_restore

    checkpoint_path = paths.checkpoints_dir / "model.pt"
    saved = save_checkpoint(detector, checkpoint_path)
    try:
        rel = saved.relative_to(paths.run_dir)
        checkpoint_rel = rel.as_posix()
    except Exception:
        checkpoint_rel = saved.as_posix()

    payload: dict[str, Any] = {
        "dataset": str(config.dataset.name),
        "category": str(config.dataset.category),
        "model": str(config.model.name),
        "recipe": recipe_name,
        "input_mode": str(config.dataset.input_mode),
        "device": str(config.model.device),
        "preset": config.model.preset,
        "resize": [int(config.dataset.resize[0]), int(config.dataset.resize[1])],
        "checkpoint": {"path": checkpoint_rel},
        "training": training,
        "run_dir": str(paths.run_dir),
    }
    payload = stamp_report_payload(payload)

    save_run_report(paths.report_json, payload)
    return payload
