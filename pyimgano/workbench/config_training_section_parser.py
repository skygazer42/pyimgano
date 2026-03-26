from __future__ import annotations

from typing import Any, Mapping

from pyimgano.workbench.config_parse_primitives import (
    _optional_bool,
    _optional_float,
    _optional_int,
    _optional_int_sequence,
    _optional_nonempty_str,
    _parse_checkpoint_name,
    _require_mapping,
)
from pyimgano.workbench.config_types import TrainingConfig


def _parse_training_config(top: Mapping[str, Any]) -> TrainingConfig:
    training_raw = top.get("training", None)
    if training_raw is None:
        return TrainingConfig()

    t_map = _require_mapping(training_raw, name="training")
    epochs = _optional_int(t_map.get("epochs", None), name="training.epochs")
    if epochs is not None and epochs <= 0:
        raise ValueError("training.epochs must be positive or null")
    lr = _optional_float(t_map.get("lr", None), name="training.lr")
    if lr is not None and lr <= 0:
        raise ValueError("training.lr must be positive or null")
    validation_fraction = _optional_float(
        t_map.get("validation_fraction", None),
        name="training.validation_fraction",
    )
    if validation_fraction is not None and not 0.0 < validation_fraction < 1.0:
        raise ValueError("training.validation_fraction must be in (0, 1) or null")
    early_stopping_patience = _optional_int(
        t_map.get("early_stopping_patience", None),
        name="training.early_stopping_patience",
    )
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("training.early_stopping_patience must be positive or null")
    early_stopping_min_delta = _optional_float(
        t_map.get("early_stopping_min_delta", None),
        name="training.early_stopping_min_delta",
    )
    if early_stopping_min_delta is not None and early_stopping_min_delta < 0:
        raise ValueError("training.early_stopping_min_delta must be non-negative or null")
    max_steps = _optional_int(t_map.get("max_steps", None), name="training.max_steps")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("training.max_steps must be positive or null")
    max_train_samples = _optional_int(
        t_map.get("max_train_samples", None),
        name="training.max_train_samples",
    )
    if max_train_samples is not None and max_train_samples <= 0:
        raise ValueError("training.max_train_samples must be positive or null")
    batch_size = _optional_int(t_map.get("batch_size", None), name="training.batch_size")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("training.batch_size must be positive or null")
    num_workers = _optional_int(t_map.get("num_workers", None), name="training.num_workers")
    if num_workers is not None and num_workers < 0:
        raise ValueError("training.num_workers must be non-negative or null")
    weight_decay = _optional_float(
        t_map.get("weight_decay", None),
        name="training.weight_decay",
    )
    if weight_decay is not None and weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative or null")
    optimizer_name_raw = t_map.get("optimizer_name", None)
    optimizer_name: str | None = None
    if optimizer_name_raw is not None:
        optimizer_name = str(optimizer_name_raw).strip().lower()
        if optimizer_name not in {"adam", "adamw", "sgd", "rmsprop"}:
            raise ValueError("training.optimizer_name must be one of: adam, adamw, sgd, rmsprop")
    optimizer_momentum = _optional_float(
        t_map.get("optimizer_momentum", None),
        name="training.optimizer_momentum",
    )
    if optimizer_momentum is not None and optimizer_momentum < 0:
        raise ValueError("training.optimizer_momentum must be non-negative or null")
    optimizer_nesterov = _optional_bool(
        t_map.get("optimizer_nesterov", None),
        name="training.optimizer_nesterov",
    )
    optimizer_dampening = _optional_float(
        t_map.get("optimizer_dampening", None),
        name="training.optimizer_dampening",
    )
    if optimizer_dampening is not None and optimizer_dampening < 0:
        raise ValueError("training.optimizer_dampening must be non-negative or null")
    adam_beta1 = _optional_float(
        t_map.get("adam_beta1", None),
        name="training.adam_beta1",
    )
    if adam_beta1 is not None and not 0.0 < adam_beta1 < 1.0:
        raise ValueError("training.adam_beta1 must be in (0, 1) or null")
    adam_beta2 = _optional_float(
        t_map.get("adam_beta2", None),
        name="training.adam_beta2",
    )
    if adam_beta2 is not None and not 0.0 < adam_beta2 < 1.0:
        raise ValueError("training.adam_beta2 must be in (0, 1) or null")
    adam_amsgrad = _optional_bool(
        t_map.get("adam_amsgrad", None),
        name="training.adam_amsgrad",
    )
    optimizer_eps = _optional_float(
        t_map.get("optimizer_eps", None),
        name="training.optimizer_eps",
    )
    if optimizer_eps is not None and optimizer_eps <= 0:
        raise ValueError("training.optimizer_eps must be positive or null")
    rmsprop_alpha = _optional_float(
        t_map.get("rmsprop_alpha", None),
        name="training.rmsprop_alpha",
    )
    if rmsprop_alpha is not None and not 0.0 < rmsprop_alpha < 1.0:
        raise ValueError("training.rmsprop_alpha must be in (0, 1) or null")
    rmsprop_centered = _optional_bool(
        t_map.get("rmsprop_centered", None),
        name="training.rmsprop_centered",
    )
    if (
        optimizer_name == "sgd"
        and optimizer_nesterov is True
        and optimizer_dampening is not None
        and optimizer_dampening > 0.0
    ):
        raise ValueError(
            "training.optimizer_dampening must be 0 when optimizer_name=sgd and optimizer_nesterov=true"
        )
    scheduler_name_raw = t_map.get("scheduler_name", None)
    scheduler_name: str | None = None
    if scheduler_name_raw is not None:
        scheduler_name = str(scheduler_name_raw).strip().lower()
        if scheduler_name not in {"step", "cosine", "exponential", "plateau", "multistep"}:
            raise ValueError(
                "training.scheduler_name must be one of: step, cosine, exponential, plateau, multistep"
            )
    scheduler_milestones = _optional_int_sequence(
        t_map.get("scheduler_milestones", None),
        name="training.scheduler_milestones",
    )
    if scheduler_milestones is not None:
        if any(v <= 0 for v in scheduler_milestones):
            raise ValueError("training.scheduler_milestones must contain positive ints")
        if tuple(sorted(scheduler_milestones)) != scheduler_milestones:
            raise ValueError("training.scheduler_milestones must be sorted ascending")
        if len(set(scheduler_milestones)) != len(scheduler_milestones):
            raise ValueError("training.scheduler_milestones must not contain duplicates")
    if scheduler_name == "multistep" and scheduler_milestones is None:
        raise ValueError("training.scheduler_milestones is required when scheduler_name=multistep")
    criterion_name_raw = t_map.get("criterion_name", None)
    criterion_name: str | None = None
    if criterion_name_raw is not None:
        criterion_name = str(criterion_name_raw).strip().lower()
        if criterion_name not in {"mse", "mae", "l1", "bce"}:
            raise ValueError("training.criterion_name must be one of: mse, mae, l1, bce")
    scheduler_step_size = _optional_int(
        t_map.get("scheduler_step_size", None),
        name="training.scheduler_step_size",
    )
    if scheduler_step_size is not None and scheduler_step_size <= 0:
        raise ValueError("training.scheduler_step_size must be positive or null")
    scheduler_gamma = _optional_float(
        t_map.get("scheduler_gamma", None),
        name="training.scheduler_gamma",
    )
    if scheduler_gamma is not None and scheduler_gamma <= 0:
        raise ValueError("training.scheduler_gamma must be positive or null")
    scheduler_t_max = _optional_int(
        t_map.get("scheduler_t_max", None),
        name="training.scheduler_t_max",
    )
    if scheduler_t_max is not None and scheduler_t_max <= 0:
        raise ValueError("training.scheduler_t_max must be positive or null")
    scheduler_eta_min = _optional_float(
        t_map.get("scheduler_eta_min", None),
        name="training.scheduler_eta_min",
    )
    if scheduler_eta_min is not None and scheduler_eta_min < 0:
        raise ValueError("training.scheduler_eta_min must be non-negative or null")
    scheduler_patience = _optional_int(
        t_map.get("scheduler_patience", None),
        name="training.scheduler_patience",
    )
    if scheduler_patience is not None and scheduler_patience < 0:
        raise ValueError("training.scheduler_patience must be non-negative or null")
    scheduler_factor = _optional_float(
        t_map.get("scheduler_factor", None),
        name="training.scheduler_factor",
    )
    if scheduler_factor is not None and scheduler_factor <= 0:
        raise ValueError("training.scheduler_factor must be positive or null")
    scheduler_min_lr = _optional_float(
        t_map.get("scheduler_min_lr", None),
        name="training.scheduler_min_lr",
    )
    if scheduler_min_lr is not None and scheduler_min_lr < 0:
        raise ValueError("training.scheduler_min_lr must be non-negative or null")
    scheduler_cooldown = _optional_int(
        t_map.get("scheduler_cooldown", None),
        name="training.scheduler_cooldown",
    )
    if scheduler_cooldown is not None and scheduler_cooldown < 0:
        raise ValueError("training.scheduler_cooldown must be non-negative or null")
    scheduler_threshold = _optional_float(
        t_map.get("scheduler_threshold", None),
        name="training.scheduler_threshold",
    )
    if scheduler_threshold is not None and scheduler_threshold < 0:
        raise ValueError("training.scheduler_threshold must be non-negative or null")
    scheduler_threshold_mode_raw = t_map.get("scheduler_threshold_mode", None)
    scheduler_threshold_mode: str | None = None
    if scheduler_threshold_mode_raw is not None:
        scheduler_threshold_mode = str(scheduler_threshold_mode_raw).strip().lower()
        if scheduler_threshold_mode not in {"rel", "abs"}:
            raise ValueError("training.scheduler_threshold_mode must be one of: rel, abs")
    scheduler_eps = _optional_float(
        t_map.get("scheduler_eps", None),
        name="training.scheduler_eps",
    )
    if scheduler_eps is not None and scheduler_eps < 0:
        raise ValueError("training.scheduler_eps must be non-negative or null")
    shuffle_train = _optional_bool(
        t_map.get("shuffle_train", None),
        name="training.shuffle_train",
    )
    drop_last = _optional_bool(
        t_map.get("drop_last", None),
        name="training.drop_last",
    )
    pin_memory = _optional_bool(
        t_map.get("pin_memory", None),
        name="training.pin_memory",
    )
    persistent_workers = _optional_bool(
        t_map.get("persistent_workers", None),
        name="training.persistent_workers",
    )
    validation_split_seed = _optional_int(
        t_map.get("validation_split_seed", None),
        name="training.validation_split_seed",
    )
    if validation_split_seed is not None and validation_split_seed < 0:
        raise ValueError("training.validation_split_seed must be non-negative or null")
    warmup_epochs = _optional_int(
        t_map.get("warmup_epochs", None),
        name="training.warmup_epochs",
    )
    if warmup_epochs is not None and warmup_epochs <= 0:
        raise ValueError("training.warmup_epochs must be positive or null")
    warmup_start_factor = _optional_float(
        t_map.get("warmup_start_factor", None),
        name="training.warmup_start_factor",
    )
    if warmup_start_factor is not None and not 0.0 <= warmup_start_factor <= 1.0:
        raise ValueError("training.warmup_start_factor must be in [0, 1] or null")
    ema_enabled = _optional_bool(
        t_map.get("ema_enabled", None),
        name="training.ema_enabled",
    )
    ema_decay = _optional_float(
        t_map.get("ema_decay", None),
        name="training.ema_decay",
    )
    if ema_decay is not None and not 0.0 < ema_decay < 1.0:
        raise ValueError("training.ema_decay must be in (0, 1) or null")
    ema_start_epoch = _optional_int(
        t_map.get("ema_start_epoch", None),
        name="training.ema_start_epoch",
    )
    if ema_start_epoch is not None and ema_start_epoch <= 0:
        raise ValueError("training.ema_start_epoch must be positive or null")
    resume_from_checkpoint = _optional_nonempty_str(
        t_map.get("resume_from_checkpoint", None),
        name="training.resume_from_checkpoint",
    )
    tracker_backend_raw = t_map.get("tracker_backend", None)
    tracker_backend: str | None = None
    if tracker_backend_raw is not None:
        tracker_backend = str(tracker_backend_raw).strip().lower()
        if tracker_backend not in {"none", "jsonl", "tensorboard", "wandb", "mlflow"}:
            raise ValueError(
                "training.tracker_backend must be one of: none, jsonl, tensorboard, wandb, mlflow"
            )
    tracker_dir = _optional_nonempty_str(
        t_map.get("tracker_dir", None),
        name="training.tracker_dir",
    )
    tracker_project = _optional_nonempty_str(
        t_map.get("tracker_project", None),
        name="training.tracker_project",
    )
    tracker_run_name = _optional_nonempty_str(
        t_map.get("tracker_run_name", None),
        name="training.tracker_run_name",
    )
    tracker_mode = _optional_nonempty_str(
        t_map.get("tracker_mode", None),
        name="training.tracker_mode",
    )
    callbacks: tuple[str, ...] = ()
    callbacks_raw = t_map.get("callbacks", None)
    if callbacks_raw is not None:
        if not isinstance(callbacks_raw, (list, tuple)):
            raise ValueError("training.callbacks must be a list/tuple of callback names")
        seen: set[str] = set()
        parsed: list[str] = []
        for raw_name in callbacks_raw:
            name = str(raw_name).strip().lower()
            if not name:
                raise ValueError("training.callbacks must contain non-empty callback names")
            if name not in {"metrics_logger", "resource_profiler"}:
                raise ValueError(
                    "training.callbacks contains unsupported callback "
                    f"{raw_name!r}. Supported: metrics_logger, resource_profiler"
                )
            if name not in seen:
                seen.add(name)
                parsed.append(name)
        callbacks = tuple(parsed)
    return TrainingConfig(
        enabled=bool(t_map.get("enabled", False)),
        epochs=epochs,
        lr=lr,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        max_steps=max_steps,
        max_train_samples=max_train_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        optimizer_momentum=optimizer_momentum,
        optimizer_nesterov=optimizer_nesterov,
        optimizer_dampening=optimizer_dampening,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_amsgrad=adam_amsgrad,
        optimizer_eps=optimizer_eps,
        rmsprop_alpha=rmsprop_alpha,
        rmsprop_centered=rmsprop_centered,
        scheduler_name=scheduler_name,
        scheduler_milestones=scheduler_milestones,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        scheduler_t_max=scheduler_t_max,
        scheduler_eta_min=scheduler_eta_min,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        scheduler_min_lr=scheduler_min_lr,
        scheduler_cooldown=scheduler_cooldown,
        scheduler_threshold=scheduler_threshold,
        scheduler_threshold_mode=scheduler_threshold_mode,
        scheduler_eps=scheduler_eps,
        criterion_name=criterion_name,
        shuffle_train=shuffle_train,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        validation_split_seed=validation_split_seed,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        ema_enabled=ema_enabled,
        ema_decay=ema_decay,
        ema_start_epoch=ema_start_epoch,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_name=_parse_checkpoint_name(t_map.get("checkpoint_name", None)),
        tracker_backend=tracker_backend,
        tracker_dir=tracker_dir,
        tracker_project=tracker_project,
        tracker_run_name=tracker_run_name,
        tracker_mode=tracker_mode,
        callbacks=callbacks,
    )


__all__ = ["_parse_training_config"]
