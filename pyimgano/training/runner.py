from __future__ import annotations

import inspect
import time
from typing import Any, Mapping, Sequence

import numpy as np


_FIT_KWARG_ALIASES: dict[str, tuple[str, ...]] = {
    "epochs": ("epochs", "epoch_num"),
    "lr": ("lr", "learning_rate"),
    "batch_size": ("batch_size",),
    "num_workers": ("num_workers",),
    "weight_decay": ("weight_decay", "l2_weight"),
    "optimizer_name": ("optimizer_name",),
    "optimizer_momentum": ("optimizer_momentum",),
    "optimizer_nesterov": ("optimizer_nesterov",),
    "optimizer_dampening": ("optimizer_dampening",),
    "adam_beta1": ("adam_beta1",),
    "adam_beta2": ("adam_beta2",),
    "adam_amsgrad": ("adam_amsgrad",),
    "optimizer_eps": ("optimizer_eps",),
    "rmsprop_alpha": ("rmsprop_alpha",),
    "rmsprop_centered": ("rmsprop_centered",),
    "scheduler_name": ("scheduler_name",),
    "scheduler_milestones": ("scheduler_milestones",),
    "scheduler_step_size": ("scheduler_step_size",),
    "scheduler_gamma": ("scheduler_gamma",),
    "scheduler_t_max": ("scheduler_t_max",),
    "scheduler_eta_min": ("scheduler_eta_min",),
    "scheduler_patience": ("scheduler_patience",),
    "scheduler_factor": ("scheduler_factor",),
    "scheduler_min_lr": ("scheduler_min_lr",),
    "scheduler_cooldown": ("scheduler_cooldown",),
    "scheduler_threshold": ("scheduler_threshold",),
    "scheduler_threshold_mode": ("scheduler_threshold_mode",),
    "scheduler_eps": ("scheduler_eps",),
    "criterion_name": ("criterion_name",),
    "shuffle_train": ("shuffle_train",),
    "drop_last": ("drop_last",),
    "pin_memory": ("pin_memory",),
    "persistent_workers": ("persistent_workers",),
    "warmup_epochs": ("warmup_epochs",),
    "warmup_start_factor": ("warmup_start_factor",),
    "ema_enabled": ("ema_enabled",),
    "ema_decay": ("ema_decay",),
    "ema_start_epoch": ("ema_start_epoch",),
    "validation_inputs": ("validation_inputs", "validation_data", "val_inputs", "val_data"),
    "max_steps": ("max_steps",),
    "early_stopping_patience": ("early_stopping_patience",),
    "early_stopping_min_delta": ("early_stopping_min_delta",),
}

_ATTR_OVERRIDE_ALIASES: dict[str, tuple[str, ...]] = {
    "epochs": ("epochs", "epoch_num"),
    "lr": ("lr", "learning_rate"),
    "batch_size": ("batch_size",),
    "num_workers": ("num_workers",),
    "weight_decay": ("weight_decay", "l2_weight"),
    "optimizer_name": ("optimizer_name",),
    "optimizer_momentum": ("optimizer_momentum",),
    "optimizer_nesterov": ("optimizer_nesterov",),
    "optimizer_dampening": ("optimizer_dampening",),
    "adam_beta1": ("adam_beta1",),
    "adam_beta2": ("adam_beta2",),
    "adam_amsgrad": ("adam_amsgrad",),
    "optimizer_eps": ("optimizer_eps",),
    "rmsprop_alpha": ("rmsprop_alpha",),
    "rmsprop_centered": ("rmsprop_centered",),
    "scheduler_name": ("scheduler_name",),
    "scheduler_milestones": ("scheduler_milestones",),
    "scheduler_step_size": ("scheduler_step_size",),
    "scheduler_gamma": ("scheduler_gamma",),
    "scheduler_t_max": ("scheduler_t_max",),
    "scheduler_eta_min": ("scheduler_eta_min",),
    "scheduler_patience": ("scheduler_patience",),
    "scheduler_factor": ("scheduler_factor",),
    "scheduler_min_lr": ("scheduler_min_lr",),
    "scheduler_cooldown": ("scheduler_cooldown",),
    "scheduler_threshold": ("scheduler_threshold",),
    "scheduler_threshold_mode": ("scheduler_threshold_mode",),
    "scheduler_eps": ("scheduler_eps",),
    "criterion_name": ("criterion_name",),
    "shuffle_train": ("shuffle_train",),
    "drop_last": ("drop_last",),
    "pin_memory": ("pin_memory",),
    "persistent_workers": ("persistent_workers",),
    "warmup_epochs": ("warmup_epochs",),
    "warmup_start_factor": ("warmup_start_factor",),
    "ema_enabled": ("ema_enabled",),
    "ema_decay": ("ema_decay",),
    "ema_start_epoch": ("ema_start_epoch",),
    "max_steps": ("max_steps",),
    "early_stopping_patience": ("early_stopping_patience",),
    "early_stopping_min_delta": ("early_stopping_min_delta",),
}


def _seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))

    try:
        import torch
    except Exception:
        return

    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        return


def _maybe_limit_train_inputs(
    train_inputs: Sequence[Any],
    *,
    max_train_samples: int | None,
    seed: int | None,
) -> list[Any]:
    inputs = list(train_inputs)
    if max_train_samples is None or max_train_samples >= len(inputs):
        return inputs

    rng = np.random.default_rng(None if seed is None else int(seed))
    indices = np.arange(len(inputs), dtype=int)
    rng.shuffle(indices)
    keep = np.sort(indices[: int(max_train_samples)])
    return [inputs[int(i)] for i in keep]


def _split_validation_inputs(
    train_inputs: Sequence[Any],
    *,
    validation_fraction: float | None,
    validation_split_seed: int | None = None,
) -> tuple[list[Any], list[Any]]:
    inputs = list(train_inputs)
    if validation_fraction is None or validation_fraction <= 0.0 or len(inputs) < 2:
        return inputs, []

    val_count = int(np.ceil(float(validation_fraction) * float(len(inputs))))
    val_count = max(1, min(len(inputs) - 1, val_count))
    if validation_split_seed is None:
        split_at = len(inputs) - val_count
        return inputs[:split_at], inputs[split_at:]

    rng = np.random.default_rng(int(validation_split_seed))
    indices = np.arange(len(inputs), dtype=int)
    rng.shuffle(indices)
    val_indices = np.sort(indices[:val_count])
    val_index_set = {int(i) for i in val_indices}
    train_outputs = [inputs[i] for i in range(len(inputs)) if i not in val_index_set]
    validation_outputs = [inputs[int(i)] for i in val_indices]
    return train_outputs, validation_outputs


def _inspect_fit_signature(detector: Any) -> tuple[set[str], bool]:
    try:
        sig = inspect.signature(detector.fit)
    except (TypeError, ValueError):
        return set(), False

    accepted: set[str] = set()
    accepts_var_kwargs = False
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if param.name in {"self", "X", "x", "y", "_y"}:
            continue
        accepted.add(str(param.name))
    return accepted, accepts_var_kwargs


def _select_fit_kwargs(
    detector: Any, requested: Mapping[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    accepted, accepts_var_kwargs = _inspect_fit_signature(detector)
    used: dict[str, Any] = {}
    explicit_support: set[str] = set()
    for canonical_name, aliases in _FIT_KWARG_ALIASES.items():
        value = requested.get(canonical_name, None)
        if value is None:
            continue
        actual_name = None
        for alias in aliases:
            if alias in accepted:
                actual_name = alias
                explicit_support.add(canonical_name)
                break
        if actual_name is None and accepts_var_kwargs:
            actual_name = aliases[0]
        if actual_name is not None:
            used[str(actual_name)] = value
    return used, explicit_support


def _apply_detector_attr_overrides(
    detector: Any,
    requested: Mapping[str, Any],
    *,
    explicit_fit_support: set[str],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for canonical_name, aliases in _ATTR_OVERRIDE_ALIASES.items():
        value = requested.get(canonical_name, None)
        if value is None:
            continue
        if canonical_name in explicit_fit_support:
            continue
        for alias in aliases:
            if hasattr(detector, alias):
                setattr(detector, alias, value)
                overrides[canonical_name] = value
                break
    return overrides


def _collect_detector_training_state(detector: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if hasattr(detector, "training_epochs_completed_"):
        state["epochs_completed"] = int(getattr(detector, "training_epochs_completed_"))
    if hasattr(detector, "training_steps_completed_"):
        state["steps_completed"] = int(getattr(detector, "training_steps_completed_"))
    if hasattr(detector, "training_stop_reason_"):
        state["stop_reason"] = getattr(detector, "training_stop_reason_")
    if hasattr(detector, "training_best_loss_"):
        best_loss = getattr(detector, "training_best_loss_")
        state["best_loss"] = None if best_loss is None else float(best_loss)
    if hasattr(detector, "training_loss_history_"):
        history = list(getattr(detector, "training_loss_history_"))
        state["loss_history"] = [float(v) for v in history]
    if hasattr(detector, "training_lr_history_"):
        history = list(getattr(detector, "training_lr_history_"))
        state["lr_history"] = [float(v) for v in history]
    if hasattr(detector, "training_last_lr_"):
        last_lr = getattr(detector, "training_last_lr_")
        state["last_lr"] = None if last_lr is None else float(last_lr)
    if hasattr(detector, "training_ema_updates_"):
        state["ema_updates"] = int(getattr(detector, "training_ema_updates_"))
    if hasattr(detector, "training_ema_applied_"):
        state["ema_applied"] = bool(getattr(detector, "training_ema_applied_"))
    return state


def _should_report_dataset_summary(
    *,
    original_train_count: int,
    train_count_used: int,
    validation_count: int,
) -> bool:
    return train_count_used != original_train_count or validation_count > 0


def micro_finetune(
    detector: Any,
    train_inputs: Sequence[Any],
    *,
    seed: int | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Best-effort micro-finetune runner for supported detectors.

    This runner intentionally keeps scope narrow: it sets seeds (best-effort),
    calls `fit(...)`, and returns a small JSON-friendly payload with timing.
    """

    total_start = time.perf_counter()
    if seed is not None:
        _seed_everything(int(seed))

    requested = dict(fit_kwargs or {})
    max_train_samples = requested.pop("max_train_samples", None)
    validation_fraction = requested.pop("validation_fraction", None)
    validation_split_seed = requested.pop("validation_split_seed", None)

    original_inputs = list(train_inputs)
    train_inputs_used = _maybe_limit_train_inputs(
        original_inputs,
        max_train_samples=(
            int(max_train_samples) if max_train_samples is not None else None
        ),
        seed=seed,
    )
    train_inputs_used, validation_inputs = _split_validation_inputs(
        train_inputs_used,
        validation_fraction=(
            float(validation_fraction) if validation_fraction is not None else None
        ),
        validation_split_seed=(
            int(validation_split_seed) if validation_split_seed is not None else None
        ),
    )
    fit_request = dict(requested)
    if validation_inputs:
        fit_request["validation_inputs"] = list(validation_inputs)
    kwargs, explicit_fit_support = _select_fit_kwargs(detector, fit_request)
    detector_attr_overrides_used = _apply_detector_attr_overrides(
        detector,
        requested,
        explicit_fit_support=explicit_fit_support,
    )

    fit_start = time.perf_counter()
    fit_kwargs_used: dict[str, Any]
    try:
        detector.fit(train_inputs_used, **kwargs)
        fit_kwargs_used = kwargs
    except TypeError:
        detector.fit(train_inputs_used)
        fit_kwargs_used = {}
    fit_s = float(time.perf_counter() - fit_start)

    total_s = float(time.perf_counter() - total_start)
    report: dict[str, Any] = {
        "seed": int(seed) if seed is not None else None,
        "fit_kwargs_used": dict(fit_kwargs_used),
        "timing": {
            "fit_s": fit_s,
            "total_s": total_s,
        },
    }
    if detector_attr_overrides_used:
        report["detector_attr_overrides_used"] = dict(detector_attr_overrides_used)
    if _should_report_dataset_summary(
        original_train_count=len(original_inputs),
        train_count_used=len(train_inputs_used),
        validation_count=len(validation_inputs),
    ):
        report["dataset"] = {
            "original_train_count": int(len(original_inputs)),
            "train_count_used": int(len(train_inputs_used)),
            "validation_count": int(len(validation_inputs)),
        }
        if validation_inputs and validation_split_seed is not None:
            report["dataset"]["validation_split_seed"] = int(validation_split_seed)
    detector_training_state = _collect_detector_training_state(detector)
    if detector_training_state:
        report["detector_training_state"] = detector_training_state
    return report
