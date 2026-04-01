# -*- coding: utf-8 -*-
"""Native deep-learning detector base contract for pyimgano.

This provides a minimal, sklearn-like deep detector interface used by
`pyimgano.models.baseCv.BaseVisionDeepDetector` and a few detectors that
implement `build_model/training_forward/evaluating_forward`.

Key responsibilities:
- contamination-based thresholding via `BaseDetector`
- a simple training loop over a torch DataLoader
- a simple evaluation loop producing per-sample anomaly scores

The API is intentionally small and geared toward the needs of this repository.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from pyimgano.train_progress import get_active_train_progress_reporter

from .base_detector import BaseDetector

logger = logging.getLogger(__name__)


def _require_torch():
    from pyimgano.utils.optional_deps import require

    return require("torch", extra="torch", purpose="deep-learning detectors")


def _coerce_loss_float(loss: Any) -> float:
    """Convert model loss outputs to a plain float without autograd warnings."""

    detached = loss.detach() if hasattr(loss, "detach") else loss
    if hasattr(detached, "item"):
        return float(detached.item())
    return float(detached)


@dataclass(frozen=True)
class _CriterionSpec:
    name: str
    factory: Callable[[], Any]


def _resolve_criterion(*, torch, criterion=None, criterion_name: str = "mse"):
    if criterion is not None:
        return criterion

    name = str(criterion_name).lower()
    mapping: dict[str, _CriterionSpec] = {
        "mse": _CriterionSpec("mse", lambda: torch.nn.MSELoss()),
        "mae": _CriterionSpec("mae", lambda: torch.nn.L1Loss()),
        "l1": _CriterionSpec("l1", lambda: torch.nn.L1Loss()),
        "bce": _CriterionSpec("bce", lambda: torch.nn.BCELoss()),
    }
    spec = mapping.get(name)
    if spec is None:
        raise ValueError(f"Unknown criterion_name: {criterion_name!r}")
    return spec.factory()


def _resolve_optimizer(
    *,
    torch,
    name: str,
    params,
    lr: float,
    weight_decay: float,
    momentum: float | None = None,
    nesterov: bool = False,
    dampening: float | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
    amsgrad: bool = False,
    eps: float | None = None,
    alpha: float | None = None,
    centered: bool = False,
):
    key = str(name).lower()
    if key in {"adam", "adamw", "sgd", "rmsprop"}:
        opt_cls = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }[key]
        if key == "sgd":
            return opt_cls(
                params,
                lr=float(lr),
                momentum=float(0.9 if momentum is None else momentum),
                weight_decay=float(weight_decay),
                nesterov=bool(nesterov),
                dampening=float(0.0 if dampening is None else dampening),
            )
        if key == "rmsprop":
            return opt_cls(
                params,
                lr=float(lr),
                weight_decay=float(weight_decay),
                momentum=float(0.0 if momentum is None else momentum),
                alpha=float(0.99 if alpha is None else alpha),
                eps=float(1e-8 if eps is None else eps),
                centered=bool(centered),
            )
        return opt_cls(
            params,
            lr=float(lr),
            betas=(
                float(0.9 if beta1 is None else beta1),
                float(0.999 if beta2 is None else beta2),
            ),
            amsgrad=bool(amsgrad),
            eps=float(1e-8 if eps is None else eps),
            weight_decay=float(weight_decay),
        )
    raise ValueError(f"Unknown optimizer_name: {name!r}")


def _resolve_scheduler(
    *,
    torch,
    name: str | None,
    optimizer,
    epoch_num: int,
    milestones,
    step_size: int | None,
    gamma: float | None,
    t_max: int | None,
    eta_min: float | None,
    patience: int | None,
    factor: float | None,
    min_lr: float | None,
    cooldown: int | None,
    threshold: float | None,
    threshold_mode: str | None,
    eps: float | None,
):
    if name is None:
        return None

    key = str(name).lower().strip()
    if not key or key == "none":
        return None

    if key == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(step_size or 1),
            gamma=float(gamma if gamma is not None else 0.1),
        )
    if key == "multistep":
        if milestones is None:
            raise ValueError("scheduler_milestones is required when scheduler_name='multistep'")
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(v) for v in milestones],
            gamma=float(gamma if gamma is not None else 0.1),
        )
    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_max or max(1, epoch_num)),
            eta_min=float(eta_min if eta_min is not None else 0.0),
        )
    if key == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(gamma if gamma is not None else 0.95),
        )
    if key == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(10 if patience is None else patience),
            factor=float(0.1 if factor is None else factor),
            min_lr=float(0.0 if min_lr is None else min_lr),
            cooldown=int(0 if cooldown is None else cooldown),
            threshold=float(1e-4 if threshold is None else threshold),
            threshold_mode=str("rel" if threshold_mode is None else threshold_mode),
            eps=float(1e-8 if eps is None else eps),
        )
    raise ValueError(f"Unknown scheduler_name: {name!r}")


def _infer_batch_item_count(batch_data: Any) -> int | None:
    if isinstance(batch_data, dict):
        for value in batch_data.values():
            count = _infer_batch_item_count(value)
            if count is not None:
                return count
        return None

    if isinstance(batch_data, (list, tuple)):
        for value in batch_data:
            count = _infer_batch_item_count(value)
            if count is not None:
                return count
        return None

    shape = getattr(batch_data, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                return int(shape[0])
        except Exception:
            pass

    try:
        return int(len(batch_data))
    except Exception:
        return None


class BaseDeepLearningDetector(BaseDetector):
    """Base class for deep-learning detectors."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        preprocessing: bool = True,
        lr: float = 1e-3,
        epoch_num: int = 10,
        batch_size: int = 32,
        optimizer_name: str = "adam",
        criterion=None,
        criterion_name: str = "mse",
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination)
        self.preprocessing = bool(preprocessing)
        self.lr = float(lr)
        self.epoch_num = int(epoch_num)
        self.batch_size = int(batch_size)
        self.num_workers = 0
        self.shuffle_train = True
        self.drop_last = False
        self.pin_memory = False
        self.persistent_workers = False
        self.optimizer_name = str(optimizer_name)
        self.optimizer_momentum: float | None = None
        self.optimizer_nesterov = False
        self.optimizer_dampening: float | None = None
        self.adam_beta1: float | None = None
        self.adam_beta2: float | None = None
        self.adam_amsgrad: bool | None = None
        self.optimizer_eps: float | None = None
        self.rmsprop_alpha: float | None = None
        self.rmsprop_centered: bool | None = None
        self.weight_decay = 0.0
        self.scheduler_name: str | None = None
        self.scheduler_milestones: list[int] | None = None
        self.scheduler_step_size: int | None = None
        self.scheduler_gamma: float | None = None
        self.scheduler_t_max: int | None = None
        self.scheduler_eta_min: float | None = None
        self.scheduler_patience: int | None = None
        self.scheduler_factor: float | None = None
        self.scheduler_min_lr: float | None = None
        self.scheduler_cooldown: int | None = None
        self.scheduler_threshold: float | None = None
        self.scheduler_threshold_mode: str | None = None
        self.scheduler_eps: float | None = None
        self.criterion_name = str(criterion_name)
        self._custom_criterion = criterion
        self._criterion_name_resolved = str(self.criterion_name).lower()
        self.verbose = int(verbose)
        self.random_state = None if random_state is None else int(random_state)
        self._kwargs = dict(kwargs)
        self.max_steps: int | None = None
        self.early_stopping_patience: int | None = None
        self.early_stopping_min_delta: float = 0.0
        self.warmup_epochs: int | None = None
        self.warmup_start_factor: float = 0.1
        self.ema_enabled = False
        self.ema_decay = 0.999
        self.ema_start_epoch = 1
        self.training_epochs_completed_: int = 0
        self.training_steps_completed_: int = 0
        self.training_stop_reason_: str | None = None
        self.training_best_loss_: float | None = None
        self.training_loss_history_: list[float] = []
        self.training_epoch_time_history_: list[float] = []
        self.training_epoch_sample_counts_: list[int] = []
        self.training_ema_updates_: int = 0
        self.training_ema_applied_ = False
        self._ema_state: dict[str, Any] | None = None

        torch = _require_torch()
        self._torch = torch

        # Device handling: accept strings like "cpu"/"cuda" and store torch.device.
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(str(device))

        # Loss/criterion
        self.criterion = _resolve_criterion(
            torch=torch, criterion=criterion, criterion_name=self.criterion_name
        )

        # Optimizer is initialized in `training_prepare` once `self.model` exists.
        self.optimizer = None
        self.scheduler = None

        # Training-time metadata
        self.data_num: Optional[int] = None
        self.feature_size: Optional[int] = None

        # Fit-time normalization (optional)
        self.X_mean: Optional[np.ndarray] = None
        self.X_std: Optional[np.ndarray] = None
        self.training_lr_history_: list[float] = []
        self.training_last_lr_: float | None = None
        self._optimizer_base_lrs: list[float] = []

        # Seed for reproducibility when callers opt into global seeding.
        if self.random_state is not None:
            self._set_seed(self.random_state)

    # ------------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        torch = _require_torch()
        import os
        import random

        os.environ["PYTHONHASHSEED"] = str(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Subclass API (sklearn-like)
    def build_model(self):  # pragma: no cover
        raise NotImplementedError

    def training_forward(self, batch_data):  # pragma: no cover
        raise NotImplementedError

    def evaluating_forward(self, batch_data):  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    def training_prepare(self) -> None:
        torch = self._torch

        if self._custom_criterion is None:
            resolved_name = str(self.criterion_name).lower()
            if self._criterion_name_resolved != resolved_name:
                self.criterion = _resolve_criterion(
                    torch=torch,
                    criterion=None,
                    criterion_name=self.criterion_name,
                )
                self._criterion_name_resolved = resolved_name

        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError("Model is not built. Call build_model() before training_prepare().")

        if hasattr(model, "to"):
            self.model = model.to(self.device)

        if self.optimizer is None:
            params = getattr(self.model, "parameters", None)
            if not callable(params):
                raise RuntimeError("Model does not expose parameters(); cannot build optimizer.")
            self.optimizer = _resolve_optimizer(
                torch=torch,
                name=self.optimizer_name,
                params=self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.optimizer_momentum,
                nesterov=self.optimizer_nesterov,
                dampening=self.optimizer_dampening,
                beta1=self.adam_beta1,
                beta2=self.adam_beta2,
                amsgrad=bool(False if self.adam_amsgrad is None else self.adam_amsgrad),
                eps=self.optimizer_eps,
                alpha=self.rmsprop_alpha,
                centered=bool(False if self.rmsprop_centered is None else self.rmsprop_centered),
            )
        self._optimizer_base_lrs = [
            float(group["lr"]) for group in getattr(self.optimizer, "param_groups", [])
        ]
        if self.scheduler is None:
            self.scheduler = _resolve_scheduler(
                torch=torch,
                name=self.scheduler_name,
                optimizer=self.optimizer,
                epoch_num=self.epoch_num,
                milestones=self.scheduler_milestones,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
                t_max=self.scheduler_t_max,
                eta_min=self.scheduler_eta_min,
                patience=self.scheduler_patience,
                factor=self.scheduler_factor,
                min_lr=self.scheduler_min_lr,
                cooldown=self.scheduler_cooldown,
                threshold=self.scheduler_threshold,
                threshold_mode=self.scheduler_threshold_mode,
                eps=self.scheduler_eps,
            )

        if hasattr(self.model, "train"):
            self.model.train()

    def _apply_warmup_lr(self, epoch_index: int) -> None:
        if self.optimizer is None or not self._optimizer_base_lrs:
            return
        if self.warmup_epochs is None:
            return

        warmup_epochs = max(1, int(self.warmup_epochs))
        if epoch_index >= warmup_epochs:
            return

        start_factor = float(self.warmup_start_factor)
        if warmup_epochs == 1:
            factor = 1.0
        else:
            factor = start_factor + (1.0 - start_factor) * (
                float(epoch_index) / float(warmup_epochs - 1)
            )
        for group, base_lr in zip(self.optimizer.param_groups, self._optimizer_base_lrs):
            group["lr"] = float(base_lr) * float(factor)

    def _should_step_scheduler(self, epoch_index: int) -> bool:
        if self.scheduler is None:
            return False
        if self.warmup_epochs is None:
            return True
        return (epoch_index + 1) > max(1, int(self.warmup_epochs))

    def _step_scheduler(self, *, epoch_index: int, mean_loss: float | None) -> None:
        if not self._should_step_scheduler(epoch_index):
            return
        if self.scheduler is None:
            return
        if str(self.scheduler_name).lower() == "plateau":
            if mean_loss is None:
                return
            self.scheduler.step(float(mean_loss))
            return
        self.scheduler.step()

    def _should_update_ema(self, *, epoch_index: int) -> bool:
        if not bool(self.ema_enabled):
            return False
        return (epoch_index + 1) >= max(1, int(self.ema_start_epoch))

    def _update_ema_state(self) -> None:
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "state_dict"):
            return

        state_dict = model.state_dict()
        if self._ema_state is None:
            self._ema_state = {}
            for key, value in state_dict.items():
                detach = getattr(value, "detach", None)
                if callable(detach):
                    self._ema_state[str(key)] = value.detach().clone()
                else:
                    self._ema_state[str(key)] = value
            return

        decay = float(self.ema_decay)
        for key, value in state_dict.items():
            shadow = self._ema_state.get(str(key), None)
            detach = getattr(value, "detach", None)
            if not callable(detach):
                self._ema_state[str(key)] = value
                continue

            current = value.detach()
            if shadow is None:
                self._ema_state[str(key)] = current.clone()
                continue

            is_floating_point = getattr(current, "is_floating_point", None)
            if callable(is_floating_point) and current.is_floating_point():
                shadow.mul_(decay).add_(current, alpha=1.0 - decay)
            else:
                shadow.copy_(current)

    def _apply_ema_state(self) -> bool:
        model = getattr(self, "model", None)
        if not bool(self.ema_enabled) or model is None or self._ema_state is None:
            return False

        load_state_dict = getattr(model, "load_state_dict", None)
        if not callable(load_state_dict):
            return False

        load_state_dict(self._ema_state, strict=False)
        return True

    def train(self, train_loader) -> None:  # noqa: ANN001
        reporter = get_active_train_progress_reporter()
        self.training_epochs_completed_ = 0
        self.training_steps_completed_ = 0
        self.training_stop_reason_ = None
        self.training_best_loss_ = None
        self.training_loss_history_ = []
        self.training_epoch_time_history_ = []
        self.training_epoch_sample_counts_ = []
        self.training_lr_history_ = []
        self.training_last_lr_ = None
        self.training_ema_updates_ = 0
        self.training_ema_applied_ = False
        self._ema_state = None

        max_steps = None if self.max_steps is None else max(1, int(self.max_steps))
        patience = (
            None
            if self.early_stopping_patience is None
            else max(1, int(self.early_stopping_patience))
        )
        min_delta = float(self.early_stopping_min_delta)
        stale_epochs = 0
        elapsed_s = 0.0

        for _epoch in range(self.epoch_num):
            self._apply_warmup_lr(_epoch)
            if self.optimizer is not None and getattr(self.optimizer, "param_groups", None):
                self.training_lr_history_.append(float(self.optimizer.param_groups[0]["lr"]))
            losses: list[float] = []
            epoch_start = time.perf_counter()
            epoch_sample_count = 0
            for batch_data in train_loader:
                batch_items = _infer_batch_item_count(batch_data)
                if batch_items is not None:
                    epoch_sample_count += int(batch_items)
                loss = self.training_forward(batch_data)
                # training_forward may return floats or tuples; only require float for now.
                losses.append(_coerce_loss_float(loss))
                if self._should_update_ema(epoch_index=_epoch):
                    self._update_ema_state()
                    self.training_ema_updates_ += 1
                self.training_steps_completed_ += 1
                if max_steps is not None and self.training_steps_completed_ >= max_steps:
                    self.training_stop_reason_ = "max_steps"
                    break

            if self.verbose >= 2:
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                logger.info("Epoch %d/%d - loss=%.6f", _epoch + 1, self.epoch_num, mean_loss)

            mean_loss_epoch: float | None = None
            if losses:
                epoch_s = float(time.perf_counter() - epoch_start)
                elapsed_s += epoch_s
                mean_loss = float(np.mean(losses))
                mean_loss_epoch = mean_loss
                self.training_loss_history_.append(mean_loss)
                self.training_epoch_time_history_.append(epoch_s)
                if epoch_sample_count > 0:
                    train_items = int(epoch_sample_count)
                elif self.data_num is not None:
                    train_items = int(self.data_num)
                else:
                    train_items = 0
                self.training_epoch_sample_counts_.append(train_items)
                metrics: dict[str, float] = {"loss": float(mean_loss)}
                if self.training_lr_history_:
                    metrics["lr"] = float(self.training_lr_history_[-1])
                metrics["epoch_s"] = epoch_s
                metrics["elapsed_s"] = elapsed_s
                metrics["eta_s"] = float(max(self.epoch_num - int(_epoch + 1), 0)) * epoch_s
                metrics["train_items"] = float(train_items)
                if epoch_s > 0.0:
                    metrics["items_per_s"] = float(train_items) / epoch_s
                reporter.on_training_epoch(
                    epoch=int(_epoch + 1),
                    total_epochs=int(self.epoch_num),
                    metrics=metrics,
                    live=True,
                )
                best_loss = self.training_best_loss_
                if best_loss is None or mean_loss < (best_loss - min_delta):
                    self.training_best_loss_ = mean_loss
                    stale_epochs = 0
                elif patience is not None:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        self.training_stop_reason_ = "early_stopping"

            self.training_epochs_completed_ = _epoch + 1
            self.epoch_update()
            self._step_scheduler(epoch_index=_epoch, mean_loss=mean_loss_epoch)
            if self.training_stop_reason_ is not None:
                break

        if self.optimizer is not None and getattr(self.optimizer, "param_groups", None):
            self.training_last_lr_ = float(self.optimizer.param_groups[0]["lr"])
        self.training_ema_applied_ = self._apply_ema_state()
        if self.training_stop_reason_ is None:
            self.training_stop_reason_ = "completed"

    def epoch_update(self) -> None:
        return None

    # ------------------------------------------------------------------
    def evaluating_prepare(self) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "eval"):
            model.eval()

    def evaluate(self, data_loader):  # noqa: ANN001
        self.evaluating_prepare()
        outs: list[np.ndarray] = []
        torch = self._torch
        with torch.no_grad():
            for batch_data in data_loader:
                out = self.evaluating_forward(batch_data)
                if isinstance(out, np.ndarray):
                    arr = out
                else:
                    detach = getattr(out, "detach", None)
                    if callable(detach):
                        arr = out.detach().cpu().numpy()
                    else:
                        arr = np.asarray(out)
                arr = np.asarray(arr)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                outs.append(arr)

        if not outs:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(outs, axis=0)

    def decision_function(self, X, batch_size: Optional[int] = None):  # noqa: ANN001, ANN201
        torch = self._torch
        x_array = np.asarray(X, dtype=np.float32)
        if x_array.ndim != 2:
            raise ValueError(f"X must be 2D array-like. Got shape {x_array.shape!r}.")

        if self.preprocessing:
            if self.X_mean is None or self.X_std is None:
                raise RuntimeError("Model is not fitted. Missing preprocessing statistics.")
            x_array = (x_array - self.X_mean) / self.X_std

        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        dummy_y = torch.zeros((x_tensor.shape[0],), dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(x_tensor, dummy_y)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(batch_size) if batch_size is not None else int(self.batch_size),
            shuffle=False,
            drop_last=False,
        )

        scores = self.evaluate(loader)
        return self.decision_function_update(scores)

    def decision_function_update(self, anomaly_scores):  # noqa: ANN001, ANN201
        return anomaly_scores

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        x_array = np.asarray(X, dtype=np.float32)
        if x_array.ndim != 2:
            raise ValueError(f"X must be 2D array-like. Got shape {x_array.shape!r}.")

        self._set_n_classes(y)
        self.data_num = int(x_array.shape[0])
        self.feature_size = int(x_array.shape[1])

        # Optional feature standardization.
        if self.preprocessing:
            self.X_mean = np.mean(x_array, axis=0)
            self.X_std = np.std(x_array, axis=0)
            self.X_std = np.where(self.X_std <= 1e-12, 1.0, self.X_std)
            x_array = (x_array - self.X_mean) / self.X_std

        # Build + train
        self.model = self.build_model()
        self.training_prepare()

        torch = self._torch
        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        dummy_y = torch.zeros((x_tensor.shape[0],), dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(x_tensor, dummy_y)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(self.batch_size),
            shuffle=bool(self.shuffle_train),
            num_workers=int(self.num_workers),
            drop_last=bool(self.drop_last),
            pin_memory=bool(self.pin_memory),
            persistent_workers=bool(self.persistent_workers) and int(self.num_workers) > 0,
        )

        self.train(loader)

        self.decision_scores_ = np.asarray(self.decision_function(x_array), dtype=np.float64)
        self._process_decision_scores()
        return self
