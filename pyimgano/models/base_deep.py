# -*- coding: utf-8 -*-
"""Native deep-learning detector base contract for pyimgano.

This provides a minimal, PyOD-like deep detector interface used by
`pyimgano.models.baseCv.BaseVisionDeepDetector` and a few detectors that
implement `build_model/training_forward/evaluating_forward`.

Key responsibilities:
- contamination-based thresholding via `BaseDetector`
- a simple training loop over a torch DataLoader
- a simple evaluation loop producing per-sample anomaly scores

The API is intentionally small and geared toward the needs of this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .base_detector import BaseDetector


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for deep-learning detectors. Install it via:\n"
            "  pip install 'torch'\n"
            f"Original error: {exc}"
        ) from exc
    return torch


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


def _resolve_optimizer(*, torch, name: str, params, lr: float):
    key = str(name).lower()
    if key in {"adam", "adamw", "sgd", "rmsprop"}:
        opt_cls = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }[key]
        if key == "sgd":
            return opt_cls(params, lr=float(lr), momentum=0.9)
        return opt_cls(params, lr=float(lr))
    raise ValueError(f"Unknown optimizer_name: {name!r}")


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
        random_state: int = 42,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination)
        self.preprocessing = bool(preprocessing)
        self.lr = float(lr)
        self.epoch_num = int(epoch_num)
        self.batch_size = int(batch_size)
        self.optimizer_name = str(optimizer_name)
        self.criterion_name = str(criterion_name)
        self.verbose = int(verbose)
        self.random_state = int(random_state)
        self._kwargs = dict(kwargs)

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

        # Training-time metadata
        self.data_num: Optional[int] = None
        self.feature_size: Optional[int] = None

        # Fit-time normalization (optional)
        self.X_mean: Optional[np.ndarray] = None
        self.X_std: Optional[np.ndarray] = None

        # Seed for reproducibility
        self._set_seed(self.random_state)

    # ------------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        torch = _require_torch()
        import random
        import os

        os.environ["PYTHONHASHSEED"] = str(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Subclass API (PyOD-like)
    def build_model(self):  # pragma: no cover
        raise NotImplementedError

    def training_forward(self, batch_data):  # pragma: no cover
        raise NotImplementedError

    def evaluating_forward(self, batch_data):  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    def training_prepare(self) -> None:
        torch = self._torch

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
            )

        if hasattr(self.model, "train"):
            self.model.train()

    def train(self, train_loader) -> None:  # noqa: ANN001
        for _epoch in range(self.epoch_num):
            losses: list[float] = []
            for batch_data in train_loader:
                loss = self.training_forward(batch_data)
                # training_forward may return floats or tuples; only require float for now.
                losses.append(float(loss))

            if self.verbose >= 2:
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                print(f"Epoch {_epoch + 1}/{self.epoch_num} - loss={mean_loss:.6f}")

            self.epoch_update()

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
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array-like. Got shape {X_arr.shape!r}.")

        if self.preprocessing:
            if self.X_mean is None or self.X_std is None:
                raise RuntimeError("Model is not fitted. Missing preprocessing statistics.")
            X_arr = (X_arr - self.X_mean) / self.X_std

        tensor_X = torch.tensor(X_arr, dtype=torch.float32)
        dummy_y = torch.zeros((tensor_X.shape[0],), dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(tensor_X, dummy_y)
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
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array-like. Got shape {X_arr.shape!r}.")

        self._set_n_classes(y)
        self.data_num = int(X_arr.shape[0])
        self.feature_size = int(X_arr.shape[1])

        # Optional feature standardization.
        if self.preprocessing:
            self.X_mean = np.mean(X_arr, axis=0)
            self.X_std = np.std(X_arr, axis=0)
            self.X_std = np.where(self.X_std <= 1e-12, 1.0, self.X_std)
            X_arr = (X_arr - self.X_mean) / self.X_std

        # Build + train
        self.model = self.build_model()
        self.training_prepare()

        torch = self._torch
        tensor_X = torch.tensor(X_arr, dtype=torch.float32)
        dummy_y = torch.zeros((tensor_X.shape[0],), dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(tensor_X, dummy_y)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            drop_last=False,
        )

        self.train(loader)

        self.decision_scores_ = np.asarray(self.decision_function(X_arr), dtype=np.float64)
        self._process_decision_scores()
        return self
