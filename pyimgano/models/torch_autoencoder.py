# -*- coding: utf-8 -*-
"""Torch MLP Autoencoder (feature-matrix) anomaly detector.

This provides a lightweight, industrially useful bridge:

  deep embeddings (or any feature vectors) -> MLP autoencoder -> recon error score

We expose:
- `core_torch_autoencoder`: operates on 2D feature matrices (N,D)
- `vision_torch_autoencoder`: vision wrapper using feature extractor registry

Score convention: higher reconstruction error => more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from pyimgano.utils.torch_activations import get_activation_by_name

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


def _require_torch():  # noqa: ANN001, ANN201 - optional-dep boundary
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "core_torch_autoencoder requires torch. Install it via:\n  pip install 'torch'\n"
            f"Original error: {exc}"
        ) from exc
    return torch


@dataclass
class _AEConfig:
    hidden_dims: tuple[int, ...]
    activation: str
    dropout: float


class _MLPAutoencoder:  # backend used by core_* wrapper
    def __init__(
        self,
        *,
        contamination: float = 0.1,  # kept for signature consistency
        hidden_dims: Sequence[int] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        preprocessing: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.contamination = float(contamination)
        self.cfg = _AEConfig(
            hidden_dims=tuple(int(d) for d in list(hidden_dims)),
            activation=str(activation),
            dropout=float(dropout),
        )
        if len(self.cfg.hidden_dims) < 1:
            raise ValueError("hidden_dims must be a non-empty sequence of ints")
        if any(int(d) < 1 for d in self.cfg.hidden_dims):
            raise ValueError("hidden_dims values must be >= 1")
        if not (0.0 <= float(self.cfg.dropout) < 1.0):
            raise ValueError("dropout must be in [0,1)")

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = str(device)
        self.preprocessing = bool(preprocessing)
        self.random_state = random_state

        self.scaler_: StandardScaler | None = None
        self.model_ = None
        self.n_features_in_: int | None = None
        self.decision_scores_: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _make_device(self):
        torch = _require_torch()
        dev = str(self.device).strip().lower()
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return torch.device(dev)

    def _set_seed(self) -> None:
        if self.random_state is None:
            return
        torch = _require_torch()
        seed = int(self.random_state)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_model(self, n_features: int):
        torch = _require_torch()
        import torch.nn as nn

        dims = [int(n_features)] + [int(d) for d in self.cfg.hidden_dims]
        act = get_activation_by_name(str(self.cfg.activation))

        enc_layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(act)
            if float(self.cfg.dropout) > 0.0:
                enc_layers.append(nn.Dropout(float(self.cfg.dropout)))

        dec_dims = list(reversed(dims))
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(act)
                if float(self.cfg.dropout) > 0.0:
                    dec_layers.append(nn.Dropout(float(self.cfg.dropout)))

        class _AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(*enc_layers)
                self.decoder = nn.Sequential(*dec_layers)

            def forward(self, x):  # noqa: ANN001, ANN201
                z = self.encoder(x)
                return self.decoder(z)

        return _AE()

    # ------------------------------------------------------------------
    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_np = check_array(X, ensure_2d=True, dtype=np.float64)
        n, d = int(X_np.shape[0]), int(X_np.shape[1])
        if n == 0:
            raise ValueError("Training set cannot be empty")
        self.n_features_in_ = int(d)

        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_train = self.scaler_.fit_transform(X_np)
        else:
            X_train = X_np

        self._set_seed()
        torch = _require_torch()
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        dev = self._make_device()
        model = self._build_model(d).to(dev)
        model.train()

        x_t = torch.as_tensor(X_train, dtype=torch.float32)
        ds = TensorDataset(x_t)
        bs = max(1, int(self.batch_size))
        loader = DataLoader(ds, batch_size=bs, shuffle=True)

        opt = torch.optim.Adam(
            model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay)
        )

        epochs = max(0, int(self.epochs))
        for _ in range(epochs):
            for (xb,) in loader:
                xb = xb.to(dev)
                opt.zero_grad(set_to_none=True)
                recon = model(xb)
                loss = F.mse_loss(recon, xb, reduction="mean")
                loss.backward()
                opt.step()

        self.model_ = model.eval()
        self.decision_scores_ = np.asarray(self.decision_function(X_np), dtype=np.float64).reshape(-1)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        if self.model_ is None or self.n_features_in_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X_np = check_array(X, ensure_2d=True, dtype=np.float64)
        if int(X_np.shape[1]) != int(self.n_features_in_):
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}")

        if self.preprocessing and self.scaler_ is not None:
            X_eval = self.scaler_.transform(X_np)
        else:
            X_eval = X_np

        torch = _require_torch()
        import torch.nn.functional as F

        dev = self._make_device()
        model = self.model_.to(dev)
        model.eval()

        with torch.no_grad():
            x_t = torch.as_tensor(X_eval, dtype=torch.float32, device=dev)
            recon = model(x_t)
            # Per-sample mean squared error.
            err = F.mse_loss(recon, x_t, reduction="none")
            err = err.mean(dim=1)
            return err.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)


@register_model(
    "core_torch_autoencoder",
    tags=("deep", "core", "features", "torch", "autoencoder", "reconstruction"),
    metadata={
        "description": "Core torch MLP autoencoder on feature matrices (reconstruction error)",
        "input": "features",
    },
)
class CoreTorchAutoencoder(CoreFeatureDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        hidden_dims: Sequence[int] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        preprocessing: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self._backend_kwargs = dict(
            contamination=float(contamination),
            hidden_dims=tuple(int(d) for d in list(hidden_dims)),
            activation=str(activation),
            dropout=float(dropout),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            device=str(device),
            preprocessing=bool(preprocessing),
            random_state=random_state,
        )
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return _MLPAutoencoder(**self._backend_kwargs)


@register_model(
    "vision_torch_autoencoder",
    tags=("vision", "deep", "torch", "autoencoder", "reconstruction"),
    metadata={
        "description": "Vision wrapper for core_torch_autoencoder (feature extractor + torch AE core)",
    },
)
class VisionTorchAutoencoder(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        hidden_dims: Sequence[int] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        preprocessing: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self._detector_kwargs = dict(
            contamination=float(contamination),
            hidden_dims=tuple(int(d) for d in list(hidden_dims)),
            activation=str(activation),
            dropout=float(dropout),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            device=str(device),
            preprocessing=bool(preprocessing),
            random_state=random_state,
        )
        super().__init__(contamination=float(contamination), feature_extractor=feature_extractor)

    def _build_detector(self):
        return _MLPAutoencoder(**self._detector_kwargs)

    def fit(self, X: Iterable, y=None):  # noqa: ANN001, ANN201
        return super().fit(X, y=y)

    def decision_function(self, X):  # noqa: ANN001, ANN201
        return super().decision_function(X)

@register_model(
    "vision_embedding_torch_autoencoder",
    tags=("vision", "deep", "torch", "autoencoder", "embeddings", "pipeline"),
    metadata={
        "description": (
            "Industrial preset: deep embeddings (torchvision_backbone) -> torch MLP autoencoder -> recon error"
        ),
    },
)
class VisionEmbeddingTorchAutoencoder(BaseVisionDetector):
    """Opinionated, stable defaults for 'embeddings -> torch AE'."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        device: str = "cpu",
        embedding_extractor: str | object = "torchvision_backbone",
        embedding_kwargs: Mapping[str, object] | None = None,
        ae_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        dev = str(device)

        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = {
                "backbone": "resnet18",
                "pretrained": False,
                "pool": "avg",
                "device": dev,
            }

        if embedding_kwargs:
            feature_extractor = {"name": str(embedding_extractor), "kwargs": dict(embedding_kwargs)}
        else:
            feature_extractor = embedding_extractor

        # Stability-first defaults: moderate AE size, preprocessing on, deterministic seed-friendly.
        default_ae = {
            "hidden_dims": (64, 32),
            "activation": "relu",
            "dropout": 0.0,
            "epochs": 30,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "device": dev,
            "preprocessing": True,
            "random_state": 0,
        }
        merged_ae = dict(default_ae)
        if ae_kwargs is not None:
            merged_ae.update(dict(ae_kwargs))

        self._detector_kwargs = dict(merged_ae)
        super().__init__(contamination=float(contamination), feature_extractor=feature_extractor)

    def _build_detector(self):
        return _MLPAutoencoder(**self._detector_kwargs)


__all__ = ["CoreTorchAutoencoder", "VisionTorchAutoencoder", "VisionEmbeddingTorchAutoencoder"]
