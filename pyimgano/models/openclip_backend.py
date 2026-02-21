from __future__ import annotations

"""OpenCLIP backend model skeletons.

This module intentionally has **no hard dependency** on `open_clip_torch`. The
OpenCLIP Python module (`open_clip`) is only imported at runtime when a model is
constructed, via `pyimgano.utils.optional_deps.require`.

The goal is to make `import pyimgano.models` safe even when optional deps are
not installed, while still registering model names so they can be discovered.
"""

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .anomalydino import PatchEmbedder, VisionAnomalyDINO

from pyimgano.utils.optional_deps import require

from .registry import register_model


def _require_open_clip(open_clip_module=None):
    if open_clip_module is not None:
        return open_clip_module
    return require("open_clip", extra="clip", purpose="OpenCLIP detectors")


def _l2_normalize(x: NDArray, *, axis: int, eps: float = 1e-12) -> NDArray:
    x_np = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x_np, axis=axis, keepdims=True)
    denom = np.maximum(denom, float(eps))
    return x_np / denom


_PromptScoreMode = Literal["diff", "ratio"]


def _prompt_patch_scores(
    patch_embeddings: NDArray,
    *,
    text_features_normal: NDArray,
    text_features_anomaly: NDArray,
    mode: _PromptScoreMode = "diff",
    eps: float = 1e-6,
) -> NDArray:
    """Compute per-patch anomaly scores from OpenCLIP-style embeddings.

    This helper is intentionally NumPy-only so it can be unit tested without
    `torch`/`open_clip` installed.
    """

    patches = np.asarray(patch_embeddings, dtype=np.float32)
    if patches.ndim != 2:
        raise ValueError(f"patch_embeddings must be 2D (num_patches, dim). Got {patches.shape}.")

    normal = np.asarray(text_features_normal, dtype=np.float32).reshape(-1)
    anomaly = np.asarray(text_features_anomaly, dtype=np.float32).reshape(-1)
    if normal.ndim != 1 or anomaly.ndim != 1:
        raise ValueError("text_features_normal/anomaly must be 1D feature vectors.")
    if normal.shape != anomaly.shape:
        raise ValueError(
            "text_features_normal and text_features_anomaly must have the same shape. "
            f"Got {normal.shape} vs {anomaly.shape}."
        )
    if patches.shape[1] != normal.shape[0]:
        raise ValueError(
            "Embedding dimension mismatch between patches and text features. "
            f"Got patches dim={patches.shape[1]} vs text dim={normal.shape[0]}."
        )

    patches_norm = _l2_normalize(patches, axis=1, eps=float(eps))
    normal_norm = _l2_normalize(normal, axis=0, eps=float(eps)).reshape(-1)
    anomaly_norm = _l2_normalize(anomaly, axis=0, eps=float(eps)).reshape(-1)

    sim_normal = patches_norm @ normal_norm
    sim_anomaly = patches_norm @ anomaly_norm

    mode_lower = str(mode).lower()
    if mode_lower == "diff":
        return np.asarray(sim_anomaly - sim_normal, dtype=np.float32)
    if mode_lower == "ratio":
        eps_float = float(eps)
        return np.asarray((sim_anomaly + eps_float) / (sim_normal + eps_float), dtype=np.float32)

    raise ValueError(f"Unknown mode: {mode}. Choose from: diff, ratio")


@register_model(
    "vision_openclip_promptscore",
    tags=("vision", "openclip", "backend"),
    metadata={
        "description": "OpenCLIP prompt scoring detector (requires pyimgano[clip])",
        "backend": "openclip",
    },
)
class VisionOpenCLIPPromptScore:
    """Skeleton for an OpenCLIP prompt-score based vision detector.

    Parameters
    ----------
    open_clip_module : optional
        Dependency injection hook. When provided, avoids importing `open_clip`.
        Intended for unit tests and advanced callers.
    """

    def __init__(
        self,
        *,
        open_clip_module=None,
        **kwargs: Any,
    ) -> None:
        # Keep the runtime dependency check lazy so importing this module never
        # requires `open_clip` to be installed.
        self._open_clip = _require_open_clip(open_clip_module)
        self._kwargs = dict(kwargs)

    def fit(self, X, y=None):  # pragma: no cover - skeleton API
        raise NotImplementedError("OpenCLIP backend skeleton: fit() not implemented yet.")

    def decision_function(self, X):  # pragma: no cover - skeleton API
        raise NotImplementedError(
            "OpenCLIP backend skeleton: decision_function() not implemented yet."
        )


@register_model(
    "vision_openclip_patchknn",
    tags=("vision", "openclip", "backend", "knn"),
    metadata={
        "description": "OpenCLIP patch embedding + kNN detector (requires pyimgano[clip])",
        "backend": "openclip",
    },
)
class VisionOpenCLIPPatchKNN:
    """Skeleton for an OpenCLIP patch embedding + kNN detector.

    Parameters
    ----------
    open_clip_module : optional
        Dependency injection hook. When provided, avoids importing `open_clip`.
    """

    def __init__(
        self,
        *,
        open_clip_module=None,
        embedder: Optional[PatchEmbedder] = None,
        knn_index: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # This detector is implemented as an AnomalyDINO-style patch-kNN model.
        # To keep tests lightweight, callers can inject a pure-numpy embedder.
        if embedder is None:
            _require_open_clip(open_clip_module)
            raise NotImplementedError(
                "Default OpenCLIP patch embedder not implemented yet. "
                "Pass an `embedder=` that implements `embed(image_path)`."
            )

        self._open_clip = open_clip_module
        self._knn_index = knn_index
        self._kwargs = dict(kwargs)
        self._core = VisionAnomalyDINO(embedder=embedder, **kwargs)

    def fit(self, X, y=None):  # pragma: no cover - skeleton API
        self._core.fit(X, y=y)
        return self

    def decision_function(self, X):  # pragma: no cover - skeleton API
        return self._core.decision_function(X)

    def predict(self, X):
        return self._core.predict(X)

    def get_anomaly_map(self, image_path: str):
        return self._core.get_anomaly_map(image_path)

    def predict_anomaly_map(self, X):
        return self._core.predict_anomaly_map(X)

    @property
    def decision_scores_(self):
        return self._core.decision_scores_

    @property
    def threshold_(self):
        return self._core.threshold_
