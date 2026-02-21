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

from .anomalydino import (
    PatchEmbedder,
    VisionAnomalyDINO,
    _aggregate_patch_scores,
    _reshape_patch_scores,
)

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
        embedder: Optional[PatchEmbedder] = None,
        text_features_normal: Optional[NDArray] = None,
        text_features_anomaly: Optional[NDArray] = None,
        contamination: float = 0.1,
        aggregation_method: str = "topk_mean",
        aggregation_topk: float = 0.01,
        mode: _PromptScoreMode = "diff",
        open_clip_module=None,
        **kwargs: Any,
    ) -> None:
        self.embedder = embedder
        self.text_features_normal = text_features_normal
        self.text_features_anomaly = text_features_anomaly
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5). Got {self.contamination}."
            )

        self.aggregation_method = str(aggregation_method)
        self.aggregation_topk = float(aggregation_topk)
        self.mode = mode
        self._kwargs = dict(kwargs)
        self._open_clip = open_clip_module

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

        # Ensure missing optional deps raise a clean ImportError, but allow unit
        # tests to inject a fake embedder + text features without OpenCLIP.
        if self.embedder is None or self.text_features_normal is None or self.text_features_anomaly is None:
            _require_open_clip(open_clip_module)
            raise NotImplementedError(
                "Automatic OpenCLIP model loading and text prompt encoding is not implemented yet. "
                "Pass `embedder=`, `text_features_normal=`, and `text_features_anomaly=`."
            )

    def _embed(self, image_path: str) -> tuple[NDArray, tuple[int, int], tuple[int, int]]:
        if self.embedder is None:  # pragma: no cover - guarded by __init__
            raise RuntimeError("embedder is required")

        patch_embeddings, grid_shape, original_size = self.embedder.embed(image_path)
        patch_embeddings_np = np.asarray(patch_embeddings, dtype=np.float32)
        if patch_embeddings_np.ndim != 2:
            raise ValueError(
                f"Expected 2D patch embeddings, got shape {patch_embeddings_np.shape}"
            )

        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if patch_embeddings_np.shape[0] != grid_h * grid_w:
            raise ValueError(
                "Patch embedding count does not match grid shape. "
                f"Got {patch_embeddings_np.shape[0]} patches for grid {grid_h}x{grid_w}."
            )

        original_h, original_w = int(original_size[0]), int(original_size[1])
        if original_h <= 0 or original_w <= 0:
            raise ValueError(f"Invalid original_size: {original_size}")

        return patch_embeddings_np, (grid_h, grid_w), (original_h, original_w)

    def fit(self, X, y=None):
        paths = list(X)
        if not paths:
            raise ValueError("X must contain at least one training image path.")

        self.decision_scores_ = np.asarray(self.decision_function(paths), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def decision_function(self, X):
        if self.text_features_normal is None or self.text_features_anomaly is None:
            raise RuntimeError("text_features_normal/text_features_anomaly are required")

        paths = list(X)
        scores = np.zeros(len(paths), dtype=np.float64)
        for i, path in enumerate(paths):
            patch_embeddings, _grid_shape, _original_size = self._embed(path)
            patch_scores = _prompt_patch_scores(
                patch_embeddings,
                text_features_normal=self.text_features_normal,
                text_features_anomaly=self.text_features_anomaly,
                mode=self.mode,
            )
            scores[i] = _aggregate_patch_scores(
                patch_scores,
                method=self.aggregation_method,
                topk=self.aggregation_topk,
            )
        return scores

    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image_path: str) -> NDArray:
        if self.text_features_normal is None or self.text_features_anomaly is None:
            raise RuntimeError("text_features_normal/text_features_anomaly are required")

        patch_embeddings, grid_shape, original_size = self._embed(image_path)
        patch_scores = _prompt_patch_scores(
            patch_embeddings,
            text_features_normal=self.text_features_normal,
            text_features_anomaly=self.text_features_anomaly,
            mode=self.mode,
        )
        patch_grid = _reshape_patch_scores(patch_scores, grid_h=grid_shape[0], grid_w=grid_shape[1])

        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required to upsample anomaly maps.\n"
                "Install it via:\n  pip install 'opencv-python'\n"
                f"Original error: {exc}"
            ) from exc

        original_h, original_w = original_size
        upsampled = cv2.resize(
            np.asarray(patch_grid, dtype=np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.asarray(upsampled, dtype=np.float32)

    def predict_anomaly_map(self, X):
        paths = list(X)
        maps = [self.get_anomaly_map(path) for path in paths]
        return np.stack(maps)


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
