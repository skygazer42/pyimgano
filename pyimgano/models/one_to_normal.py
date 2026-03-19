from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .registry import register_model


def _as_float_array(image: Any) -> NDArray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected image-like array with ndim >= 2. Got shape {arr.shape}.")
    return arr


def _normalize_with_backend(normalizer: Any, image: NDArray) -> NDArray:
    if normalizer is None:
        raise ValueError("normalizer is required for vision_one_to_normal.")

    if hasattr(normalizer, "normalize"):
        normalized = normalizer.normalize(image)
    elif callable(normalizer):
        normalized = normalizer(image)
    else:
        raise TypeError("normalizer must be callable or implement .normalize(image).")

    normalized_arr = np.asarray(normalized, dtype=np.float32)
    if normalized_arr.shape != image.shape:
        raise ValueError(
            "Normalized output must match the input shape. "
            f"Got {normalized_arr.shape} vs {image.shape}."
        )
    return normalized_arr


@register_model(
    "vision_one_to_normal",
    tags=("vision", "deep", "reconstruction", "few-shot", "pixel_map", "numpy", "one_to_normal"),
    metadata={
        "description": "One-to-Normal family adapter with residual scoring and residual maps.",
        "paper": "One-to-Normal",
        "year": 2025,
        "supervision": "few-shot",
    },
)
class VisionOneToNormal:
    def __init__(
        self,
        *,
        normalizer: Any = None,
        contamination: float = 0.1,
    ) -> None:
        self.normalizer = normalizer
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None
        self.support_residual_mean_: float | None = None

    def get_anomaly_map(self, image: Any) -> NDArray:
        image_arr = _as_float_array(image)
        normalized = _normalize_with_backend(self.normalizer, image_arr)
        residual = np.abs(image_arr - normalized)
        if residual.ndim == 2:
            return residual.astype(np.float32, copy=False)
        return np.mean(residual, axis=-1).astype(np.float32, copy=False)

    def predict_anomaly_map(self, x: Iterable[Any]) -> NDArray:
        items = list(x)
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(item) for item in items]
        return np.stack(maps, axis=0).astype(np.float32, copy=False)

    def decision_function(self, x):
        items = list(x)
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, item in enumerate(items):
            scores[i] = float(np.mean(self.get_anomaly_map(item)))
        return scores

    def fit(self, x, _y=None):
        items = list(x)
        if not items:
            raise ValueError("X must contain at least one support image.")
        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.support_residual_mean_ = float(np.mean(self.decision_scores_))
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def predict(self, x):
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = np.asarray(self.decision_function(x), dtype=np.float64)
        return (scores > float(self.threshold_)).astype(np.int64)
