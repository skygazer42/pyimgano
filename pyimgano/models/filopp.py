from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .registry import register_model


def _as_patch_array(value: Any) -> NDArray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D patch embedding array. Got shape {arr.shape}.")
    return arr


def _encode_with_vlm(vlm_backend: Any, image: Any) -> tuple[NDArray, tuple[int, int]]:
    if vlm_backend is None:
        raise ValueError("vlm_backend is required for vision_filopp.")

    if hasattr(vlm_backend, "encode_image"):
        encoded = vlm_backend.encode_image(image)
    elif callable(vlm_backend):
        encoded = vlm_backend(image)
    else:
        raise TypeError("vlm_backend must be callable or implement .encode_image(image).")

    if isinstance(encoded, tuple):
        if len(encoded) < 2:
            raise ValueError("vlm_backend tuple output must be (patches, grid_shape) or richer.")
        patches, grid_shape = encoded[0], encoded[1]
        grid = (int(grid_shape[0]), int(grid_shape[1]))
        return _as_patch_array(patches), grid

    patches = _as_patch_array(encoded)
    return patches, (patches.shape[0], 1)


class _PromptLocalizationBackend:
    def fit(self, support_patch_sets: list[NDArray]):
        self.normal_prompt_ = np.mean(np.concatenate(support_patch_sets, axis=0), axis=0).astype(
            np.float32,
            copy=False,
        )
        return self

    def score(self, patch_embeddings: NDArray) -> tuple[float, NDArray]:
        delta = np.asarray(patch_embeddings, dtype=np.float32) - self.normal_prompt_[None, :]
        patch_scores = np.linalg.norm(delta, axis=1).astype(np.float32, copy=False)
        return float(np.max(patch_scores)), patch_scores


@register_model(
    "vision_filopp",
    tags=("vision", "deep", "clip", "pixel_map", "zero-shot", "numpy", "filopp"),
    metadata={
        "description": "FiLo++ family adapter with VLM patch encoding and localization-head scoring.",
        "paper": "FiLo++",
        "year": 2025,
        "supervision": "zero-shot",
    },
)
class VisionFiLoPP:
    def __init__(
        self,
        *,
        vlm_backend: Any = None,
        localization_backend: Any = None,
        contamination: float = 0.1,
    ) -> None:
        self.vlm_backend = vlm_backend
        self.localization_backend = (
            localization_backend if localization_backend is not None else _PromptLocalizationBackend()
        )
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None

    def fit(self, X, y=None):
        items = list(X)
        if not items:
            raise ValueError("X must contain at least one support sample.")
        support_patch_sets = [_encode_with_vlm(self.vlm_backend, item)[0] for item in items]
        if not hasattr(self.localization_backend, "fit") or not hasattr(self.localization_backend, "score"):
            raise TypeError(
                "localization_backend must implement .fit(support_patch_sets) and .score(patch_embeddings)."
            )
        self.localization_backend.fit(support_patch_sets)
        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _score_item(self, item: Any) -> tuple[float, NDArray, tuple[int, int]]:
        patches, grid_shape = _encode_with_vlm(self.vlm_backend, item)
        image_score, patch_scores = self.localization_backend.score(patches)
        patch_scores_arr = np.asarray(patch_scores, dtype=np.float32).reshape(-1)
        if patch_scores_arr.shape[0] != patches.shape[0]:
            raise ValueError(
                "localization_backend.score must return one score per patch. "
                f"Got {patch_scores_arr.shape[0]} scores for {patches.shape[0]} patches."
            )
        if int(np.prod(grid_shape)) != patch_scores_arr.shape[0]:
            raise ValueError(
                "grid_shape must match the number of patches. "
                f"Got grid {grid_shape} for {patch_scores_arr.shape[0]} patches."
            )
        return float(image_score), patch_scores_arr, grid_shape

    def decision_function(self, X):
        items = list(X)
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, item in enumerate(items):
            scores[i] = self._score_item(item)[0]
        return scores

    def get_anomaly_map(self, image: Any) -> NDArray:
        _score, patch_scores, grid_shape = self._score_item(image)
        return patch_scores.reshape(grid_shape).astype(np.float32, copy=False)

    def predict_anomaly_map(self, X: Iterable[Any]):
        items = list(X)
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(item) for item in items]
        return np.stack(maps, axis=0).astype(np.float32, copy=False)

    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = np.asarray(self.decision_function(X), dtype=np.float64)
        return (scores > float(self.threshold_)).astype(np.int64)
