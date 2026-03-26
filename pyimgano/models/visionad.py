from __future__ import annotations

from typing import Any, Iterable, cast

import numpy as np
from numpy.typing import NDArray

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .registry import register_model


def _as_float_array(value: Any) -> NDArray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D patch embedding array. Got shape {arr.shape}.")
    return arr


def _call_embedder(
    embedder: Any, image: Any
) -> tuple[NDArray, tuple[int, int] | None, tuple[int, int] | None]:
    if embedder is None:
        raise ValueError("embedder is required for vision_visionad.")

    if hasattr(embedder, "embed"):
        output = embedder.embed(image)
    elif callable(embedder):
        output = embedder(image)
    else:
        raise TypeError("embedder must be callable or implement .embed(image).")

    if isinstance(output, tuple):
        if len(output) == 3:
            patches, grid_shape, original_size = output
            return (
                _as_float_array(patches),
                (int(grid_shape[0]), int(grid_shape[1])),
                (int(original_size[0]), int(original_size[1])),
            )
        raise ValueError("embedder tuple output must be (patches, grid_shape, original_size).")

    return _as_float_array(output), None, None


class _CentroidSearchBackend:
    def fit(self, train_patches: list[NDArray]):
        self.train_centroid_ = np.mean(np.concatenate(train_patches, axis=0), axis=0).astype(
            np.float32,
            copy=False,
        )
        return self

    def score(self, patch_grid: NDArray) -> tuple[float, NDArray]:
        delta = np.asarray(patch_grid, dtype=np.float32) - self.train_centroid_[None, :]
        patch_scores = np.linalg.norm(delta, axis=1).astype(np.float32, copy=False)
        return float(np.max(patch_scores)), patch_scores


@register_model(
    "vision_visionad",
    tags=("vision", "deep", "neighbors", "few-shot", "numpy", "visionad"),
    metadata={
        "description": "VisionAD family adapter with search-backend scoring over patch embeddings.",
        "paper": "Search is All You Need for Few-shot Anomaly Detection",
        "year": 2025,
        "supervision": "few-shot",
    },
)
class VisionVisionAD:
    def __init__(
        self,
        *,
        embedder: Any = None,
        search_backend: Any = None,
        contamination: float = 0.1,
    ) -> None:
        self.embedder = embedder
        self.search_backend = (
            search_backend if search_backend is not None else _CentroidSearchBackend()
        )
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None

    def fit(self, x: object = MISSING, y=None, **kwargs: object):
        del y
        items = list(cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not items:
            raise ValueError("X must contain at least one support sample.")
        embedded = [_call_embedder(self.embedder, item)[0] for item in items]
        if not hasattr(self.search_backend, "fit") or not hasattr(self.search_backend, "score"):
            raise TypeError(
                "search_backend must implement .fit(train_patches) and .score(patches)."
            )
        self.search_backend.fit(embedded)
        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _score_item(
        self, item: Any
    ) -> tuple[float, NDArray, tuple[int, int] | None, tuple[int, int] | None]:
        patches, grid_shape, original_size = _call_embedder(self.embedder, item)
        image_score, patch_scores = self.search_backend.score(patches)
        patch_scores_arr = np.asarray(patch_scores, dtype=np.float32).reshape(-1)
        if patch_scores_arr.shape[0] != patches.shape[0]:
            raise ValueError(
                "search_backend.score must return one patch score per patch. "
                f"Got {patch_scores_arr.shape[0]} scores for {patches.shape[0]} patches."
            )
        return float(image_score), patch_scores_arr, grid_shape, original_size

    def decision_function(self, x: object = MISSING, **kwargs: object):
        items = list(
            cast(
                Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
            )
        )
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, item in enumerate(items):
            scores[i] = self._score_item(item)[0]
        return scores

    def get_anomaly_map(self, image: Any) -> NDArray:
        _score, patch_scores, grid_shape, _original_size = self._score_item(image)
        if grid_shape is None:
            raise ValueError("embedder must return grid/original metadata for anomaly maps.")
        return patch_scores.reshape(grid_shape).astype(np.float32, copy=False)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        items = list(
            cast(
                Iterable[Any],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
            )
        )
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(item) for item in items]
        return np.stack(maps, axis=0).astype(np.float32, copy=False)

    def predict(self, x: object = MISSING, **kwargs: object):
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = np.asarray(
            self.decision_function(
                cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
            ),
            dtype=np.float64,
        )
        return (scores > float(self.threshold_)).astype(np.int64)
