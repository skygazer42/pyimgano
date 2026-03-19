from __future__ import annotations

from typing import Any, Iterable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .registry import register_model


def _as_patch_array(value: Any) -> NDArray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D patch embedding array. Got shape {arr.shape}.")
    return arr


def _as_vector(value: Any) -> NDArray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Anchor vectors must be non-empty.")
    return arr


def _encode_with_clip(clip_backend: Any, image: Any) -> tuple[NDArray, tuple[int, int]]:
    if clip_backend is None:
        raise ValueError("clip_backend is required for vision_aaclip.")
    if hasattr(clip_backend, "encode_image"):
        encoded = clip_backend.encode_image(image)
    elif callable(clip_backend):
        encoded = clip_backend(image)
    else:
        raise TypeError("clip_backend must be callable or implement .encode_image(image).")

    if isinstance(encoded, tuple):
        if len(encoded) < 2:
            raise ValueError("clip_backend tuple output must be (patches, grid_shape) or richer.")
        patches, grid_shape = encoded[0], encoded[1]
        return _as_patch_array(patches), (int(grid_shape[0]), int(grid_shape[1]))

    patches = _as_patch_array(encoded)
    return patches, (patches.shape[0], 1)


def _safe_unit_norm(vector: NDArray) -> NDArray:
    arr = _as_vector(vector)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32, copy=False)


def _cosine_similarity(lhs: NDArray, rhs: NDArray) -> float:
    return float(np.dot(_safe_unit_norm(lhs), _safe_unit_norm(rhs)))


class _DefaultAnchorBackend:
    def fit(self, support_patch_sets: list[NDArray]):
        support = np.concatenate(support_patch_sets, axis=0)
        self.normal_anchor_ = np.mean(support, axis=0).astype(np.float32, copy=False)
        self.anomaly_anchor_ = -self.normal_anchor_
        return self

    def get_anchors(self) -> dict[str, NDArray]:
        return {
            "normal": _safe_unit_norm(self.normal_anchor_),
            "anomaly": _safe_unit_norm(self.anomaly_anchor_),
        }


def _resolve_anchors(anchor_backend: Any, support_patch_sets: list[NDArray]) -> dict[str, NDArray]:
    backend = anchor_backend if anchor_backend is not None else _DefaultAnchorBackend()
    if hasattr(backend, "fit"):
        backend.fit(support_patch_sets)
    if not hasattr(backend, "get_anchors"):
        raise TypeError("anchor_backend must implement .get_anchors() and may implement .fit(...).")
    raw = backend.get_anchors()
    if not isinstance(raw, Mapping):
        raise TypeError("anchor_backend.get_anchors() must return a mapping.")
    if "normal" not in raw or "anomaly" not in raw:
        raise ValueError("anchor_backend.get_anchors() must return 'normal' and 'anomaly' anchors.")
    return {
        "normal": _safe_unit_norm(raw["normal"]),
        "anomaly": _safe_unit_norm(raw["anomaly"]),
    }


@register_model(
    "vision_aaclip",
    tags=("vision", "deep", "clip", "pixel_map", "zero-shot", "numpy", "aaclip"),
    metadata={
        "description": "AA-CLIP family adapter with anomaly-aware anchor competition over patch embeddings.",
        "paper": "AA-CLIP",
        "year": 2025,
        "supervision": "zero-shot",
    },
)
class VisionAAClip:
    def __init__(
        self,
        *,
        clip_backend: Any = None,
        anchor_backend: Any = None,
        contamination: float = 0.1,
    ) -> None:
        self.clip_backend = clip_backend
        self.anchor_backend = anchor_backend
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.anchors_: dict[str, NDArray] | None = None
        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None

    def fit(self, x: object = MISSING, y=None, **kwargs: object):
        del y
        items = list(cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not items:
            raise ValueError("X must contain at least one support sample.")
        support_patch_sets = [_encode_with_clip(self.clip_backend, item)[0] for item in items]
        self.anchors_ = _resolve_anchors(self.anchor_backend, support_patch_sets)
        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _score_item(self, item: Any) -> tuple[float, NDArray, tuple[int, int]]:
        if self.anchors_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        patches, grid_shape = _encode_with_clip(self.clip_backend, item)
        patch_scores = np.zeros((patches.shape[0],), dtype=np.float32)
        for i, patch in enumerate(patches):
            patch_scores[i] = float(
                _cosine_similarity(patch, self.anchors_["anomaly"])
                - _cosine_similarity(patch, self.anchors_["normal"])
            )
        if int(np.prod(grid_shape)) != patch_scores.shape[0]:
            raise ValueError(
                "grid_shape must match the number of patches. "
                f"Got grid {grid_shape} for {patch_scores.shape[0]} patches."
            )
        return float(np.max(patch_scores)), patch_scores, grid_shape

    def decision_function(self, x: object = MISSING, **kwargs: object):
        items = list(
            cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
        )
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, item in enumerate(items):
            scores[i] = self._score_item(item)[0]
        return scores

    def get_anomaly_map(self, image: Any) -> NDArray:
        _score, patch_scores, grid_shape = self._score_item(image)
        return patch_scores.reshape(grid_shape).astype(np.float32, copy=False)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object):
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
