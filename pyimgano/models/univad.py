from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .registry import register_model


def _as_vector(value: Any) -> NDArray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Layer features must be non-empty.")
    return arr


def _extract_layers(feature_extractor: Any, image: Any) -> Mapping[str, NDArray]:
    if feature_extractor is None:
        raise ValueError("feature_extractor is required for vision_univad.")

    if hasattr(feature_extractor, "extract"):
        raw = feature_extractor.extract(image)
    elif callable(feature_extractor):
        raw = feature_extractor(image)
    else:
        raise TypeError("feature_extractor must be callable or implement .extract(image).")

    if isinstance(raw, Mapping):
        return {str(name): _as_vector(value) for name, value in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {f"layer_{i}": _as_vector(value) for i, value in enumerate(raw)}
    return {"layer_0": _as_vector(raw)}


def _resolve_weights(layer_names: Sequence[str], layer_weights: Any) -> dict[str, float]:
    if layer_weights is None:
        return {name: 1.0 for name in layer_names}
    if isinstance(layer_weights, Mapping):
        return {str(name): float(layer_weights.get(name, 0.0)) for name in layer_names}
    if isinstance(layer_weights, Sequence) and not isinstance(layer_weights, (str, bytes)):
        weights = list(layer_weights)
        if len(weights) != len(layer_names):
            raise ValueError("Sequence layer_weights must match the number of layers.")
        return {name: float(weights[i]) for i, name in enumerate(layer_names)}
    raise TypeError("layer_weights must be a mapping, a sequence, or None.")


def _fuse_layers(layers: Mapping[str, NDArray], layer_weights: Any) -> NDArray:
    names = list(layers.keys())
    weights = _resolve_weights(names, layer_weights)
    fused_parts: list[NDArray] = []
    for name in names:
        fused_parts.append(layers[name] * float(weights[name]))
    return np.concatenate(fused_parts, axis=0).astype(np.float32, copy=False)


class _PrototypeSupportBackend:
    def fit(self, support_vectors: NDArray):
        self.support_vectors_ = np.asarray(support_vectors, dtype=np.float32)
        self.prototype_ = np.mean(self.support_vectors_, axis=0).astype(np.float32, copy=False)
        return self

    def score(self, vector: NDArray) -> float:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        return float(np.linalg.norm(vec - self.prototype_))


@register_model(
    "vision_univad",
    tags=("vision", "deep", "neighbors", "few-shot", "numpy", "univad"),
    metadata={
        "description": "UniVAD family adapter with multi-layer support fusion and prototype scoring.",
        "paper": "UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection",
        "year": 2025,
        "supervision": "few-shot",
    },
)
class VisionUniVAD:
    def __init__(
        self,
        *,
        feature_extractor: Any = None,
        support_backend: Any = None,
        layer_weights: Any = None,
        contamination: float = 0.1,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.support_backend = support_backend if support_backend is not None else _PrototypeSupportBackend()
        self.layer_weights = layer_weights
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None

    def _vectorize(self, image: Any) -> NDArray:
        layers = _extract_layers(self.feature_extractor, image)
        return _fuse_layers(layers, self.layer_weights)

    def fit(self, X, _y=None):
        items = list(X)
        if not items:
            raise ValueError("X must contain at least one support sample.")
        support_vectors = np.stack([self._vectorize(item) for item in items], axis=0)
        if not hasattr(self.support_backend, "fit") or not hasattr(self.support_backend, "score"):
            raise TypeError("support_backend must implement .fit(vectors) and .score(vector).")
        self.support_backend.fit(support_vectors)
        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def decision_function(self, X):
        items = list(X)
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, item in enumerate(items):
            scores[i] = float(self.support_backend.score(self._vectorize(item)))
        return scores

    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = np.asarray(self.decision_function(X), dtype=np.float64)
        return (scores > float(self.threshold_)).astype(np.int64)
