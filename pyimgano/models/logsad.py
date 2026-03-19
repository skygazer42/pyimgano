from __future__ import annotations

from typing import Any, Iterable, cast

import numpy as np
from numpy.typing import NDArray

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .registry import register_model


def _as_image(value: Any) -> NDArray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected image-like array with ndim >= 2. Got shape {arr.shape}.")
    return arr


def _as_map(value: Any, *, shape: tuple[int, ...]) -> NDArray:
    arr = np.asarray(value, dtype=np.float32)
    expected = shape[:2]
    if arr.shape != expected:
        raise ValueError(f"Expected anomaly map shape {expected}, got {arr.shape}.")
    return arr


def _score_visual(visual_backend: Any, image: NDArray) -> tuple[float, NDArray]:
    if visual_backend is None:
        anomaly_map = np.mean(np.abs(image), axis=-1) if image.ndim > 2 else np.abs(image)
        return float(np.max(anomaly_map)), anomaly_map.astype(np.float32, copy=False)

    if hasattr(visual_backend, "score"):
        raw = visual_backend.score(image)
    elif callable(visual_backend):
        raw = visual_backend(image)
    else:
        raise TypeError("visual_backend must be callable or implement .score(image).")

    if isinstance(raw, tuple):
        if len(raw) != 2:
            raise ValueError("visual_backend.score(...) must return (score, anomaly_map).")
        score, anomaly_map = raw
        return float(score), _as_map(anomaly_map, shape=image.shape)

    raise TypeError("visual_backend.score(...) must return (score, anomaly_map).")


def _score_logic(logic_backend: Any, image: NDArray) -> float:
    if logic_backend is None:
        return float(np.mean(image > 0.0))

    if hasattr(logic_backend, "score"):
        return float(logic_backend.score(image))
    if callable(logic_backend):
        return float(logic_backend(image))
    raise TypeError("logic_backend must be callable or implement .score(image).")


@register_model(
    "vision_logsad",
    tags=("vision", "deep", "zero-shot", "pixel_map", "numpy", "logsad"),
    metadata={
        "description": "LogSAD family adapter with combined visual-structural and logic-rule anomaly scoring.",
        "paper": "Towards Training-free Anomaly Detection with Vision and Language Foundation Models",
        "year": 2025,
        "supervision": "zero-shot",
    },
)
class VisionLogSAD:
    def __init__(
        self,
        *,
        logic_backend: Any = None,
        visual_backend: Any = None,
        visual_weight: float = 1.0,
        logic_weight: float = 1.0,
        contamination: float = 0.1,
    ) -> None:
        self.logic_backend = logic_backend
        self.visual_backend = visual_backend
        self.visual_weight = float(visual_weight)
        self.logic_weight = float(logic_weight)
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None

    def fit(self, x: object = MISSING, _y=None, **kwargs: object):
        del _y
        items = list(cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not items:
            raise ValueError("X must contain at least one support sample.")
        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _score_item(self, item: Any) -> tuple[float, NDArray]:
        image = _as_image(item)
        visual_score, anomaly_map = _score_visual(self.visual_backend, image)
        logic_score = _score_logic(self.logic_backend, image)
        total = self.visual_weight * visual_score + self.logic_weight * logic_score
        return float(total), anomaly_map

    def decision_function(self, x: object = MISSING, **kwargs: object):
        items = list(
            cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
        )
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, item in enumerate(items):
            scores[i] = self._score_item(item)[0]
        return scores

    def get_anomaly_map(self, image: Any) -> NDArray:
        return self._score_item(image)[1].astype(np.float32, copy=False)

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
