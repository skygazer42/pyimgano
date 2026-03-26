from __future__ import annotations

from typing import Any, Iterable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."


def _as_vector(value: Any) -> NDArray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Encoded features must be non-empty.")
    return arr


def _encode_image(clip_backend: Any, image: Any) -> NDArray:
    if clip_backend is None:
        raise ValueError("clip_backend is required for vision_adaclip.")
    if hasattr(clip_backend, "encode_image"):
        encoded = clip_backend.encode_image(image)
    elif callable(clip_backend):
        encoded = clip_backend(image)
    else:
        raise TypeError("clip_backend must be callable or implement .encode_image(image).")
    return _as_vector(encoded)


def _read_prompt_dict(prompt_backend: Any, method_name: str, *args: Any) -> dict[str, NDArray]:
    if prompt_backend is None:
        raise ValueError("prompt_backend is required for vision_adaclip.")
    if not hasattr(prompt_backend, method_name):
        raise TypeError(f"prompt_backend must implement .{method_name}(...).")
    raw = getattr(prompt_backend, method_name)(*args)
    if not isinstance(raw, Mapping):
        raise TypeError(f"prompt_backend.{method_name}(...) must return a mapping.")
    if "normal" not in raw or "anomaly" not in raw:
        raise ValueError(
            f"prompt_backend.{method_name}(...) must return 'normal' and 'anomaly' prompts."
        )
    return {
        "normal": _as_vector(raw["normal"]),
        "anomaly": _as_vector(raw["anomaly"]),
    }


def _safe_unit_norm(vector: NDArray) -> NDArray:
    arr = _as_vector(vector)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32, copy=False)


def _cosine_similarity(lhs: NDArray, rhs: NDArray) -> float:
    lhs_unit = _safe_unit_norm(lhs)
    rhs_unit = _safe_unit_norm(rhs)
    return float(np.dot(lhs_unit, rhs_unit))


@register_model(
    "vision_adaclip",
    tags=("vision", "deep", "clip", "zero-shot", "numpy", "adaclip"),
    metadata={
        "description": "AdaCLIP family adapter with static/dynamic prompt fusion for zero-shot scoring.",
        "paper": "AdaCLIP",
        "year": 2024,
        "supervision": "zero-shot",
    },
)
class VisionAdaCLIP:
    def __init__(
        self,
        *,
        clip_backend: Any = None,
        prompt_backend: Any = None,
        dynamic_prompt_weight: float = 0.5,
        contamination: float = 0.1,
    ) -> None:
        self.clip_backend = clip_backend
        self.prompt_backend = prompt_backend
        self.dynamic_prompt_weight = float(dynamic_prompt_weight)
        if not (0.0 <= self.dynamic_prompt_weight <= 1.0):
            raise ValueError(
                f"dynamic_prompt_weight must be in [0, 1]. Got {self.dynamic_prompt_weight}."
            )

        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.static_prompts_: dict[str, NDArray] | None = None
        self.dynamic_prompts_: dict[str, NDArray] | None = None
        self.hybrid_prompts_: dict[str, NDArray] | None = None
        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None

    def _build_hybrid_prompts(self) -> dict[str, NDArray]:
        if self.static_prompts_ is None or self.dynamic_prompts_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        weight = self.dynamic_prompt_weight
        prompts: dict[str, NDArray] = {}
        for key in ("normal", "anomaly"):
            prompts[key] = _safe_unit_norm(
                (1.0 - weight) * self.static_prompts_[key] + weight * self.dynamic_prompts_[key]
            )
        return prompts

    def fit(self, x: object = MISSING, y=None, **kwargs: object):
        del y
        items = list(cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not items:
            raise ValueError("X must contain at least one support sample.")

        support_features = np.stack(
            [_encode_image(self.clip_backend, item) for item in items], axis=0
        )
        self.static_prompts_ = _read_prompt_dict(self.prompt_backend, "get_static_prompts")
        if hasattr(self.prompt_backend, "adapt"):
            self.dynamic_prompts_ = _read_prompt_dict(
                self.prompt_backend, "adapt", support_features
            )
        else:
            self.dynamic_prompts_ = dict(self.static_prompts_)
        self.hybrid_prompts_ = self._build_hybrid_prompts()
        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _score_feature(self, feature: NDArray) -> float:
        if self.hybrid_prompts_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        anomaly_score = _cosine_similarity(feature, self.hybrid_prompts_["anomaly"])
        normal_score = _cosine_similarity(feature, self.hybrid_prompts_["normal"])
        return float(anomaly_score - normal_score)

    def decision_function(self, x: object = MISSING, **kwargs: object):
        items = list(
            cast(
                Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
            )
        )
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, item in enumerate(items):
            scores[i] = self._score_feature(_encode_image(self.clip_backend, item))
        return scores

    def predict(self, x: object = MISSING, **kwargs: object):
        if self.threshold_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        scores = np.asarray(
            self.decision_function(
                cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
            ),
            dtype=np.float64,
        )
        return (scores > float(self.threshold_)).astype(np.int64)
