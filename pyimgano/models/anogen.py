from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from ._image_batch import coerce_rgb_image_batch
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."





def _as_float_array(image: Any) -> NDArray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected image-like array with ndim >= 2. Got shape {arr.shape}.")
    return arr


def _as_mask(mask: Any, *, shape: tuple[int, ...]) -> NDArray:
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.shape != shape[:2]:
        raise ValueError(f"Expected mask shape {shape[:2]}, got {mask_arr.shape}.")
    return mask_arr


def _call_generator(generator: Any, image: NDArray) -> tuple[NDArray, NDArray, dict[str, Any]]:
    if generator is None:
        raise ValueError("generator is required for vision_anogen_adapter.")

    if hasattr(generator, "generate"):
        output = generator.generate(image)
    elif callable(generator):
        output = generator(image)
    else:
        raise TypeError("generator must be callable or implement .generate(image).")

    if not isinstance(output, tuple):
        raise TypeError("generator output must be a tuple.")

    if len(output) == 2:
        anomalous, mask = output
        meta: dict[str, Any] = {}
    elif len(output) == 3:
        anomalous, mask, meta = output
    else:
        raise ValueError("generator must return (anomalous, mask) or (anomalous, mask, meta).")

    anomalous_arr = _as_float_array(anomalous)
    if anomalous_arr.shape != image.shape:
        raise ValueError(
            "Generated anomalous sample must match the input shape. "
            f"Got {anomalous_arr.shape} vs {image.shape}."
        )

    return anomalous_arr, _as_mask(mask, shape=image.shape), dict(meta or {})


@register_model(
    "vision_anogen_adapter",
    tags=("vision", "deep", "reconstruction", "few-shot", "numpy", "anogen", "pixel_map"),
    metadata={
        "description": "AnoGen family adapter with deterministic generator hooks and residual scoring.",
        "paper": "Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation",
        "year": 2025,
        "supervision": "few-shot",
    },
)
class VisionAnoGenAdapter:
    def __init__(
        self,
        *,
        generator: Any = None,
        scoring_backend: Any = None,
        contamination: float = 0.1,
    ) -> None:
        self.generator = generator
        self.scoring_backend = scoring_backend
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.support_images_: list[NDArray] | None = None
        self.support_prototype_: NDArray | None = None
        self.generated_pairs_: list[dict[str, Any]] | None = None
        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None

    def generate_training_pairs(self, x: Iterable[Any]) -> list[dict[str, Any]]:
        items = list(x)
        pairs: list[dict[str, Any]] = []
        for item in items:
            image = _as_float_array(item)
            anomalous, mask, meta = _call_generator(self.generator, image)
            pairs.append(
                {
                    "normal": image,
                    "anomalous": anomalous,
                    "mask": mask,
                    "meta": meta,
                }
            )
        return pairs

    def fit(self, x, y=None):
        del y
        self.support_images_ = list(coerce_rgb_image_batch(x).astype(np.float32, copy=False))
        if not self.support_images_:
            raise ValueError("X must contain at least one support image.")
        first_shape = self.support_images_[0].shape
        for arr in self.support_images_[1:]:
            if arr.shape != first_shape:
                raise ValueError(
                    "All support images must share the same shape. "
                    f"Expected {first_shape}, got {arr.shape}."
                )

        self.support_prototype_ = np.mean(np.stack(self.support_images_, axis=0), axis=0).astype(
            np.float32,
            copy=False,
        )
        self.generated_pairs_ = self.generate_training_pairs(self.support_images_)
        self.decision_scores_ = np.asarray(self.decision_function(self.support_images_), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _default_score_map(self, image: NDArray) -> NDArray:
        if self.support_prototype_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        residual = np.abs(image - self.support_prototype_)
        if residual.ndim == 2:
            return residual.astype(np.float32, copy=False)
        return np.mean(residual, axis=-1).astype(np.float32, copy=False)

    def _score_with_backend(self, image: NDArray) -> float:
        backend = self.scoring_backend
        if backend is None:
            return float(np.mean(self._default_score_map(image)))

        if hasattr(backend, "score"):
            return float(
                backend.score(
                    image,
                    prototype=self.support_prototype_,
                    generated_pairs=self.generated_pairs_,
                )
            )
        if callable(backend):
            return float(
                backend(
                    image,
                    prototype=self.support_prototype_,
                    generated_pairs=self.generated_pairs_,
                )
            )
        raise TypeError("scoring_backend must be callable or implement .score(...).")

    def decision_function(self, x):
        if self.support_prototype_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        items = list(coerce_rgb_image_batch(x).astype(np.float32, copy=False))
        scores = np.zeros((len(items),), dtype=np.float64)
        for i, item in enumerate(items):
            scores[i] = self._score_with_backend(_as_float_array(item))
        return scores

    def get_anomaly_map(self, image: Any) -> NDArray:
        return self._default_score_map(_as_float_array(image))

    def predict_anomaly_map(self, x: Iterable[Any]) -> NDArray:
        items = list(coerce_rgb_image_batch(x).astype(np.float32, copy=False))
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(item) for item in items]
        return np.stack(maps, axis=0).astype(np.float32, copy=False)

    def predict(self, x):
        if self.threshold_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        scores = np.asarray(self.decision_function(x), dtype=np.float64)
        return (scores > float(self.threshold_)).astype(np.int64)
