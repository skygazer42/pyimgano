from __future__ import annotations

"""Patch-embedding + classical core anomaly map (industrial baseline).

This model implements a pragmatic "deep patch embeddings + classical core" route:

  image -> conv patch embeddings -> core_* detector -> patch scores -> anomaly map

Why this exists
--------------
- `vision_patchcore_lite_map` is a great baseline, but hardcodes a memory-bank kNN core.
- In industrial settings you often want to reuse the existing `core_*` ecosystem:
  robust/statistical scorers, calibration, and well-tested score semantics.

Key constraints
---------------
- Offline-safe by default (`pretrained=False`).
- No new required dependencies.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pyimgano.features.torchvision_conv_patch_embedder import TorchvisionConvPatchEmbedder

from .patchknn_core import AggregationMethod, aggregate_patch_scores, reshape_patch_scores
from .registry import create_model, register_model


@dataclass
class _EmbeddedImage:
    patch_embeddings: NDArray
    grid_shape: Tuple[int, int]
    original_size: Tuple[int, int]


def _build_core_detector(
    core_detector: str | type | Any,
    *,
    contamination: float,
    core_kwargs: Mapping[str, Any],
):
    kwargs = dict(core_kwargs)
    # Avoid accidentally double-passing contamination.
    kwargs.pop("contamination", None)

    if isinstance(core_detector, str):
        return create_model(str(core_detector), contamination=float(contamination), **kwargs)
    if isinstance(core_detector, type):
        return core_detector(contamination=float(contamination), **kwargs)
    return core_detector


@register_model(
    "vision_patch_embedding_core_map",
    tags=("vision", "classical", "pipeline", "embeddings", "patch", "pixel_map"),
    metadata={
        "description": "Patch embeddings + core detector anomaly map (generic industrial baseline)",
    },
)
class VisionPatchEmbeddingCoreMap:
    """Patch embeddings + `core_*` detector anomaly map.

    Notes
    -----
    - The embedder is injectable so unit tests (and offline deployments) can run
      without any implicit weight downloads.
    - The core detector is also injectable; by default it uses `core_dtc` which
      is deterministic and dependency-light.
    """

    def __init__(
        self,
        *,
        embedder: Optional[TorchvisionConvPatchEmbedder] = None,
        contamination: float = 0.1,
        pretrained: bool = False,
        # classical core detector
        core_detector: str | type | Any = "core_dtc",
        core_kwargs: Mapping[str, Any] | None = None,
        # aggregation (patch scores -> image score)
        aggregation_method: AggregationMethod = "topk_mean",
        aggregation_topk: float = 0.01,
        # default embedder knobs
        backbone: str = "resnet18",
        node: str = "layer3",
        device: str = "cpu",
        image_size: int = 224,
        normalize_embeddings: bool = True,
        eps: float = 1e-12,
    ) -> None:
        if embedder is None:
            embedder = TorchvisionConvPatchEmbedder(
                backbone=str(backbone),
                node=str(node),
                pretrained=bool(pretrained),
                device=str(device),
                image_size=int(image_size),
                normalize=bool(normalize_embeddings),
                eps=float(eps),
            )

        self.embedder = embedder
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self.pretrained = bool(pretrained)

        self.core_detector = core_detector
        self.core_kwargs = dict(core_kwargs or {})

        self.aggregation_method = aggregation_method
        self.aggregation_topk = float(aggregation_topk)

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

        self._core = None

    def _embed(self, image: Union[str, np.ndarray]) -> _EmbeddedImage:
        patch_embeddings, grid_shape, original_size = self.embedder.embed(image)
        patch_embeddings_np = np.asarray(patch_embeddings, dtype=np.float32)
        if patch_embeddings_np.ndim != 2:
            raise ValueError(f"Expected 2D patch embeddings, got shape {patch_embeddings_np.shape}")

        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if patch_embeddings_np.shape[0] != grid_h * grid_w:
            raise ValueError(
                "Patch embedding count does not match grid shape. "
                f"Got {patch_embeddings_np.shape[0]} patches for grid {grid_h}x{grid_w}."
            )

        original_h, original_w = int(original_size[0]), int(original_size[1])
        if original_h <= 0 or original_w <= 0:
            raise ValueError(f"Invalid original_size: {original_size}")

        return _EmbeddedImage(
            patch_embeddings=patch_embeddings_np,
            grid_shape=(grid_h, grid_w),
            original_size=(original_h, original_w),
        )

    def _patch_scores(self, embedded: _EmbeddedImage) -> NDArray:
        if self._core is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = np.asarray(self._core.decision_function(embedded.patch_embeddings), dtype=np.float64).reshape(-1)
        if scores.shape[0] != embedded.patch_embeddings.shape[0]:
            raise ValueError(
                "core_detector.decision_function must return one score per patch. "
                f"Got {scores.shape[0]} for {embedded.patch_embeddings.shape[0]} patches."
            )
        return np.asarray(scores, dtype=np.float32)

    def fit(self, X: Iterable[Union[str, np.ndarray]], y=None):
        items = list(X)
        if not items:
            raise ValueError("X must contain at least one training image.")

        embedded_train = [self._embed(item) for item in items]
        patch_matrix = np.concatenate([e.patch_embeddings for e in embedded_train], axis=0)

        core = _build_core_detector(
            self.core_detector,
            contamination=float(self.contamination),
            core_kwargs=self.core_kwargs,
        )
        if not hasattr(core, "fit") or not hasattr(core, "decision_function"):
            raise TypeError("core_detector must implement .fit(X) and .decision_function(X)")

        core.fit(patch_matrix)
        self._core = core

        self.decision_scores_ = self.decision_function(items)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def decision_function(self, X: Iterable[Union[str, np.ndarray]]) -> NDArray:
        items = list(X)
        scores = np.zeros(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            embedded = self._embed(item)
            patch_scores = self._patch_scores(embedded)
            scores[i] = aggregate_patch_scores(
                patch_scores,
                method=self.aggregation_method,
                topk=self.aggregation_topk,
            )
        return scores

    def predict(self, X: Iterable[Union[str, np.ndarray]]) -> NDArray:
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores > float(self.threshold_)).astype(np.int64)

    def get_anomaly_map(self, image: Union[str, np.ndarray]) -> NDArray:
        embedded = self._embed(image)
        patch_scores = self._patch_scores(embedded)
        patch_grid = reshape_patch_scores(
            patch_scores,
            grid_h=embedded.grid_shape[0],
            grid_w=embedded.grid_shape[1],
        )

        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required to upsample anomaly maps.\n"
                "Install it via:\n  pip install 'opencv-python'\n"
                f"Original error: {exc}"
            ) from exc

        original_h, original_w = embedded.original_size
        upsampled = cv2.resize(
            np.asarray(patch_grid, dtype=np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.asarray(upsampled, dtype=np.float32)

    def predict_anomaly_map(self, X: Iterable[Union[str, np.ndarray]]) -> NDArray:
        items = list(X)
        maps = [self.get_anomaly_map(item) for item in items]
        return np.stack(maps)


__all__ = ["VisionPatchEmbeddingCoreMap"]

