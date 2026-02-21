from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .anomalydino import PatchEmbedder, TorchHubDinoV2Embedder
from .knn_index import KNNIndex, build_knn_index
from .patchknn_core import AggregationMethod, aggregate_patch_scores, reshape_patch_scores
from .registry import register_model


@dataclass
class _EmbeddedImage:
    patch_embeddings: NDArray
    grid_shape: Tuple[int, int]
    original_size: Tuple[int, int]


@register_model(
    "vision_softpatch",
    tags=("vision", "deep", "softpatch", "patchknn", "robust"),
    metadata={
        "description": "SoftPatch-inspired robust patch-memory detector (few-shot friendly)",
    },
)
class VisionSoftPatch:
    """SoftPatch-inspired robust patch-memory detector (inference-first).

    Notes
    -----
    This detector is intentionally implemented in a Patch-kNN style:

    - Extract patch embeddings for each image.
    - Build a patch memory bank from normal/reference images.
    - Score patches by kNN distance to the memory bank.
    - Upsample patch scores to an anomaly map and aggregate to an image score.

    The embedder is injectable so unit tests can run without torch and without
    downloading weights.
    """

    def __init__(
        self,
        *,
        embedder: Optional[PatchEmbedder] = None,
        contamination: float = 0.1,
        knn_backend: str = "sklearn",
        n_neighbors: int = 1,
        coreset_sampling_ratio: float = 1.0,
        random_seed: int = 0,
        train_patch_outlier_quantile: float = 0.0,
        aggregation_method: AggregationMethod = "topk_mean",
        aggregation_topk: float = 0.01,
        device: str = "cpu",
        image_size: int = 518,
        dino_model_name: str = "dinov2_vits14",
    ) -> None:
        if embedder is None:
            embedder = TorchHubDinoV2Embedder(
                model_name=dino_model_name,
                device=device,
                image_size=image_size,
            )

        self.embedder = embedder

        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5). Got {self.contamination}."
            )

        self.knn_backend = str(knn_backend)
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1. Got {self.n_neighbors}.")

        self.coreset_sampling_ratio = float(coreset_sampling_ratio)
        if not (0.0 < self.coreset_sampling_ratio <= 1.0):
            raise ValueError(
                "coreset_sampling_ratio must be in (0, 1]. "
                f"Got {self.coreset_sampling_ratio}."
            )
        self.random_seed = int(random_seed)

        self.train_patch_outlier_quantile = float(train_patch_outlier_quantile)
        if not (0.0 <= self.train_patch_outlier_quantile < 1.0):
            raise ValueError(
                "train_patch_outlier_quantile must be in [0, 1). "
                f"Got {self.train_patch_outlier_quantile}."
            )

        self.aggregation_method = aggregation_method
        self.aggregation_topk = float(aggregation_topk)

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

        self._memory_bank: Optional[NDArray] = None
        self._knn_index: Optional[KNNIndex] = None
        self._n_neighbors_fit: Optional[int] = None

    @property
    def memory_bank_size_(self) -> int:
        if self._memory_bank is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return int(self._memory_bank.shape[0])

    def _embed(self, image_path: str) -> _EmbeddedImage:
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

        return _EmbeddedImage(
            patch_embeddings=patch_embeddings_np,
            grid_shape=(grid_h, grid_w),
            original_size=(original_h, original_w),
        )

    def fit(self, X: Iterable[str], y=None):
        paths = list(X)
        if not paths:
            raise ValueError("X must contain at least one training image path.")

        embedded_train = [self._embed(path) for path in paths]
        memory_bank = np.concatenate([e.patch_embeddings for e in embedded_train], axis=0)

        if self.coreset_sampling_ratio < 1.0:
            rng = np.random.default_rng(self.random_seed)
            n_total = int(memory_bank.shape[0])
            n_keep = max(1, int(math.ceil(self.coreset_sampling_ratio * n_total)))
            n_keep = min(n_keep, n_total)
            keep_idx = rng.choice(n_total, size=n_keep, replace=False)
            memory_bank = memory_bank[keep_idx]

        self._memory_bank = memory_bank

        effective_k = min(max(1, self.n_neighbors), int(memory_bank.shape[0]))
        self._n_neighbors_fit = effective_k
        self._knn_index = build_knn_index(backend=self.knn_backend, n_neighbors=effective_k)
        self._knn_index.fit(memory_bank)

        self.decision_scores_ = self.decision_function(paths)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _patch_scores(self, embedded: _EmbeddedImage) -> NDArray:
        if self._knn_index is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._n_neighbors_fit is None:
            raise RuntimeError("Internal error: missing fitted neighbor count.")

        distances, _indices = self._knn_index.kneighbors(
            embedded.patch_embeddings,
            n_neighbors=self._n_neighbors_fit,
        )
        distances_np = np.asarray(distances, dtype=np.float32)
        if distances_np.ndim != 2:
            raise RuntimeError(f"Expected 2D kNN distances, got shape {distances_np.shape}")

        return distances_np.min(axis=1)

    def decision_function(self, X: Iterable[str]) -> NDArray:
        paths = list(X)
        scores = np.zeros(len(paths), dtype=np.float64)
        for i, path in enumerate(paths):
            embedded = self._embed(path)
            patch_scores = self._patch_scores(embedded)
            scores[i] = aggregate_patch_scores(
                patch_scores,
                method=self.aggregation_method,
                topk=self.aggregation_topk,
            )
        return scores

    def predict(self, X: Iterable[str]) -> NDArray:
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image_path: str) -> NDArray:
        embedded = self._embed(image_path)
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

    def predict_anomaly_map(self, X: Iterable[str]) -> NDArray:
        paths = list(X)
        maps = [self.get_anomaly_map(path) for path in paths]
        return np.stack(maps)

