from __future__ import annotations

"""OpenCLIP patch-level anomaly map (optional backend; no downloads by default).

This module registers a simple, industrial-friendly baseline:

  OpenCLIP ViT patch embeddings -> distance-to-normal-template -> anomaly map

Key constraints:
- `open_clip` is an optional dependency (pyimgano[clip])
- default `openclip_pretrained=None` to avoid implicit weight downloads
"""

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import require

from .openclip_backend import OpenCLIPViTPatchEmbedder
from .patchknn_core import aggregate_patch_scores, reshape_patch_scores
from .registry import register_model


def _require_open_clip(open_clip_module=None):
    if open_clip_module is not None:
        return open_clip_module
    return require("open_clip", extra="clip", purpose="OpenCLIP patch map detector")


def _l2_normalize(x: NDArray, *, axis: int, eps: float = 1e-12) -> NDArray:
    x_np = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x_np, axis=axis, keepdims=True)
    denom = np.maximum(denom, float(eps))
    return x_np / denom


@register_model(
    "vision_openclip_patch_map",
    tags=("vision", "deep", "clip", "openclip", "backend", "pixel_map"),
    metadata={
        "description": "OpenCLIP patch template distance anomaly map (requires pyimgano[clip])",
        "backend": "openclip",
    },
)
class VisionOpenCLIPPatchMap:
    """OpenCLIP patch-template distance anomaly map.

    The detector learns a single "normal template" vector in patch-embedding
    space (mean of normal patches). Anomaly score per patch is cosine distance
    to that template.
    """

    def __init__(
        self,
        *,
        open_clip_module=None,
        embedder: Optional[OpenCLIPViTPatchEmbedder] = None,
        openclip_model_name: str = "ViT-B-32",
        openclip_pretrained: Optional[str] = None,
        device: str = "cpu",
        force_image_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        contamination: float = 0.1,
        aggregation_method: str = "topk_mean",
        aggregation_topk: float = 0.01,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.openclip_model_name = str(openclip_model_name)
        self.openclip_pretrained = openclip_pretrained
        self.device = str(device)
        self.force_image_size = force_image_size
        self.normalize_embeddings = bool(normalize_embeddings)
        self.contamination = float(contamination)
        self.aggregation_method = str(aggregation_method)
        self.aggregation_topk = float(aggregation_topk)
        self.eps = float(eps)
        self._kwargs = dict(kwargs)

        self._open_clip = open_clip_module
        if embedder is None:
            open_clip = _require_open_clip(open_clip_module)
            self._open_clip = open_clip
            embedder = OpenCLIPViTPatchEmbedder(
                open_clip_module=open_clip,
                model_name=self.openclip_model_name,
                pretrained=self.openclip_pretrained,
                device=self.device,
                force_image_size=self.force_image_size,
                normalize=self.normalize_embeddings,
            )
        self.embedder = embedder

        self.template_: NDArray | None = None
        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None

    def _embed(self, image: Union[str, np.ndarray]):
        if self.embedder is None:  # pragma: no cover
            raise RuntimeError("embedder is required")
        patch_embeddings, grid_shape, original_size = self.embedder.embed(image)
        patches = np.asarray(patch_embeddings, dtype=np.float32)
        if patches.ndim != 2:
            raise ValueError(f"Expected 2D patch embeddings, got shape {patches.shape}")
        if self.normalize_embeddings:
            patches = _l2_normalize(patches, axis=1, eps=float(self.eps))
        return patches, grid_shape, original_size

    def fit(self, X, y=None):
        items = list(X)
        if not items:
            raise ValueError("X must contain at least one training image.")

        # Online mean of all patch embeddings.
        sum_vec = None
        count = 0
        for item in items:
            patches, _grid, _orig = self._embed(item)
            if sum_vec is None:
                sum_vec = patches.sum(axis=0, dtype=np.float64)
            else:
                sum_vec = sum_vec + patches.sum(axis=0, dtype=np.float64)
            count += int(patches.shape[0])

        if sum_vec is None or count <= 0:
            raise ValueError("Failed to compute template from training set.")

        template = (sum_vec / float(count)).astype(np.float32)
        if self.normalize_embeddings:
            template = _l2_normalize(template, axis=0, eps=float(self.eps)).reshape(-1)
        self.template_ = template

        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - float(self.contamination)))
        return self

    def decision_function(self, X):
        if self.template_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        items = list(X)
        scores = np.zeros(len(items), dtype=np.float64)
        tpl = np.asarray(self.template_, dtype=np.float32).reshape(-1)
        for i, item in enumerate(items):
            patches, _grid, _orig = self._embed(item)
            if patches.shape[1] != tpl.shape[0]:
                raise ValueError("Embedding dimension mismatch between patches and template.")
            sim = patches @ tpl
            patch_scores = np.asarray(1.0 - sim, dtype=np.float32)
            scores[i] = aggregate_patch_scores(
                patch_scores,
                method=self.aggregation_method,
                topk=self.aggregation_topk,
            )
        return scores

    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores > float(self.threshold_)).astype(np.int64)

    def get_anomaly_map(self, image: Union[str, np.ndarray]) -> NDArray:
        if self.template_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        tpl = np.asarray(self.template_, dtype=np.float32).reshape(-1)
        patches, grid_shape, original_size = self._embed(image)
        sim = patches @ tpl
        patch_scores = np.asarray(1.0 - sim, dtype=np.float32)
        patch_grid = reshape_patch_scores(patch_scores, grid_h=grid_shape[0], grid_w=grid_shape[1])

        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required to upsample anomaly maps.\n"
                "Install it via:\n  pip install 'opencv-python'\n"
                f"Original error: {exc}"
            ) from exc

        original_h, original_w = original_size
        upsampled = cv2.resize(
            np.asarray(patch_grid, dtype=np.float32),
            (int(original_w), int(original_h)),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.asarray(upsampled, dtype=np.float32)

    def predict_anomaly_map(self, X):
        items = list(X)
        maps = [self.get_anomaly_map(item) for item in items]
        if not maps:
            raise ValueError("X must be non-empty")

        first_shape = maps[0].shape
        for m in maps[1:]:
            if m.shape != first_shape:
                raise ValueError(
                    "Inconsistent anomaly map shapes. " f"Expected {first_shape}, got {m.shape}."
                )
        return np.stack(maps)
