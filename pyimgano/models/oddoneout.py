"""Odd-One-Out (CVPR 2025) neighbor comparison detector.

This module provides a **vision wrapper** that aligns with pyimgano's industrial
contracts:
- Offline-by-default (no implicit backbone weight downloads)
- `BaseDetector` semantics (higher score ⇒ more anomalous)
- Feature-extractor driven vision path (paths/ndarrays → embeddings → core detector)

Implementation note
-------------------
The core algorithm lives in `core_oddoneout` (feature-matrix first). This
`vision_oddoneout` wrapper simply:

  torchvision_backbone embeddings  +  core_oddoneout backend

Users can also compose the same route via `vision_embedding_core` by selecting
`core_detector="core_oddoneout"`.
"""

from __future__ import annotations

from typing import Any, Optional

from .baseml import BaseVisionDetector
from .core_oddoneout import _OddOneOutBackend
from .registry import register_model


@register_model(
    "vision_oddoneout",
    tags=("vision", "classical", "neighbors", "oddoneout", "cvpr2025", "embeddings"),
    metadata={
        "description": "Odd-One-Out neighbor comparison (CVPR 2025-inspired) on deep embeddings",
        "input": "paths",
    },
)
class VisionOddOneOut(BaseVisionDetector):
    """Odd-One-Out neighbor comparison on vision embeddings (offline-by-default)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        # Embedding defaults (industrial-safe)
        backbone: str = "resnet18",
        image_size: int = 64,
        batch_size: int = 8,
        device: str = "cpu",
        # Core detector params
        n_neighbors: int = 5,
        metric: str = "minkowski",
        p: int = 2,
        method: str = "mean",
        normalize: bool = True,
        eps: float = 1e-12,
        n_jobs: int = 1,
        random_state: Optional[int] = 0,
    ) -> None:
        self._detector_kwargs = dict(
            contamination=float(contamination),
            n_neighbors=int(n_neighbors),
            metric=str(metric),
            p=int(p),
            method=str(method),
            normalize=bool(normalize),
            eps=float(eps),
            n_jobs=int(n_jobs),
            random_state=random_state,
        )

        if feature_extractor is None:
            # Default to a lightweight offline embedding extractor.
            feature_extractor = {
                "name": "torchvision_backbone",
                "kwargs": {
                    "backbone": str(backbone),
                    "pretrained": False,
                    "pool": "avg",
                    "device": str(device),
                    "batch_size": int(batch_size),
                    "image_size": int(image_size),
                },
            }

        super().__init__(contamination=float(contamination), feature_extractor=feature_extractor)

    def _build_detector(self):
        return _OddOneOutBackend(**self._detector_kwargs)


__all__ = ["VisionOddOneOut"]
