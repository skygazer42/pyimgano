# -*- coding: utf-8 -*-
"""Convenience model: (embedding extractor) + (core detector).

This is the recommended industrial baseline route:
  images -> deep embeddings -> classical core detector -> score

It is implemented as a thin, registry-friendly wrapper around
`pyimgano.pipelines.feature_pipeline.VisionFeaturePipeline`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pyimgano.models.registry import register_model
from pyimgano.pipelines.feature_pipeline import VisionFeaturePipeline


@register_model(
    "vision_embedding_core",
    tags=("vision", "classical", "pipeline", "embeddings"),
    metadata={
        "description": "Embedding extractor + core detector pipeline (industrial baseline)",
    },
)
class VisionEmbeddingCoreDetector(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_detector: str | type | Any = "core_ecod",
        core_kwargs: Mapping[str, Any] | None = None,
        contamination: float = 0.1,
    ) -> None:
        if embedding_kwargs:
            feature_extractor = {"name": str(embedding_extractor), "kwargs": dict(embedding_kwargs)}
        else:
            feature_extractor = embedding_extractor

        super().__init__(
            core_detector=core_detector,
            core_kwargs=core_kwargs,
            feature_extractor=feature_extractor,
            contamination=float(contamination),
        )

