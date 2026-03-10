from __future__ import annotations

from typing import Any

from .registry import register_model


@register_model(
    "vision_visionad",
    tags=("vision", "deep", "neighbors", "few-shot", "numpy", "visionad"),
    metadata={
        "description": "VisionAD family placeholder for training-free search-based FSAD.",
        "paper": "Search is All You Need for Few-shot Anomaly Detection",
        "year": 2025,
        "supervision": "few-shot",
    },
)
class VisionVisionAD:
    def __init__(self, *, embedder: Any = None, search_backend: Any = None) -> None:
        self.embedder = embedder
        self.search_backend = search_backend

    def fit(self, X, y=None):
        raise NotImplementedError("vision_visionad placeholder: implementation lives on its feature branch.")

    def decision_function(self, X):
        raise NotImplementedError("vision_visionad placeholder: implementation lives on its feature branch.")
