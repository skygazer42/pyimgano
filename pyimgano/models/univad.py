from __future__ import annotations

from typing import Any

from .registry import register_model


@register_model(
    "vision_univad",
    tags=("vision", "deep", "neighbors", "few-shot", "numpy", "univad"),
    metadata={
        "description": "UniVAD family placeholder for unified few-shot anomaly detection.",
        "paper": "UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection",
        "year": 2025,
        "supervision": "few-shot",
    },
)
class VisionUniVAD:
    def __init__(self, *, feature_extractor: Any = None, support_backend: Any = None) -> None:
        self.feature_extractor = feature_extractor
        self.support_backend = support_backend

    def fit(self, X, y=None):
        raise NotImplementedError("vision_univad placeholder: implementation lives on its feature branch.")

    def decision_function(self, X):
        raise NotImplementedError("vision_univad placeholder: implementation lives on its feature branch.")
