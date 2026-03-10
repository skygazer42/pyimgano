from __future__ import annotations

from typing import Any

from .registry import register_model


@register_model(
    "vision_anogen_adapter",
    tags=("vision", "deep", "reconstruction", "few-shot", "numpy", "anogen"),
    metadata={
        "description": "AnoGen family placeholder for anomaly-driven generation adapters.",
        "paper": "Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation",
        "year": 2025,
        "supervision": "few-shot",
    },
)
class VisionAnoGenAdapter:
    def __init__(self, *, generator: Any = None, scoring_backend: Any = None) -> None:
        self.generator = generator
        self.scoring_backend = scoring_backend

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        raise NotImplementedError(
            "vision_anogen_adapter placeholder: implementation lives on its feature branch."
        )
