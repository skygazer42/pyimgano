from __future__ import annotations

from typing import Any

from .registry import register_model


@register_model(
    "vision_one_to_normal",
    tags=("vision", "deep", "reconstruction", "few-shot", "pixel_map", "numpy"),
    metadata={
        "description": "One-to-Normal family placeholder for personalized anomaly-to-normal adaptation.",
        "paper": "One-to-Normal",
        "year": 2025,
        "supervision": "few-shot",
    },
)
class VisionOneToNormal:
    def __init__(self, *, normalizer: Any = None) -> None:
        self.normalizer = normalizer

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        raise NotImplementedError(
            "vision_one_to_normal placeholder: implementation lives on its feature branch."
        )

    def predict_anomaly_map(self, X):
        raise NotImplementedError(
            "vision_one_to_normal placeholder: implementation lives on its feature branch."
        )
