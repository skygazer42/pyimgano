from __future__ import annotations

from typing import Any

from .registry import register_model


@register_model(
    "vision_filopp",
    tags=("vision", "deep", "clip", "pixel_map", "zero-shot", "numpy", "filopp"),
    metadata={
        "description": "FiLo++ family placeholder for fine-grained VLM anomaly localization.",
        "paper": "FiLo++",
        "year": 2025,
        "supervision": "zero-shot",
    },
)
class VisionFiLoPP:
    def __init__(self, *, vlm_backend: Any = None, localization_backend: Any = None) -> None:
        self.vlm_backend = vlm_backend
        self.localization_backend = localization_backend

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        raise NotImplementedError("vision_filopp placeholder: implementation lives on its feature branch.")

    def predict_anomaly_map(self, X):
        raise NotImplementedError("vision_filopp placeholder: implementation lives on its feature branch.")
