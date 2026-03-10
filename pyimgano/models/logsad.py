from __future__ import annotations

from typing import Any

from .registry import register_model


@register_model(
    "vision_logsad",
    tags=("vision", "deep", "zero-shot", "pixel_map", "numpy", "logsad"),
    metadata={
        "description": "LogSAD family placeholder for logical and structural anomaly detection.",
        "paper": "Towards Training-free Anomaly Detection with Vision and Language Foundation Models",
        "year": 2025,
        "supervision": "zero-shot",
    },
)
class VisionLogSAD:
    def __init__(self, *, logic_backend: Any = None, visual_backend: Any = None) -> None:
        self.logic_backend = logic_backend
        self.visual_backend = visual_backend

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        raise NotImplementedError("vision_logsad placeholder: implementation lives on its feature branch.")

    def predict_anomaly_map(self, X):
        raise NotImplementedError("vision_logsad placeholder: implementation lives on its feature branch.")
