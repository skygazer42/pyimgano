from __future__ import annotations

from typing import Any

from .registry import register_model


@register_model(
    "vision_aaclip",
    tags=("vision", "deep", "clip", "pixel_map", "zero-shot", "numpy", "aaclip"),
    metadata={
        "description": "AA-CLIP family placeholder for anomaly-aware CLIP anomaly detection.",
        "paper": "AA-CLIP",
        "year": 2025,
        "supervision": "zero-shot",
    },
)
class VisionAAClip:
    def __init__(self, *, clip_backend: Any = None, anchor_backend: Any = None) -> None:
        self.clip_backend = clip_backend
        self.anchor_backend = anchor_backend

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        raise NotImplementedError("vision_aaclip placeholder: implementation lives on its feature branch.")

    def predict_anomaly_map(self, X):
        raise NotImplementedError("vision_aaclip placeholder: implementation lives on its feature branch.")
