from __future__ import annotations

from typing import Any

from .registry import register_model


@register_model(
    "vision_adaclip",
    tags=("vision", "deep", "clip", "zero-shot", "numpy", "adaclip"),
    metadata={
        "description": "AdaCLIP family placeholder for hybrid prompt zero-shot anomaly detection.",
        "paper": "AdaCLIP",
        "year": 2024,
        "supervision": "zero-shot",
    },
)
class VisionAdaCLIP:
    def __init__(self, *, clip_backend: Any = None, prompt_backend: Any = None) -> None:
        self.clip_backend = clip_backend
        self.prompt_backend = prompt_backend

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        raise NotImplementedError("vision_adaclip placeholder: implementation lives on its feature branch.")
