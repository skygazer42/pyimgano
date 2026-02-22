# -*- coding: utf-8 -*-
"""Deep Isolation Forest (DIF) vision wrapper.

DIF is implemented in PyOD. In PyImgAno, we run DIF on top of image features
produced by a pluggable feature extractor.
"""

from __future__ import annotations

from typing import Optional, Sequence

from pyod.models.dif import DIF

from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "vision_dif",
    tags=("vision", "deep", "dif", "pyod"),
    metadata={"description": "Deep Isolation Forest (DIF) wrapper via PyOD"},
)
class VisionDIF(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's DIF detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        batch_size: int = 1000,
        representation_dim: int = 20,
        hidden_neurons: Optional[Sequence[int]] = None,
        hidden_activation: str = "tanh",
        skip_connection: bool = False,
        n_ensemble: int = 50,
        n_estimators: int = 6,
        max_samples: int = 256,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self._detector_kwargs = {
            "batch_size": int(batch_size),
            "representation_dim": int(representation_dim),
            "hidden_neurons": list(hidden_neurons) if hidden_neurons is not None else None,
            "hidden_activation": str(hidden_activation),
            "skip_connection": bool(skip_connection),
            "n_ensemble": int(n_ensemble),
            "n_estimators": int(n_estimators),
            "max_samples": int(max_samples),
            "random_state": random_state,
            "device": device,
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return DIF(contamination=self.contamination, **self._detector_kwargs)

