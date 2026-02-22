# -*- coding: utf-8 -*-
"""PyOD AutoEncoder vision wrapper.

PyOD provides an `AutoEncoder` detector for generic tabular embeddings. In
PyImgAno, we apply it to image feature vectors produced by a pluggable feature
extractor.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from pyod.models.auto_encoder import AutoEncoder

from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "vision_auto_encoder",
    tags=("vision", "deep", "autoencoder", "pyod"),
    metadata={"description": "PyOD AutoEncoder wrapper (feature-based)"},
)
class VisionAutoEncoder(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's `AutoEncoder` detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        preprocessing: bool = True,
        lr: float = 1e-3,
        epoch_num: int = 10,
        batch_size: int = 32,
        optimizer_name: str = "adam",
        device: Optional[str] = None,
        random_state: int = 42,
        use_compile: bool = False,
        compile_mode: str = "default",
        verbose: int = 1,
        optimizer_params: Optional[dict[str, Any]] = None,
        hidden_neuron_list: Optional[Sequence[int]] = None,
        hidden_activation_name: str = "relu",
        batch_norm: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        self._detector_kwargs = {
            "preprocessing": bool(preprocessing),
            "lr": float(lr),
            "epoch_num": int(epoch_num),
            "batch_size": int(batch_size),
            "optimizer_name": str(optimizer_name),
            "device": device,
            "random_state": int(random_state),
            "use_compile": bool(use_compile),
            "compile_mode": str(compile_mode),
            "verbose": int(verbose),
            "hidden_activation_name": str(hidden_activation_name),
            "batch_norm": bool(batch_norm),
            "dropout_rate": float(dropout_rate),
        }
        if optimizer_params is not None:
            self._detector_kwargs["optimizer_params"] = optimizer_params
        if hidden_neuron_list is not None:
            self._detector_kwargs["hidden_neuron_list"] = list(hidden_neuron_list)

        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return AutoEncoder(contamination=self.contamination, **self._detector_kwargs)

