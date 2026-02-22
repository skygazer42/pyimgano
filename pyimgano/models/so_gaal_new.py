# -*- coding: utf-8 -*-
"""SO-GAAL (new implementation) vision wrapper.

PyOD provides both `so_gaal` and `so_gaal_new` modules. This wrapper exposes the
`so_gaal_new` variant under a separate registry name.
"""

from __future__ import annotations

from typing import Optional

from pyod.models.so_gaal_new import SO_GAAL as SO_GAAL_NEW

from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "vision_so_gaal_new",
    tags=("vision", "deep", "gan", "so_gaal", "pyod"),
    metadata={"description": "SO-GAAL (new) wrapper via PyOD"},
)
class VisionSOGaalNew(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's `so_gaal_new` implementation."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        preprocessing: bool = True,
        epoch_num: int = 100,
        criterion_name: str = "bce",
        device: Optional[str] = None,
        random_state: int = 42,
        use_compile: bool = False,
        compile_mode: str = "default",
        verbose: int = 1,
        lr_d: float = 0.01,
        lr_g: float = 0.0001,
        momentum: float = 0.9,
    ) -> None:
        self._detector_kwargs = {
            "preprocessing": bool(preprocessing),
            "epoch_num": int(epoch_num),
            "criterion_name": str(criterion_name),
            "device": device,
            "random_state": int(random_state),
            "use_compile": bool(use_compile),
            "compile_mode": str(compile_mode),
            "verbose": int(verbose),
            "lr_d": float(lr_d),
            "lr_g": float(lr_g),
            "momentum": float(momentum),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return SO_GAAL_NEW(contamination=self.contamination, **self._detector_kwargs)

