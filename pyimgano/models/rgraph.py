# -*- coding: utf-8 -*-
"""RGraph vision wrapper.

RGraph is implemented in PyOD. In PyImgAno, we apply it to image feature vectors
produced by a pluggable feature extractor.
"""

from __future__ import annotations

from pyod.models.rgraph import RGraph

from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "vision_rgraph",
    tags=("vision", "classical", "rgraph", "pyod"),
    metadata={"description": "RGraph wrapper via PyOD"},
)
class VisionRGraph(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's RGraph detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        transition_steps: int = 10,
        n_nonzero: int = 10,
        gamma: float = 50.0,
        gamma_nz: bool = True,
        algorithm: str = "lasso_lars",
        tau: float = 1.0,
        maxiter_lasso: int = 1000,
        preprocessing: bool = True,
        blocksize_test_data: int = 10,
        support_init: str = "L2",
        maxiter: int = 40,
        support_size: int = 100,
        active_support: bool = True,
        fit_intercept_LR: bool = False,
        verbose: bool = True,
    ) -> None:
        self._detector_kwargs = {
            "transition_steps": int(transition_steps),
            "n_nonzero": int(n_nonzero),
            "gamma": float(gamma),
            "gamma_nz": bool(gamma_nz),
            "algorithm": str(algorithm),
            "tau": float(tau),
            "maxiter_lasso": int(maxiter_lasso),
            "preprocessing": bool(preprocessing),
            "blocksize_test_data": int(blocksize_test_data),
            "support_init": str(support_init),
            "maxiter": int(maxiter),
            "support_size": int(support_size),
            "active_support": bool(active_support),
            "fit_intercept_LR": bool(fit_intercept_LR),
            "verbose": bool(verbose),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return RGraph(contamination=self.contamination, **self._detector_kwargs)

