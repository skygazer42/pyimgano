# -*- coding: utf-8 -*-
"""Score-standardizer wrapper for vision detectors.

This wraps any `vision_*` detector and standardizes its output scores in an
unsupervised way (rank/zscore/robust/minmax) based on training scores.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from pyimgano.calibration.score_standardization import ScoreStandardizer
from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .registry import create_model, register_model


def _as_kwargs(obj: Mapping[str, Any] | None) -> dict[str, Any]:
    if obj is None:
        return {}
    return dict(obj)


@register_model(
    "vision_score_standardizer",
    tags=("vision", "wrapper", "calibration", "score"),
    metadata={
        "description": "Wrap a vision detector and standardize its scores (rank/zscore/robust/minmax)",
        "type": "wrapper",
    },
)
class VisionScoreStandardizer(BaseDetector):
    def __init__(
        self,
        *,
        base_detector: str | type | Any = "vision_knn",
        base_kwargs: Mapping[str, Any] | None = None,
        method: str = "rank",
        eps: float = 1e-12,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.base_detector = base_detector
        self.base_kwargs = _as_kwargs(base_kwargs)
        self.method = str(method)
        self.eps = float(eps)

        self.base_model_ = None
        self.standardizer_ = None

    def _build_base(self):
        spec = self.base_detector
        if isinstance(spec, str):
            if spec == "vision_score_standardizer":
                raise ValueError("vision_score_standardizer cannot wrap itself")
            return create_model(
                spec, contamination=float(self.contamination), **dict(self.base_kwargs)
            )
        if isinstance(spec, type):
            return spec(contamination=float(self.contamination), **dict(self.base_kwargs))
        return spec

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        base = self._build_base()
        fit = getattr(base, "fit", None)
        if not callable(fit):
            raise TypeError("base_detector must provide a .fit(X) method")

        try:
            fit(X, y=y)
        except TypeError:
            fit(X)

        if hasattr(base, "decision_scores_"):
            train_scores = np.asarray(getattr(base, "decision_scores_"), dtype=np.float64).reshape(
                -1
            )
        else:
            train_scores = np.asarray(base.decision_function(X), dtype=np.float64).reshape(-1)

        std = ScoreStandardizer(method=str(self.method), eps=float(self.eps)).fit(train_scores)
        self.base_model_ = base
        self.standardizer_ = std

        self.decision_scores_ = std.transform(train_scores)
        self._process_decision_scores()
        self._set_n_classes(y)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["base_model_", "standardizer_"])
        base = self.base_model_
        std: ScoreStandardizer = self.standardizer_  # type: ignore[assignment]
        scores = np.asarray(base.decision_function(X), dtype=np.float64).reshape(-1)  # type: ignore[union-attr]
        return std.transform(scores)
