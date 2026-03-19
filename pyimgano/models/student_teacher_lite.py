# -*- coding: utf-8 -*-
"""Student-Teacher lite (embedding regression residual).

This is a small, industrially practical approximation inspired by STFPM:
- Extract teacher embeddings and student embeddings
- Fit a linear map student -> teacher on normal data
- Score by residual norm ||teacher - mapped(student)||

Advantages:
- No heavy end-to-end training loop required
- Works with any registered feature extractors (torchvision, OpenCLIP, handcrafted, ...)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pyimgano.features.protocols import FittableFeatureExtractor
from pyimgano.features.registry import resolve_feature_extractor
from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .registry import register_model


def _ridge_solve(student_features: np.ndarray, teacher_features: np.ndarray, *, ridge: float) -> np.ndarray:
    """Solve W = argmin ||S W - T||^2 + ridge ||W||^2."""

    student_features = np.asarray(student_features, dtype=np.float64)
    teacher_features = np.asarray(teacher_features, dtype=np.float64)
    d = int(student_features.shape[1])
    lhs = student_features.T @ student_features + float(ridge) * np.eye(d, dtype=np.float64)
    rhs = student_features.T @ teacher_features
    weights = np.linalg.solve(lhs, rhs)
    return np.asarray(weights, dtype=np.float64)


@register_model(
    "vision_student_teacher_lite",
    tags=("vision", "classical", "embeddings", "student_teacher"),
    metadata={
        "description": "Student-Teacher lite: linear map residual between two embedding extractors",
        "type": "distillation_lite",
    },
)
class VisionStudentTeacherLite(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        teacher_extractor: Any = "torchvision_multilayer",
        teacher_kwargs: Mapping[str, Any] | None = None,
        student_extractor: Any = "torchvision_backbone",
        student_kwargs: Mapping[str, Any] | None = None,
        ridge: float = 1e-6,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.teacher_extractor = resolve_feature_extractor(
            {"name": teacher_extractor, "kwargs": dict(teacher_kwargs or {})}
            if teacher_kwargs is not None
            else teacher_extractor
        )
        self.student_extractor = resolve_feature_extractor(
            {"name": student_extractor, "kwargs": dict(student_kwargs or {})}
            if student_kwargs is not None
            else student_extractor
        )
        self.ridge = float(ridge)

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        items = list(x)
        if not items:
            raise ValueError("Training set cannot be empty")

        self._set_n_classes(y)

        # Fit feature extractors if they support it.
        if isinstance(self.teacher_extractor, FittableFeatureExtractor):
            self.teacher_extractor.fit(items, y=y)
        if isinstance(self.student_extractor, FittableFeatureExtractor):
            self.student_extractor.fit(items, y=y)

        teacher_features = np.asarray(self.teacher_extractor.extract(items), dtype=np.float64)
        student_features = np.asarray(self.student_extractor.extract(items), dtype=np.float64)
        if teacher_features.ndim == 1:
            teacher_features = teacher_features.reshape(-1, 1)
        if student_features.ndim == 1:
            student_features = student_features.reshape(-1, 1)
        if teacher_features.shape[0] != student_features.shape[0]:
            raise ValueError("teacher and student extractors must return same number of rows")

        weights = _ridge_solve(student_features, teacher_features, ridge=float(self.ridge))
        resid = teacher_features - (student_features @ weights)
        scores = np.linalg.norm(resid, axis=1)

        self.W_ = weights
        self.decision_scores_ = np.asarray(scores, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201
        require_fitted(self, ["W_"])
        items = list(x)
        if not items:
            return np.zeros((0,), dtype=np.float64)

        teacher_features = np.asarray(self.teacher_extractor.extract(items), dtype=np.float64)
        student_features = np.asarray(self.student_extractor.extract(items), dtype=np.float64)
        if teacher_features.ndim == 1:
            teacher_features = teacher_features.reshape(-1, 1)
        if student_features.ndim == 1:
            student_features = student_features.reshape(-1, 1)

        weights = np.asarray(self.W_, dtype=np.float64)  # type: ignore[attr-defined]
        resid = teacher_features - (student_features @ weights)
        scores = np.linalg.norm(resid, axis=1)
        return np.asarray(scores, dtype=np.float64).reshape(-1)
