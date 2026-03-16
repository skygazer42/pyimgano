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


def _ridge_solve(
    student_embeddings: np.ndarray,
    teacher_embeddings: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Solve W = argmin ||S W - T||^2 + ridge ||W||^2."""

    student = np.asarray(student_embeddings, dtype=np.float64)
    teacher = np.asarray(teacher_embeddings, dtype=np.float64)
    d = int(student.shape[1])
    a_mat = student.T @ student + float(ridge) * np.eye(d, dtype=np.float64)
    b_mat = student.T @ teacher
    weights = np.linalg.solve(a_mat, b_mat)
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

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        items = list(X)
        if not items:
            raise ValueError("Training set cannot be empty")

        self._set_n_classes(y)

        # Fit feature extractors if they support it.
        if isinstance(self.teacher_extractor, FittableFeatureExtractor):
            self.teacher_extractor.fit(items, y=y)
        if isinstance(self.student_extractor, FittableFeatureExtractor):
            self.student_extractor.fit(items, y=y)

        T = np.asarray(self.teacher_extractor.extract(items), dtype=np.float64)
        S = np.asarray(self.student_extractor.extract(items), dtype=np.float64)
        if T.ndim == 1:
            T = T.reshape(-1, 1)
        if S.ndim == 1:
            S = S.reshape(-1, 1)
        if T.shape[0] != S.shape[0]:
            raise ValueError("teacher and student extractors must return same number of rows")

        W = _ridge_solve(S, T, ridge=float(self.ridge))
        resid = T - (S @ W)
        scores = np.linalg.norm(resid, axis=1)

        self.W_ = W
        self.decision_scores_ = np.asarray(scores, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["W_"])
        items = list(X)
        if not items:
            return np.zeros((0,), dtype=np.float64)

        T = np.asarray(self.teacher_extractor.extract(items), dtype=np.float64)
        S = np.asarray(self.student_extractor.extract(items), dtype=np.float64)
        if T.ndim == 1:
            T = T.reshape(-1, 1)
        if S.ndim == 1:
            S = S.reshape(-1, 1)

        W = np.asarray(self.W_, dtype=np.float64)  # type: ignore[attr-defined]
        resid = T - (S @ W)
        scores = np.linalg.norm(resid, axis=1)
        return np.asarray(scores, dtype=np.float64).reshape(-1)
