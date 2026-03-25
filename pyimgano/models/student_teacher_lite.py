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

from pathlib import Path
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.protocols import FittableFeatureExtractor
from pyimgano.features.registry import FEATURE_REGISTRY, resolve_feature_extractor
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


def _feature_extractor_to_spec(extractor: Any) -> dict[str, Any]:
    if not isinstance(extractor, BaseFeatureExtractor):
        raise TypeError(
            "StudentTeacherLite checkpointing requires feature extractors derived from "
            f"BaseFeatureExtractor, got {type(extractor).__name__}"
        )

    name = None
    for entry in FEATURE_REGISTRY._registry.values():  # noqa: SLF001 - internal registry walk
        ctor = entry.constructor
        if isinstance(ctor, type) and isinstance(extractor, ctor):
            name = str(entry.name)
            break
    if name is None:
        raise ValueError(
            "Unable to serialize feature extractor because it is not registered: "
            f"{type(extractor).__module__}.{type(extractor).__name__}"
        )

    params = {}
    for key, value in dict(extractor.get_params(deep=False)).items():
        if isinstance(value, BaseFeatureExtractor):
            params[str(key)] = _feature_extractor_to_spec(value)
        else:
            params[str(key)] = value

    state: dict[str, Any] = {}
    for key, value in dict(getattr(extractor, "__dict__", {})).items():
        if key in params:
            continue
        if key == "_pca":
            state[key] = value
            continue
        if str(key).startswith("_"):
            continue
        if isinstance(value, ModuleType) or callable(value):
            continue
        state[key] = value

    model = getattr(extractor, "_model", None)
    state_dict = getattr(model, "state_dict", None)
    if model is not None and callable(state_dict):
        raw_state = state_dict()
        normalized_state = {}
        for key, value in dict(raw_state).items():
            detach = getattr(value, "detach", None)
            cpu = getattr(value, "cpu", None)
            if callable(detach) and callable(cpu):
                try:
                    normalized_state[str(key)] = detach().cpu()
                    continue
                except Exception:
                    pass
            normalized_state[str(key)] = value
        state["_model_state_dict"] = normalized_state

    return {"name": name, "kwargs": params, "state": state}


def _feature_extractor_from_spec(payload: Mapping[str, Any]) -> BaseFeatureExtractor:
    spec = {
        "name": str(payload["name"]),
        "kwargs": dict(payload.get("kwargs", {})),
    }
    extractor = resolve_feature_extractor(spec)
    if not isinstance(extractor, BaseFeatureExtractor):
        raise TypeError(
            "Resolved feature extractor does not implement BaseFeatureExtractor: "
            f"{type(extractor).__name__}"
        )

    state = payload.get("state", {})
    if isinstance(state, Mapping):
        for key, value in state.items():
            if str(key) == "_model_state_dict":
                continue
            setattr(extractor, str(key), value)
        model_state = state.get("_model_state_dict", None)
        ensure_ready = getattr(extractor, "_ensure_ready", None)
        if isinstance(model_state, Mapping) and callable(ensure_ready):
            ensure_ready()
            model = getattr(extractor, "_model", None)
            load_state_dict = getattr(model, "load_state_dict", None)
            if callable(load_state_dict):
                load_state_dict(dict(model_state), strict=False)
    return extractor


@register_model(
    "vision_student_teacher_lite",
    tags=("vision", "classical", "embeddings", "student_teacher"),
    metadata={
        "description": "Student-Teacher lite: linear map residual between two embedding extractors",
        "type": "distillation_lite",
        "supports_save_load": False,
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

    def save_checkpoint(self, path: str | Path) -> Path:
        require_fitted(self, ["W_"])

        from pyimgano.models.serialization import save_model

        payload = {
            "schema_version": 1,
            "detector": "vision_student_teacher_lite",
            "config": {
                "contamination": float(self.contamination),
                "ridge": float(self.ridge),
            },
            "teacher_extractor": _feature_extractor_to_spec(self.teacher_extractor),
            "student_extractor": _feature_extractor_to_spec(self.student_extractor),
            "state": {
                "W_": np.asarray(self.W_, dtype=np.float64),  # type: ignore[attr-defined]
                "decision_scores_": np.asarray(self.decision_scores_, dtype=np.float64),
                "threshold_": (
                    float(self.threshold_) if getattr(self, "threshold_", None) is not None else None
                ),
                "labels_": (
                    np.asarray(self.labels_, dtype=np.int64)
                    if getattr(self, "labels_", None) is not None
                    else None
                ),
                "_classes": getattr(self, "_classes", None),
                "threshold_method": str(getattr(self, "threshold_method", "quantile")),
                "pot_tail_fraction": float(getattr(self, "pot_tail_fraction", 0.1)),
                "pot_min_exceedances": int(getattr(self, "pot_min_exceedances", 20)),
            },
        }
        return save_model(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        from pyimgano.models.serialization import load_model

        payload = load_model(path)
        if not isinstance(payload, Mapping):
            raise ValueError("Invalid StudentTeacherLite checkpoint payload: expected a mapping.")
        if str(payload.get("detector", "")) != "vision_student_teacher_lite":
            raise ValueError(
                "Invalid StudentTeacherLite checkpoint payload: detector marker mismatch."
            )

        teacher_payload = payload.get("teacher_extractor", None)
        student_payload = payload.get("student_extractor", None)
        if not isinstance(teacher_payload, Mapping) or not isinstance(student_payload, Mapping):
            raise ValueError("Invalid StudentTeacherLite checkpoint payload: missing extractor specs.")

        self.teacher_extractor = _feature_extractor_from_spec(teacher_payload)
        self.student_extractor = _feature_extractor_from_spec(student_payload)

        config_payload = payload.get("config", None)
        if isinstance(config_payload, Mapping):
            contamination = config_payload.get("contamination", None)
            ridge = config_payload.get("ridge", None)
            if contamination is not None:
                self.contamination = float(contamination)
            if ridge is not None:
                self.ridge = float(ridge)

        state = payload.get("state", None)
        if not isinstance(state, Mapping):
            raise ValueError("Invalid StudentTeacherLite checkpoint payload: missing detector state.")

        self.W_ = np.asarray(state["W_"], dtype=np.float64)
        self.decision_scores_ = np.asarray(state["decision_scores_"], dtype=np.float64)
        threshold = state.get("threshold_", None)
        if threshold is not None:
            self.threshold_ = float(threshold)
        labels = state.get("labels_", None)
        if labels is not None:
            self.labels_ = np.asarray(labels, dtype=np.int64)
        classes = state.get("_classes", None)
        if classes is not None:
            self._classes = int(classes)
        self.threshold_method = str(state.get("threshold_method", "quantile"))
        self.pot_tail_fraction = float(state.get("pot_tail_fraction", 0.1))
        self.pot_min_exceedances = int(state.get("pot_min_exceedances", 20))
