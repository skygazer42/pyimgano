from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class RegistryModelEstimator(BaseEstimator):
    """scikit-learn compatible estimator for registry models.

    This adapter provides a minimal sklearn/PyOD-like contract for `pyimgano`
    registry models, without requiring every detector implementation to inherit
    from sklearn's base classes.

    Notes
    -----
    - This is a light wrapper: it constructs the underlying detector via
      `pyimgano.models.create_model(...)` inside `fit()`.
    - `sklearn.base.clone()` is supported via a custom `get_params/set_params`
      implementation that exposes constructor kwargs as flat params.
    """

    def __init__(self, model: str, **model_kwargs: Any) -> None:
        self.model = str(model)
        self._model_kwargs: dict[str, Any] = dict(model_kwargs)

    @staticmethod
    def _normalize_X(X, *, name: str):  # noqa: ANN001, ANN201 - sklearn signature
        if X is None:
            raise TypeError(f"{name} must not be None.")
        if isinstance(X, (str, Path)):
            raise TypeError(
                f"{name} must be an array-like or an iterable of samples; got a single path-like "
                f"({type(X).__name__}). Wrap it in a list, e.g. [{name}]."
            )
        if isinstance(X, np.ndarray):
            if X.shape[0] == 0:
                raise ValueError(f"{name} must be non-empty.")
            return X

        # sklearn style: accept iterables, but reject non-iterables early.
        if not isinstance(X, Iterable):
            raise TypeError(f"{name} must be an array-like or an iterable of samples.")

        items = list(X)
        if not items:
            raise ValueError(f"{name} must be non-empty.")
        return items

    @staticmethod
    def _num_samples(X) -> int | None:  # noqa: ANN001 - sklearn input types
        try:
            return int(len(X))
        except Exception:
            return None

    @classmethod
    def _ensure_1d_scores(cls, scores, *, expected: int | None) -> np.ndarray:  # noqa: ANN001
        arr = np.asarray(scores, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if expected is not None and arr.shape[0] != expected:
            raise ValueError(
                "Underlying detector returned an unexpected number of scores: "
                f"expected {expected}, got {arr.shape[0]}."
            )
        return arr

    # ---------------------------------------------------------------------
    # sklearn parameter protocol
    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002 - sklearn signature
        params = {"model": self.model}
        params.update(dict(self._model_kwargs))
        return params

    def set_params(self, **params: Any) -> "RegistryModelEstimator":
        if "model" in params:
            self.model = str(params.pop("model"))
        for key, value in params.items():
            self._model_kwargs[str(key)] = value
        return self

    # ---------------------------------------------------------------------
    # estimator behavior
    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn signature
        import pyimgano.models  # noqa: F401 - registry population side effects
        from pyimgano.models.registry import create_model, list_models

        X_norm = self._normalize_X(X, name="X")

        try:
            detector = create_model(self.model, **dict(self._model_kwargs))
        except KeyError as exc:
            available = ", ".join(list_models()[:25])
            suffix = "" if len(list_models()) <= 25 else ", ..."
            raise ValueError(
                f"Unknown model name: {self.model!r}. "
                f"Available (partial): {available}{suffix}"
            ) from exc

        try:
            detector.fit(X_norm, y)
        except TypeError:
            detector.fit(X_norm)
        self.detector_ = detector
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn signature
        detector = getattr(self, "detector_", None)
        if detector is None:
            raise NotFittedError("Estimator is not fitted. Call fit() before decision_function().")

        X_norm = self._normalize_X(X, name="X")
        expected = self._num_samples(X_norm)
        scores = detector.decision_function(X_norm)
        return self._ensure_1d_scores(scores, expected=expected)

    def predict(self, X):  # noqa: ANN001, ANN201 - sklearn signature
        detector = getattr(self, "detector_", None)
        if detector is None:
            raise NotFittedError("Estimator is not fitted. Call fit() before predict().")

        X_norm = self._normalize_X(X, name="X")
        expected = self._num_samples(X_norm)

        if hasattr(detector, "predict"):
            preds = detector.predict(X_norm)
            arr = np.asarray(preds, dtype=int)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if expected is not None and arr.shape[0] != expected:
                raise ValueError(
                    "Underlying detector returned an unexpected number of predictions: "
                    f"expected {expected}, got {arr.shape[0]}."
                )

            # Normalize {1,-1} (sklearn-style) to {0,1} (pyod-style).
            unique = set(np.unique(arr).tolist())
            if unique.issubset({-1, 1}):
                return (arr == -1).astype(int)

            return arr

        threshold = getattr(detector, "threshold_", None)
        if threshold is None:
            raise AttributeError(
                "Underlying detector does not expose predict() or threshold_."
            )
        scores = self.decision_function(X_norm)
        return (scores >= float(threshold)).astype(int)

    def score_samples(self, X):  # noqa: ANN001, ANN201 - sklearn signature
        """Alias for `decision_function` (sklearn convention)."""

        return self.decision_function(X)
