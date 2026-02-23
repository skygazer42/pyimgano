from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


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
        from pyimgano.models.registry import create_model

        detector = create_model(self.model, **dict(self._model_kwargs))
        try:
            detector.fit(X, y)
        except TypeError:
            detector.fit(X)
        self.detector_ = detector
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn signature
        detector = getattr(self, "detector_", None)
        if detector is None:
            raise AttributeError("Estimator is not fitted. Call fit() before decision_function().")
        scores = detector.decision_function(X)
        arr = np.asarray(scores)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    def predict(self, X):  # noqa: ANN001, ANN201 - sklearn signature
        detector = getattr(self, "detector_", None)
        if detector is None:
            raise AttributeError("Estimator is not fitted. Call fit() before predict().")

        if hasattr(detector, "predict"):
            preds = detector.predict(X)
            arr = np.asarray(preds, dtype=int)
            if arr.ndim != 1:
                arr = arr.reshape(-1)

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
        scores = self.decision_function(X)
        return (scores >= float(threshold)).astype(int)

