from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from sklearn.utils import check_array

from .base_detector import BaseDetector


def _looks_like_torch_tensor(x: Any) -> bool:
    """Best-effort torch.Tensor detection without importing torch.

    This avoids optional-import `try/except` blocks in core codepaths while still
    supporting torch tensors when torch is installed.
    """

    mod = getattr(getattr(x, "__class__", None), "__module__", "")
    if not isinstance(mod, str) or not mod.startswith("torch"):
        return False
    return bool(hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"))


def _to_numpy_2d(X: Any) -> np.ndarray:
    """Convert feature-matrix inputs to a 2D numpy array (best-effort).

    Supports:
    - numpy arrays
    - torch tensors (if torch is installed)
    - array-like inputs convertible via `np.asarray`
    """

    if isinstance(X, np.ndarray):
        arr = X
    else:
        if _looks_like_torch_tensor(X):  # pragma: no cover - depends on torch being available
            arr = X.detach().cpu().numpy()
        else:
            arr = np.asarray(X)

    arr = check_array(arr, ensure_2d=True, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


class CoreFeatureDetector(BaseDetector):
    """Base class for `core_*` feature-matrix detectors.

    Subclasses implement `_build_detector()` which returns an object supporting:
    - `.fit(X)` and `.decision_function(X) -> scores`
    - optionally `.decision_scores_` after fitting
    """

    def __init__(self, *, contamination: float = 0.1) -> None:
        super().__init__(contamination=float(contamination))
        self.detector = self._build_detector()

    @abstractmethod
    def _build_detector(self):  # noqa: ANN201 - generic backend object
        raise NotImplementedError

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        x_array = _to_numpy_2d(X)
        self.detector.fit(x_array)

        if hasattr(self.detector, "decision_scores_"):
            scores = np.asarray(
                getattr(self.detector, "decision_scores_"), dtype=np.float64
            ).reshape(-1)
        else:
            scores = np.asarray(self.detector.decision_function(x_array), dtype=np.float64).reshape(-1)

        if scores.shape[0] != x_array.shape[0]:
            raise ValueError(
                "Detector decision_scores_ must have one score per training sample. "
                f"Got {scores.shape[0]} for {x_array.shape[0]} samples."
            )

        self.decision_scores_ = scores
        self._process_decision_scores()
        self._set_n_classes(y)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        x_array = _to_numpy_2d(X)
        scores = np.asarray(self.detector.decision_function(x_array), dtype=np.float64).reshape(-1)
        if scores.shape[0] != x_array.shape[0]:
            raise ValueError(
                "Detector decision_function must return one score per sample. "
                f"Got {scores.shape[0]} for {x_array.shape[0]} samples."
            )
        return scores
