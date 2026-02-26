# -*- coding: utf-8 -*-
"""Native detector base contract for pyimgano.

This module implements the minimal subset of the PyOD detector contract that
`pyimgano` relies on across its registry models:

- contamination-based thresholding (`threshold_`, `labels_`)
- binary predictions (`predict` returns {0,1})
- optional probability conversion (`predict_proba` is added in a later task)

Notes
-----
The design is inspired by PyOD's `BaseDetector` contract.
PyOD is BSD 2-Clause licensed.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


class BaseDetector:
    """Base class for anomaly detectors.

    Parameters
    ----------
    contamination:
        Expected proportion of outliers in (0, 0.5]. Used to derive `threshold_`
        from training scores.
    """

    def __init__(self, contamination: float = 0.1) -> None:
        if isinstance(contamination, (float, int)):
            if not (0.0 < float(contamination) <= 0.5):
                raise ValueError(
                    f"contamination must be in (0, 0.5], got: {contamination}"
                )
        self.contamination = float(contamination)

    # ---------------------------------------------------------------------
    # Subclass API
    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like signature
        raise NotImplementedError

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like signature
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Shared helpers
    def _set_n_classes(self, y: Any) -> "BaseDetector":
        """Set the number of classes (compat hook).

        In unsupervised settings this is typically binary. Some utilities rely
        on `_classes` existing.
        """

        self._classes = 2
        if y is not None:
            y_arr = np.asarray(y)
            # Best-effort: keep contract without pulling sklearn validation.
            unique = np.unique(y_arr)
            self._classes = int(len(unique))
            warnings.warn("y should not be presented in unsupervised learning.")
        return self

    def _process_decision_scores(self) -> "BaseDetector":
        """Compute threshold and training labels from `decision_scores_`."""

        if not hasattr(self, "decision_scores_"):
            raise AttributeError("decision_scores_ missing; set it before calling _process_decision_scores().")

        scores = np.asarray(self.decision_scores_, dtype=np.float64)
        if scores.ndim != 1:
            scores = scores.reshape(-1)
        self.decision_scores_ = scores

        # contamination is a float for now. (A future extension could support
        # objects with an `.eval()` method like PyThresh.)
        threshold = np.percentile(scores, 100.0 * (1.0 - float(self.contamination)))
        labels = (scores > threshold).astype(int).ravel()

        self.threshold_ = float(threshold)
        self.labels_ = labels
        return self

    # ---------------------------------------------------------------------
    def predict(self, X, return_confidence: bool = False):  # noqa: ANN001, ANN201
        """Predict binary anomaly labels.

        Returns 0 for inliers and 1 for outliers.
        """

        if return_confidence:
            raise NotImplementedError("return_confidence is not implemented in native BaseDetector")

        if not hasattr(self, "threshold_"):
            raise RuntimeError("Model must be fitted before calling predict().")

        scores = np.asarray(self.decision_function(X), dtype=np.float64)
        if scores.ndim != 1:
            scores = scores.reshape(-1)
        return (scores > float(self.threshold_)).astype(int).ravel()

    def predict_proba(  # noqa: ANN001, ANN201 - PyOD-like signature
        self,
        X,
        method: str = "linear",
        return_confidence: bool = False,
    ):
        """Predict outlier probability as a 2-class array ``[p(inlier), p(outlier)]``.

        This keeps compatibility with the PyOD `BaseDetector.predict_proba` contract.
        """

        if return_confidence:
            raise NotImplementedError(
                "return_confidence is not implemented in native BaseDetector"
            )

        if not hasattr(self, "decision_scores_"):
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        train_scores = np.asarray(self.decision_scores_, dtype=np.float64).reshape(-1)
        test_scores = np.asarray(self.decision_function(X), dtype=np.float64).reshape(-1)

        n_classes = int(getattr(self, "_classes", 2) or 2)
        if n_classes < 2:
            n_classes = 2

        probs = np.zeros((len(test_scores), n_classes), dtype=np.float64)

        if method == "linear":
            lo = float(np.min(train_scores))
            hi = float(np.max(train_scores))
            denom = hi - lo
            if denom <= 0.0:
                outlier = np.zeros_like(test_scores, dtype=np.float64)
            else:
                outlier = (test_scores - lo) / denom
                outlier = np.clip(outlier, 0.0, 1.0)
            probs[:, 1] = outlier
            probs[:, 0] = 1.0 - outlier
            return probs

        if method == "unify":
            # PyOD uses erf unification then clamps to [0,1].
            # Keep the same behavior for contract compatibility.
            try:
                from scipy.special import erf  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "scipy is required for predict_proba(method='unify'). "
                    "Install it via:\n  pip install 'scipy'\n"
                    f"Original error: {exc}"
                ) from exc

            mu = float(np.mean(train_scores))
            sigma = float(np.std(train_scores))
            denom = max(sigma * float(np.sqrt(2.0)), 1e-12)
            pre = (test_scores - mu) / denom
            outlier = np.clip(erf(pre), 0.0, 1.0)
            probs[:, 1] = outlier
            probs[:, 0] = 1.0 - outlier
            return probs

        raise ValueError(f"method {method!r} is not a valid probability conversion method")
