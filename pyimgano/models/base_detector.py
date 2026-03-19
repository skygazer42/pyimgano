# -*- coding: utf-8 -*-
"""Native detector base contract for pyimgano.

This module implements a small, dependency-light anomaly detector contract used
across `pyimgano` registry models:

- contamination-based thresholding (`threshold_`, `labels_`)
- binary predictions (`predict` returns {0,1})
- optional probability conversion (`predict_proba`)

All detectors follow the same scoring convention:
**higher score => more anomalous**.
"""

from __future__ import annotations

import inspect
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

    input_mode = "features"

    def __init__(self, contamination: float = 0.1) -> None:
        if isinstance(contamination, (float, int)):
            if not (0.0 < float(contamination) <= 0.5):
                raise ValueError(f"contamination must be in (0, 0.5], got: {contamination}")
        self.contamination = float(contamination)
        # Thresholding configuration (default: quantile based on contamination).
        # This is intentionally attribute-based so subclasses don't need to
        # thread extra kwargs through every constructor.
        self.threshold_method = "quantile"  # quantile|pot
        self.pot_tail_fraction = 0.1
        self.pot_min_exceedances = 20

    # ---------------------------------------------------------------------
    # Subclass API
    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like signature
        raise NotImplementedError

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like signature
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
            raise AttributeError(
                "decision_scores_ missing; set it before calling _process_decision_scores()."
            )

        scores = np.asarray(self.decision_scores_, dtype=np.float64)
        if scores.ndim != 1:
            scores = scores.reshape(-1)
        self.decision_scores_ = scores

        method = str(getattr(self, "threshold_method", "quantile")).strip().lower()

        if method in {"quantile", "percentile"}:
            threshold = np.percentile(scores, 100.0 * (1.0 - float(self.contamination)))
        elif method in {"pot", "peak_over_threshold", "peaks_over_threshold"}:
            from pyimgano.calibration.pot_threshold import fit_pot_threshold

            threshold, info = fit_pot_threshold(
                scores,
                alpha=float(self.contamination),
                tail_fraction=float(getattr(self, "pot_tail_fraction", 0.1)),
                min_exceedances=int(getattr(self, "pot_min_exceedances", 20)),
            )
            # Expose best-effort diagnostics for users.
            self.pot_info_ = info
        else:
            raise ValueError(f"Unknown threshold_method: {method!r}. Use 'quantile' or 'pot'.")

        labels = (scores > threshold).astype(int).ravel()

        self.threshold_ = float(threshold)
        self.labels_ = labels
        return self

    def use_pot_thresholding(
        self,
        *,
        tail_fraction: float = 0.1,
        min_exceedances: int = 20,
    ) -> "BaseDetector":
        """Enable POT thresholding for the next call to `_process_decision_scores()`."""

        self.threshold_method = "pot"
        self.pot_tail_fraction = float(tail_fraction)
        self.pot_min_exceedances = int(min_exceedances)
        return self

    def use_quantile_thresholding(self) -> "BaseDetector":
        """Use simple quantile thresholding derived from `contamination`."""

        self.threshold_method = "quantile"
        return self

    # ---------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return init parameters (sklearn-compatible best-effort)."""

        sig = inspect.signature(self.__class__.__init__)
        out: dict[str, Any] = {}
        for name in sig.parameters:
            if name == "self":
                continue
            if hasattr(self, name):
                out[name] = getattr(self, name)

        if not deep:
            return out

        # Nested params (very small subset of sklearn's behavior).
        nested: dict[str, Any] = {}
        for k, v in out.items():
            if hasattr(v, "get_params") and callable(getattr(v, "get_params")):
                try:
                    for nk, nv in v.get_params(deep=True).items():  # type: ignore[call-arg]
                        nested[f"{k}__{nk}"] = nv
                except Exception:
                    continue
        out.update(nested)
        return out

    def set_params(self, **params: Any) -> "BaseDetector":
        """Set init parameters (sklearn-compatible best-effort)."""

        valid = self.get_params(deep=False)
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"Invalid parameter {k!r} for {self.__class__.__name__}.")
            setattr(self, k, v)
        return self

    def _linear_outlier_probability(
        self,
        scores: Any,
        *,
        train_scores: Any | None = None,
    ) -> np.ndarray:
        if train_scores is None:
            if not hasattr(self, "decision_scores_"):
                raise RuntimeError("Model must be fitted before calling predict_proba().")
            train_scores = self.decision_scores_

        train_scores_arr = np.asarray(train_scores, dtype=np.float64).reshape(-1)
        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        lo = float(np.min(train_scores_arr))
        hi = float(np.max(train_scores_arr))
        denom = hi - lo
        if denom <= 0.0:
            return np.full(scores_arr.shape, 0.5, dtype=np.float64)
        outlier = (scores_arr - lo) / denom
        return np.clip(outlier, 0.0, 1.0)

    def _label_confidence_from_scores(
        self,
        scores: Any,
        *,
        labels: Any | None = None,
    ) -> np.ndarray:
        """Estimate predicted-label confidence on ``[0, 1]``."""

        if not hasattr(self, "decision_scores_"):
            raise RuntimeError("Model must be fitted before calling confidence helpers.")

        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        train_scores = np.asarray(self.decision_scores_, dtype=np.float64).reshape(-1)
        threshold = getattr(self, "threshold_", None)

        if labels is None:
            if threshold is not None:
                labels_arr = (scores_arr > float(threshold)).astype(np.int64)
            else:
                probs = self._linear_outlier_probability(scores_arr, train_scores=train_scores)
                labels_arr = (probs >= 0.5).astype(np.int64)
                return np.where(labels_arr == 1, probs, 1.0 - probs)
        else:
            labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)

        if labels_arr.shape[0] != scores_arr.shape[0]:
            raise ValueError("labels must have the same length as scores")

        if threshold is None:
            probs = self._linear_outlier_probability(scores_arr, train_scores=train_scores)
            return np.where(labels_arr == 1, probs, 1.0 - probs)

        lo = float(np.min(train_scores))
        hi = float(np.max(train_scores))
        thr = float(threshold)
        inlier_span = max(thr - lo, 1e-12)
        outlier_span = max(hi - thr, 1e-12)

        conf = np.empty_like(scores_arr, dtype=np.float64)
        inlier_mask = labels_arr <= 0
        if np.any(inlier_mask):
            conf[inlier_mask] = (thr - scores_arr[inlier_mask]) / inlier_span
        if np.any(~inlier_mask):
            conf[~inlier_mask] = (scores_arr[~inlier_mask] - thr) / outlier_span
        return np.clip(conf, 0.0, 1.0)

    def predict_confidence(self, x):  # noqa: ANN001, ANN201
        """Return predicted-label confidence on ``[0, 1]``."""

        scores = np.asarray(self.decision_function(x), dtype=np.float64).reshape(-1)
        labels = None
        if hasattr(self, "threshold_"):
            labels = (scores > float(self.threshold_)).astype(np.int64)
        return self._label_confidence_from_scores(scores, labels=labels)

    def predict(self, x, return_confidence: bool = False):  # noqa: ANN001, ANN201
        """Predict binary anomaly labels.

        Returns 0 for inliers and 1 for outliers.
        """

        if not hasattr(self, "threshold_"):
            raise RuntimeError("Model must be fitted before calling predict().")

        scores = np.asarray(self.decision_function(x), dtype=np.float64)
        if scores.ndim != 1:
            scores = scores.reshape(-1)
        labels = (scores > float(self.threshold_)).astype(int).ravel()
        if not bool(return_confidence):
            return labels

        conf = self._label_confidence_from_scores(scores, labels=labels)
        return labels, conf

    def predict_with_rejection(
        self,
        x,
        *,
        confidence_threshold: float,
        reject_label: int = -2,
        return_confidence: bool = False,
    ):  # noqa: ANN001, ANN201
        if not 0.0 < float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1].")

        labels, conf = self.predict(x, return_confidence=True)
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
        conf_arr = np.asarray(conf, dtype=np.float64).reshape(-1)
        labels_arr[conf_arr < float(confidence_threshold)] = int(reject_label)
        if not bool(return_confidence):
            return labels_arr
        return labels_arr, conf_arr

    def fit_predict(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like helper
        """Fit detector then return labels for X."""

        self.fit(x, y=y)
        return self.predict(x)

    def score_samples(self, x):  # noqa: ANN001, ANN201 - sklearn-like alias
        """Alias for `decision_function` (sklearn naming)."""

        return self.decision_function(x)

    def predict_proba(  # noqa: ANN001, ANN201 - 2-class proba API
        self,
        x,
        method: str = "linear",
        return_confidence: bool = False,
    ):
        """Predict outlier probability as a 2-class array ``[p(inlier), p(outlier)]``.

        This keeps compatibility with common 2-class `predict_proba` tooling.
        """

        if not hasattr(self, "decision_scores_"):
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        train_scores = np.asarray(self.decision_scores_, dtype=np.float64).reshape(-1)
        test_scores = np.asarray(self.decision_function(x), dtype=np.float64).reshape(-1)

        n_classes = int(getattr(self, "_classes", 2) or 2)
        if n_classes < 2:
            n_classes = 2

        probs = np.zeros((len(test_scores), n_classes), dtype=np.float64)

        if method == "linear":
            outlier = self._linear_outlier_probability(test_scores, train_scores=train_scores)
            probs[:, 1] = outlier
            probs[:, 0] = 1.0 - outlier
        elif method == "unify":
            # Unification via erf, then clamp to [0,1].
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
        else:
            raise ValueError(f"method {method!r} is not a valid probability conversion method")

        if not bool(return_confidence):
            return probs

        if hasattr(self, "threshold_"):
            labels = (test_scores > float(self.threshold_)).astype(np.int64)
        else:
            labels = np.argmax(probs, axis=1).astype(np.int64)
        conf = self._label_confidence_from_scores(test_scores, labels=labels)
        return probs, conf
