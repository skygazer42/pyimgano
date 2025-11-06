# -*- coding: utf-8 -*-
"""
MCD (Minimum Covariance Determinant) wrapper.

MCD is a robust estimator of covariance that can identify outliers
based on Mahalanobis distance from a robust estimate.

Reference:
    Rousseeuw, P.J. and Driessen, K.V., 1999.
    A fast algorithm for the minimum covariance determinant estimator.
    Technometrics.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    from pyod.models.mcd import MCD as _PyODMCD

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODMCD = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_mcd",
    tags=("vision", "classical", "statistical", "mcd", "robust"),
    metadata={
        "description": "MCD - Robust covariance-based outlier detector",
        "paper": "Rousseeuw & Driessen, Technometrics 1999",
        "year": 1999,
        "robust": True,
        "parametric": True,
    },
)
class VisionMCD(BaseVisionDetector):
    """
    Vision-compatible MCD detector for anomaly detection.

    MCD uses robust covariance estimation to detect outliers based on
    Mahalanobis distance. It's resistant to outlier contamination.

    Parameters
    ----------
    feature_extractor : object
        Feature extractor with an 'extract' method.
    contamination : float, default=0.1
        Expected proportion of outliers (0 < contamination < 0.5).
    support_fraction : float, optional
        Proportion of points to include in the support.
        If None, minimum value is used: (n_samples + n_features + 1) / 2.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        Outlier scores of training data.
    threshold_ : float
        Threshold for binary classification.
    labels_ : ndarray of shape (n_samples,)
        Binary labels (0: normal, 1: outlier).

    Examples
    --------
    >>> from pyimgano import models, utils
    >>> extractor = utils.ImagePreprocessor(resize=(224, 224))
    >>> detector = models.create_model(
    ...     "vision_mcd",
    ...     feature_extractor=extractor,
    ...     contamination=0.1,
    ...     random_state=42
    ... )
    >>> detector.fit(train_paths)
    >>> predictions = detector.predict(test_paths)

    Notes
    -----
    - MCD is robust to outliers in the training data
    - Assumes roughly Gaussian distribution
    - Works well for low to medium dimensional data
    - Computational complexity: O(n²p) where p is dimension

    References
    ----------
    .. [1] Rousseeuw, P.J. and Driessen, K.V., 1999.
           A fast algorithm for the minimum covariance determinant estimator.
           Technometrics, 41(3), pp.212-223.
    """

    def __init__(
        self,
        *,
        feature_extractor,
        contamination: float = 0.1,
        support_fraction: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if not _PYOD_AVAILABLE:
            raise ImportError(
                "PyOD is not available. Install it with:\n"
                "  pip install 'pyod>=1.1.0'\n"
                f"Original error: {_IMPORT_ERROR}"
            )

        if not 0 < contamination < 0.5:
            raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

        if support_fraction is not None and not 0 < support_fraction <= 1:
            raise ValueError(
                f"support_fraction must be in (0, 1], got {support_fraction}"
            )

        self.support_fraction = support_fraction
        self.random_state = random_state
        self._detector_kwargs = {
            "contamination": contamination,
            "support_fraction": support_fraction,
            "random_state": random_state,
        }

        logger.debug(
            "Initializing VisionMCD with contamination=%.2f, support_fraction=%s",
            contamination,
            support_fraction,
        )

        super().__init__(
            contamination=contamination,
            feature_extractor=feature_extractor,
        )

    def _build_detector(self):
        """Build the underlying PyOD MCD detector."""
        return _PyODMCD(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None):
        """
        Fit the MCD detector on training images.

        Parameters
        ----------
        X : iterable of str
            Training image paths.
        y : ndarray, optional
            Not used, present for API consistency.

        Returns
        -------
        self : VisionMCD
            Fitted estimator.

        Raises
        ------
        ValueError
            If features are invalid or insufficient samples.
        """
        logger.info("Fitting MCD detector on images")

        try:
            features = self.feature_extractor.extract(X)
            features = np.asarray(features, dtype=np.float64)

            if features.ndim != 2:
                raise ValueError(f"Expected 2D features, got shape {features.shape}")

            if np.isnan(features).any() or np.isinf(features).any():
                raise ValueError("Features contain NaN or Inf values")

            n_samples, n_features = features.shape
            if n_samples < n_features + 2:
                logger.warning(
                    "MCD may be unstable with %d samples and %d features. "
                    "Recommend at least %d samples.",
                    n_samples,
                    n_features,
                    n_features + 2,
                )

            logger.debug("Feature shape: %s", features.shape)

            self.detector.fit(features)
            self.decision_scores_ = self.detector.decision_scores_
            self._process_decision_scores()

            logger.info("MCD fit complete. Threshold: %.4f", self.threshold_)

        except Exception as e:
            logger.error("MCD fitting failed: %s", e)
            raise

        return self

    def decision_function(self, X: Union[Iterable[str], NDArray]) -> NDArray[np.float64]:
        """
        Compute anomaly scores for test images.

        Parameters
        ----------
        X : iterable of str or ndarray
            Test image paths or pre-extracted features.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (higher = more anomalous).

        Raises
        ------
        RuntimeError
            If detector is not fitted.
        """
        if not hasattr(self.detector, "decision_scores_"):
            raise RuntimeError("Detector must be fitted before prediction")

        try:
            features = self.feature_extractor.extract(X)
            features = np.asarray(features, dtype=np.float64)

            if features.ndim != 2:
                raise ValueError(f"Expected 2D features, got shape {features.shape}")

            scores = self.detector.decision_function(features)
            logger.debug("Computed MCD scores for %d samples", len(scores))

            return scores

        except Exception as e:
            logger.error("MCD prediction failed: %s", e)
            raise
