# -*- coding: utf-8 -*-
"""
ECOD (Empirical Cumulative Distribution-based Outlier Detection) wrapper.

ECOD is a parameter-free, highly interpretable outlier detection algorithm
based on empirical CDF functions. It's one of the top-performing methods
on standard benchmarks.

Reference:
    Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X., 2022.
    ECOD: Unsupervised Outlier Detection Using Empirical Cumulative
    Distribution Functions. IEEE Transactions on Knowledge and Data
    Engineering (TKDE).

Requirements:
    - PyOD >= 0.9.7 (ECOD added in this version)
    - PyOD >= 1.0.9 recommended (scipy compatibility fix)
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
    from pyod.models.ecod import ECOD as _PyODECOD

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODECOD = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_ecod",
    tags=("vision", "classical", "ecod", "parameter-free", "high-performance"),
    metadata={
        "description": "ECOD - Empirical CDF-based outlier detector (TKDE 2022)",
        "paper": "Li et al., TKDE 2022",
        "year": 2022,
        "interpretable": True,
        "parameter_free": True,
        "fast": True,
        "benchmark_rank": "top-tier",
    },
)
class VisionECOD(BaseVisionDetector):
    """
    Vision-compatible ECOD detector for anomaly detection.

    ECOD uses empirical cumulative distribution functions to detect outliers
    without requiring parameter tuning. It's highly interpretable and consistently
    ranks among the top performers in benchmarks.

    Parameters
    ----------
    feature_extractor : object
        Feature extractor with an 'extract' method.
    contamination : float, default=0.1
        Expected proportion of outliers in the dataset (0 < contamination < 0.5).
    n_jobs : int, default=1
        Number of parallel jobs. -1 uses all processors.

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
    ...     "vision_ecod",
    ...     feature_extractor=extractor,
    ...     contamination=0.1
    ... )
    >>> detector.fit(train_paths)
    >>> predictions = detector.predict(test_paths)

    Notes
    -----
    - ECOD is parameter-free and does not require hyperparameter tuning
    - Works well on high-dimensional data
    - Computationally efficient (O(n*d) complexity)
    - Provides interpretable outlier scores

    References
    ----------
    .. [1] Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X., 2022.
           ECOD: Unsupervised Outlier Detection Using Empirical Cumulative
           Distribution Functions. IEEE Transactions on Knowledge and Data
           Engineering.
    """

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_jobs: int = 1,
    ) -> None:
        if not _PYOD_AVAILABLE:
            raise ImportError(
                "PyOD is not available. Install it with:\n"
                "  pip install 'pyod>=1.1.0'\n"
                f"Original error: {_IMPORT_ERROR}"
            )

        # Validate contamination
        if not 0 < contamination < 0.5:
            raise ValueError(
                f"contamination must be in (0, 0.5), got {contamination}"
            )

        self.n_jobs = n_jobs
        self._detector_kwargs = {"contamination": contamination, "n_jobs": n_jobs}

        logger.debug(
            "Initializing VisionECOD with contamination=%.2f, n_jobs=%d",
            contamination,
            n_jobs,
        )

        super().__init__(
            contamination=contamination,
            feature_extractor=feature_extractor,
        )

    def _build_detector(self):
        """Build the underlying PyOD ECOD detector."""
        return _PyODECOD(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None):
        """
        Fit the ECOD detector on training images.

        Parameters
        ----------
        X : iterable of str
            Training image paths.
        y : ndarray, optional
            Not used, present for API consistency.

        Returns
        -------
        self : VisionECOD
            Fitted estimator.

        Raises
        ------
        ValueError
            If feature extraction fails or produces invalid features.
        """
        logger.info("Fitting ECOD detector on %d images", len(list(X)))

        try:
            features = self.feature_extractor.extract(X)
            features = np.asarray(features, dtype=np.float64)

            if features.ndim != 2:
                raise ValueError(
                    f"Expected 2D features, got shape {features.shape}"
                )

            if np.isnan(features).any() or np.isinf(features).any():
                raise ValueError("Features contain NaN or Inf values")

            logger.debug("Feature shape: %s", features.shape)

            self.detector.fit(features)
            self.decision_scores_ = self.detector.decision_scores_
            self._process_decision_scores()

            logger.info(
                "ECOD fit complete. Threshold: %.4f, "
                "Score range: [%.4f, %.4f]",
                self.threshold_,
                self.decision_scores_.min(),
                self.decision_scores_.max(),
            )

        except Exception as e:
            logger.error("ECOD fitting failed: %s", e)
            raise

        return self

    def decision_function(
        self, X: Union[Iterable[str], NDArray]
    ) -> NDArray[np.float64]:
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
        ValueError
            If features are invalid.
        RuntimeError
            If detector is not fitted.
        """
        if not hasattr(self.detector, "decision_scores_"):
            raise RuntimeError("Detector must be fitted before prediction")

        try:
            features = self.feature_extractor.extract(X)
            features = np.asarray(features, dtype=np.float64)

            if features.ndim != 2:
                raise ValueError(
                    f"Expected 2D features, got shape {features.shape}"
                )

            scores = self.detector.decision_function(features)
            logger.debug(
                "Computed scores for %d samples. Range: [%.4f, %.4f]",
                len(scores),
                scores.min(),
                scores.max(),
            )

            return scores

        except Exception as e:
            logger.error("ECOD prediction failed: %s", e)
            raise
