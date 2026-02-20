# -*- coding: utf-8 -*-
"""
COPOD (Copula-Based Outlier Detection) wrapper.

COPOD is a parameter-free, highly efficient outlier detection algorithm
that uses copula-based methods. It's consistently ranked among the top
performers in benchmark comparisons.

Reference:
    Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, H.G., 2020.
    COPOD: Copula-Based Outlier Detection.
    IEEE International Conference on Data Mining (ICDM).

Requirements:
    - PyOD >= 0.9.0 (COPOD added in this version)
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
    from pyod.models.copod import COPOD as _PyODCOPOD

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODCOPOD = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_copod",
    tags=("vision", "classical", "copod", "parameter-free", "high-performance"),
    metadata={
        "description": "COPOD - Copula-based outlier detector (ICDM 2020)",
        "paper": "Li et al., ICDM 2020",
        "year": 2020,
        "fast": True,
        "parameter_free": True,
        "benchmark_rank": "top-tier",
    },
)
class VisionCOPOD(BaseVisionDetector):
    """
    Vision-compatible COPOD detector for anomaly detection.

    COPOD uses copula functions to model joint distributions and detect outliers.
    It requires no parameter tuning and is very fast, making it ideal for
    production environments.

    Parameters
    ----------
    feature_extractor : object
        Feature extractor with an 'extract' method.
    contamination : float, default=0.1
        Expected proportion of outliers in dataset (0 < contamination < 0.5).
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
    ...     "vision_copod",
    ...     feature_extractor=extractor
    ... )
    >>> detector.fit(train_paths)
    >>> predictions = detector.predict(test_paths)

    Notes
    -----
    - COPOD is parameter-free and very fast
    - Works well on high-dimensional data
    - Consistently ranks in top tier of benchmark comparisons
    - Computationally efficient (O(n*d) complexity)

    References
    ----------
    .. [1] Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, H.G., 2020.
           COPOD: Copula-Based Outlier Detection. IEEE International Conference
           on Data Mining (ICDM).
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

        if not 0 < contamination < 0.5:
            raise ValueError(
                f"contamination must be in (0, 0.5), got {contamination}"
            )

        self.n_jobs = n_jobs
        self._detector_kwargs = {"contamination": contamination, "n_jobs": n_jobs}

        logger.debug(
            "Initializing VisionCOPOD with contamination=%.2f, n_jobs=%d",
            contamination,
            n_jobs,
        )

        super().__init__(
            contamination=contamination,
            feature_extractor=feature_extractor,
        )

    def _build_detector(self):
        """Build the underlying PyOD COPOD detector."""
        return _PyODCOPOD(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None):
        """
        Fit the COPOD detector on training images.

        Parameters
        ----------
        X : iterable of str
            Training image paths.
        y : ndarray, optional
            Not used, present for API consistency.

        Returns
        -------
        self : VisionCOPOD
            Fitted estimator.

        Raises
        ------
        ValueError
            If feature extraction produces invalid features.
        """
        logger.info("Fitting COPOD detector on images")

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
                "COPOD fit complete. Threshold: %.4f",
                self.threshold_,
            )

        except Exception as e:
            logger.error("COPOD fitting failed: %s", e)
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
        RuntimeError
            If detector is not fitted.
        ValueError
            If features are invalid.
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
                "Computed COPOD scores for %d samples",
                len(scores),
            )

            return scores

        except Exception as e:
            logger.error("COPOD prediction failed: %s", e)
            raise
