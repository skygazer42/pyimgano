# -*- coding: utf-8 -*-
"""
Feature Bagging ensemble wrapper.

Feature Bagging improves stability and accuracy by training multiple
base detectors on different random subsets of features.

Reference:
    Lazarevic, A. and Kumar, V., 2005.
    Feature bagging for outlier detection.
    ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
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
    from pyod.models.feature_bagging import FeatureBagging as _PyODFeatureBagging

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODFeatureBagging = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_feature_bagging",
    tags=("vision", "ensemble", "feature_bagging", "high-performance"),
    metadata={
        "description": "Feature Bagging - Ensemble outlier detector",
        "paper": "Lazarevic & Kumar, KDD 2005",
        "year": 2005,
        "ensemble": True,
        "robust": True,
    },
)
class VisionFeatureBagging(BaseVisionDetector):
    """
    Vision-compatible Feature Bagging detector for anomaly detection.

    Feature Bagging builds multiple base detectors on random feature subsets
    and combines their predictions. This improves both stability and accuracy,
    especially on high-dimensional data.

    Parameters
    ----------
    feature_extractor : object
        Feature extractor with an 'extract' method.
    contamination : float, default=0.1
        Expected proportion of outliers (0 < contamination < 0.5).
    n_estimators : int, default=10
        Number of base estimators in the ensemble.
    max_features : float or int, default=1.0
        Number of features to draw for each base estimator.
        - If float: proportion of features (0 < max_features <= 1.0)
        - If int: absolute number of features
    bootstrap_features : bool, default=False
        Whether to sample features with replacement.
    n_jobs : int, default=1
        Number of parallel jobs. -1 uses all processors.
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
    ...     "vision_feature_bagging",
    ...     feature_extractor=extractor,
    ...     n_estimators=50,
    ...     max_features=0.7,
    ...     n_jobs=-1
    ... )
    >>> detector.fit(train_paths)
    >>> predictions = detector.predict(test_paths)

    Notes
    -----
    - Feature Bagging reduces variance and improves stability
    - Effective for high-dimensional data
    - Can use any base detector (default: LOF)
    - Computational complexity: O(k × base_complexity) where k = n_estimators

    References
    ----------
    .. [1] Lazarevic, A. and Kumar, V., 2005.
           Feature bagging for outlier detection.
           ACM SIGKDD International Conference on Knowledge Discovery
           and Data Mining.
    """

    def __init__(
        self,
        *,
        feature_extractor,
        contamination: float = 0.1,
        n_estimators: int = 10,
        max_features: Union[int, float] = 1.0,
        bootstrap_features: bool = False,
        n_jobs: int = 1,
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

        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")

        if isinstance(max_features, float):
            if not 0 < max_features <= 1.0:
                raise ValueError(
                    f"max_features as float must be in (0, 1], got {max_features}"
                )
        elif isinstance(max_features, int):
            if max_features < 1:
                raise ValueError(f"max_features as int must be >= 1, got {max_features}")

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._detector_kwargs = {
            "contamination": contamination,
            "n_estimators": n_estimators,
            "max_features": max_features,
            "bootstrap_features": bootstrap_features,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }

        logger.debug(
            "Initializing VisionFeatureBagging with n_estimators=%d, "
            "max_features=%s, n_jobs=%d",
            n_estimators,
            max_features,
            n_jobs,
        )

        super().__init__(
            contamination=contamination,
            feature_extractor=feature_extractor,
        )

    def _build_detector(self):
        """Build the underlying PyOD Feature Bagging detector."""
        return _PyODFeatureBagging(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None):
        """
        Fit the Feature Bagging detector on training images.

        Parameters
        ----------
        X : iterable of str
            Training image paths.
        y : ndarray, optional
            Not used, present for API consistency.

        Returns
        -------
        self : VisionFeatureBagging
            Fitted estimator.
        """
        logger.info(
            "Fitting Feature Bagging detector with %d estimators on images",
            self.n_estimators,
        )

        try:
            features = self.feature_extractor.extract(X)
            features = np.asarray(features, dtype=np.float64)

            if features.ndim != 2:
                raise ValueError(f"Expected 2D features, got shape {features.shape}")

            if np.isnan(features).any() or np.isinf(features).any():
                raise ValueError("Features contain NaN or Inf values")

            logger.debug("Feature shape: %s", features.shape)

            self.detector.fit(features)
            self.decision_scores_ = self.detector.decision_scores_
            self._process_decision_scores()

            logger.info("Feature Bagging fit complete. Threshold: %.4f", self.threshold_)

        except Exception as e:
            logger.error("Feature Bagging fitting failed: %s", e)
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
        """
        if not hasattr(self.detector, "decision_scores_"):
            raise RuntimeError("Detector must be fitted before prediction")

        try:
            features = self.feature_extractor.extract(X)
            features = np.asarray(features, dtype=np.float64)

            if features.ndim != 2:
                raise ValueError(f"Expected 2D features, got shape {features.shape}")

            scores = self.detector.decision_function(features)
            logger.debug("Computed Feature Bagging scores for %d samples", len(scores))

            return scores

        except Exception as e:
            logger.error("Feature Bagging prediction failed: %s", e)
            raise
