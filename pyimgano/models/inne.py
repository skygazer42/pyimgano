# -*- coding: utf-8 -*-
"""
INNE (Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles) wrapper.

INNE improves upon Isolation Forest by using nearest-neighbor information.
It's faster and more accurate than traditional Isolation Forest.

Reference:
    Bandaragoda, T.R., Ting, K.M., Albrecht, D., Liu, F.T. and Wells, J.R., 2014.
    Efficient anomaly detection by isolation using nearest neighbour ensemble.
    IEEE International Conference on Data Mining Workshop.
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
    from pyod.models.inne import INNE as _PyODINNE

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODINNE = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_inne",
    tags=("vision", "classical", "isolation", "inne", "fast"),
    metadata={
        "description": "INNE - Isolation using Nearest-Neighbor Ensembles",
        "paper": "Bandaragoda et al., ICDM 2014",
        "year": 2014,
        "fast": True,
        "scalable": True,
    },
)
class VisionINNE(BaseVisionDetector):
    """
    Vision-compatible INNE detector for anomaly detection.

    INNE combines isolation principles with nearest-neighbor information
    for fast and accurate outlier detection. It often outperforms
    traditional Isolation Forest.

    Parameters
    ----------
    feature_extractor : object
        Feature extractor with an 'extract' method.
    contamination : float, default=0.1
        Expected proportion of outliers (0 < contamination < 0.5).
    n_estimators : int, default=200
        Number of base estimators in the ensemble.
    max_samples : int or float, default='auto'
        Number of samples to draw for each base estimator.
        - If int: absolute number
        - If float: proportion of dataset
        - If 'auto': min(256, n_samples)
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
    ...     "vision_inne",
    ...     feature_extractor=extractor,
    ...     n_estimators=200,
    ...     random_state=42
    ... )
    >>> detector.fit(train_paths)
    >>> predictions = detector.predict(test_paths)

    Notes
    -----
    - INNE is faster than Isolation Forest
    - More accurate on many datasets
    - Scalable to large datasets
    - Computational complexity: O(n log n)

    References
    ----------
    .. [1] Bandaragoda, T.R., Ting, K.M., Albrecht, D., Liu, F.T. and Wells, J.R., 2014.
           Efficient anomaly detection by isolation using nearest neighbour ensemble.
           IEEE International Conference on Data Mining Workshop.
    """

    def __init__(
        self,
        *,
        feature_extractor,
        contamination: float = 0.1,
        n_estimators: int = 200,
        max_samples: Union[int, float, str] = "auto",
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

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self._detector_kwargs = {
            "contamination": contamination,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "random_state": random_state,
        }

        logger.debug(
            "Initializing VisionINNE with n_estimators=%d, max_samples=%s",
            n_estimators,
            max_samples,
        )

        super().__init__(
            contamination=contamination,
            feature_extractor=feature_extractor,
        )

    def _build_detector(self):
        """Build the underlying PyOD INNE detector."""
        return _PyODINNE(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None):
        """
        Fit the INNE detector on training images.

        Parameters
        ----------
        X : iterable of str
            Training image paths.
        y : ndarray, optional
            Not used, present for API consistency.

        Returns
        -------
        self : VisionINNE
            Fitted estimator.
        """
        logger.info("Fitting INNE detector with %d estimators on images", self.n_estimators)

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

            logger.info("INNE fit complete. Threshold: %.4f", self.threshold_)

        except Exception as e:
            logger.error("INNE fitting failed: %s", e)
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
            logger.debug("Computed INNE scores for %d samples", len(scores))

            return scores

        except Exception as e:
            logger.error("INNE prediction failed: %s", e)
            raise
