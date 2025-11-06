# -*- coding: utf-8 -*-
"""
COF (Connectivity-Based Outlier Factor) wrapper.

COF considers the connectivity between data points to identify outliers.
It's an improvement over LOF for datasets with varying densities.

Reference:
    Tang, J., Chen, Z., Fu, A.W.C. and Cheung, D.W., 2002.
    Enhancing effectiveness of outlier detections for low density patterns.
    Pacific-Asia Conference on Knowledge Discovery and Data Mining.
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
    from pyod.models.cof import COF as _PyODCOF

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODCOF = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_cof",
    tags=("vision", "classical", "neighbors", "cof"),
    metadata={
        "description": "COF - Connectivity-based outlier detector",
        "paper": "Tang et al., PAKDD 2002",
        "year": 2002,
        "density_based": True,
    },
)
class VisionCOF(BaseVisionDetector):
    """
    Vision-compatible COF detector for anomaly detection.

    COF uses connectivity patterns between data points to detect outliers.
    It's particularly effective for datasets with varying local densities.

    Parameters
    ----------
    feature_extractor : object
        Feature extractor with an 'extract' method.
    contamination : float, default=0.1
        Expected proportion of outliers (0 < contamination < 0.5).
    n_neighbors : int, default=20
        Number of neighbors to use.

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
    ...     "vision_cof",
    ...     feature_extractor=extractor,
    ...     n_neighbors=20
    ... )
    >>> detector.fit(train_paths)
    >>> predictions = detector.predict(test_paths)

    Notes
    -----
    - COF is effective for varying density datasets
    - More robust than LOF in some cases
    - Computational complexity: O(n²)

    References
    ----------
    .. [1] Tang, J., Chen, Z., Fu, A.W.C. and Cheung, D.W., 2002.
           Enhancing effectiveness of outlier detections for low density patterns.
           Pacific-Asia Conference on Knowledge Discovery and Data Mining.
    """

    def __init__(
        self,
        *,
        feature_extractor,
        contamination: float = 0.1,
        n_neighbors: int = 20,
    ) -> None:
        if not _PYOD_AVAILABLE:
            raise ImportError(
                "PyOD is not available. Install it with:\n"
                "  pip install 'pyod>=1.1.0'\n"
                f"Original error: {_IMPORT_ERROR}"
            )

        if not 0 < contamination < 0.5:
            raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")

        self.n_neighbors = n_neighbors
        self._detector_kwargs = {
            "contamination": contamination,
            "n_neighbors": n_neighbors,
        }

        logger.debug(
            "Initializing VisionCOF with contamination=%.2f, n_neighbors=%d",
            contamination,
            n_neighbors,
        )

        super().__init__(
            contamination=contamination,
            feature_extractor=feature_extractor,
        )

    def _build_detector(self):
        """Build the underlying PyOD COF detector."""
        return _PyODCOF(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None):
        """
        Fit the COF detector on training images.

        Parameters
        ----------
        X : iterable of str
            Training image paths.
        y : ndarray, optional
            Not used, present for API consistency.

        Returns
        -------
        self : VisionCOF
            Fitted estimator.
        """
        logger.info("Fitting COF detector on images")

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

            logger.info("COF fit complete. Threshold: %.4f", self.threshold_)

        except Exception as e:
            logger.error("COF fitting failed: %s", e)
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
            logger.debug("Computed COF scores for %d samples", len(scores))

            return scores

        except Exception as e:
            logger.error("COF prediction failed: %s", e)
            raise
