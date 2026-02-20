# -*- coding: utf-8 -*-
"""
KNN (K-Nearest Neighbors) outlier detector wrapper.

KNN is a simple and effective outlier detection method based on the distance
to k-nearest neighbors. It's easy to understand and implement.

Reference:
    Ramaswamy, S., Rastogi, R. and Shim, K., 2000.
    Efficient algorithms for mining outliers from large data sets.
    ACM SIGMOD Record.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .baseml import BaseVisionDetector
from .registry import register_model

try:  # pragma: no cover - optional dependency guard
    from pyod.models.knn import KNN as _PyODKNN
except ImportError as exc:  # pragma: no cover - surface install guidance
    _PyODKNN = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CoreKNN(_PyODKNN if _PyODKNN is not None else object):
    """Shallow wrapper bridging PyOD KNN to registry."""

    def __init__(self, *args, **kwargs):
        if _PyODKNN is None:
            raise ImportError(
                "pyod.models.knn is unavailable. Install pyod to use KNN."
            ) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)  # type: ignore[misc]


@register_model(
    "vision_knn",
    tags=("vision", "classical", "neighbors", "knn"),
    metadata={
        "description": "Vision wrapper for KNN outlier detector",
        "paper": "SIGMOD 2000",
        "year": 2000,
        "simple": True,
        "interpretable": True,
    },
)
class VisionKNN(BaseVisionDetector):
    """
    Vision-compatible KNN detector for anomaly detection.

    KNN detects outliers based on the distance to their k-nearest neighbors.
    Points that are far from their neighbors are considered outliers.

    Parameters
    ----------
    feature_extractor : object
        Feature extractor with an 'extract' method that converts images to features.
    contamination : float, optional (default=0.1)
        The amount of contamination of the data set.
    n_neighbors : int, optional (default=5)
        Number of neighbors to use by default.
    method : str, optional (default='largest')
        {'largest', 'mean', 'median'}
        - 'largest': use the distance to the kth neighbor as the outlier score
        - 'mean': use the average of all k neighbors as the outlier score
        - 'median': use the median of the distance to k neighbors as the outlier score
    radius : float, optional (default=1.0)
        Range of parameter space to use by default for radius_neighbors queries.
    algorithm : str, optional (default='auto')
        {'auto', 'ball_tree', 'kd_tree', 'brute'}
        Algorithm used to compute the nearest neighbors.
    leaf_size : int, optional (default=30)
        Leaf size passed to BallTree or KDTree.
    metric : str or callable, optional (default='minkowski')
        Metric used for distance computation.
    p : int, optional (default=2)
        Parameter for the Minkowski metric.
    n_jobs : int, optional (default=1)
        The number of parallel jobs to run for neighbors search.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.

    Examples
    --------
    >>> from pyimgano import models, utils
    >>> feature_extractor = utils.ImagePreprocessor(resize=(224, 224))
    >>> detector = models.create_model(
    ...     "vision_knn",
    ...     feature_extractor=feature_extractor,
    ...     n_neighbors=10,
    ...     method='largest'
    ... )
    >>> detector.fit(train_image_paths)
    >>> predictions = detector.predict(test_image_paths)
    """

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 5,
        method: str = 'largest',
        **kwargs
    ):
        self.detector_kwargs = dict(
            contamination=contamination,
            n_neighbors=n_neighbors,
            method=method,
            **kwargs
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreKNN(**self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        """
        Fit detector using training images.

        Parameters
        ----------
        X : iterable of str
            Training image paths.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        features = np.asarray(self.feature_extractor.extract(X))
        self.detector.fit(features)
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """
        Predict raw anomaly score of X using the fitted detector.

        Parameters
        ----------
        X : iterable of str
            Test image paths.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        features = np.asarray(self.feature_extractor.extract(X))
        return self.detector.decision_function(features)
