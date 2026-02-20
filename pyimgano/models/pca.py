# -*- coding: utf-8 -*-
"""
PCA (Principal Component Analysis) outlier detector wrapper.

PCA-based outlier detection uses the reconstruction error from principal
components to identify anomalies. Classic and widely used method.

Reference:
    Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003.
    A novel anomaly detection scheme based on principal component classifier.
    ICDM.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .baseml import BaseVisionDetector
from .registry import register_model

try:  # pragma: no cover - optional dependency guard
    from pyod.models.pca import PCA as _PyODPCA
except ImportError as exc:  # pragma: no cover - surface install guidance
    _PyODPCA = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CorePCA(_PyODPCA if _PyODPCA is not None else object):
    """Shallow wrapper bridging PyOD PCA to registry."""

    def __init__(self, *args, **kwargs):
        if _PyODPCA is None:
            raise ImportError(
                "pyod.models.pca is unavailable. Install pyod to use PCA."
            ) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)  # type: ignore[misc]


@register_model(
    "vision_pca",
    tags=("vision", "classical", "linear", "pca"),
    metadata={
        "description": "Vision wrapper for PCA-based outlier detector",
        "paper": "ICDM 2003",
        "year": 2003,
        "classic": True,
        "interpretable": True,
    },
)
class VisionPCA(BaseVisionDetector):
    """
    Vision-compatible PCA detector for anomaly detection.

    PCA detects outliers by measuring the reconstruction error when projecting
    data onto principal components. High reconstruction errors indicate anomalies.

    Parameters
    ----------
    feature_extractor : object
        Feature extractor with an 'extract' method that converts images to features.
    contamination : float, optional (default=0.1)
        The amount of contamination of the data set.
    n_components : int, float, str or None, optional (default=None)
        Number of components to keep.
        - If int, keep n_components.
        - If float between 0 and 1, select the number of components such that
          the amount of variance explained is greater than n_components.
        - If 'mle', Mingen's MLE is used to guess the dimension.
        - If None, min(n_samples, n_features) - 1 components are kept.
    n_selected_components : int, optional (default=None)
        Number of selected principal components for calculating the outlier scores.
        If not set, use all principal components.
    whiten : bool, optional (default=False)
        When True, the components_ vectors are divided by n_samples times
        components_ to ensure uncorrelated outputs with unit component-wise variances.
    svd_solver : str, optional (default='auto')
        {'auto', 'full', 'arpack', 'randomized'}
    weighted : bool, optional (default=True)
        If True, the eigenvalues are used in score computation.
    standardization : bool, optional (default=True)
        If True, perform standardization first to convert data to zero mean and unit variance.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.

    Examples
    --------
    >>> from pyimgano import models, utils
    >>> feature_extractor = utils.ImagePreprocessor(resize=(224, 224))
    >>> detector = models.create_model(
    ...     "vision_pca",
    ...     feature_extractor=feature_extractor,
    ...     n_components=0.95,  # Keep 95% variance
    ...     contamination=0.1
    ... )
    >>> detector.fit(train_image_paths)
    >>> scores = detector.decision_function(test_image_paths)
    """

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_components=None,
        n_selected_components=None,
        whiten: bool = False,
        svd_solver: str = 'auto',
        weighted: bool = True,
        standardization: bool = True,
        **kwargs
    ):
        self.detector_kwargs = dict(
            contamination=contamination,
            n_components=n_components,
            n_selected_components=n_selected_components,
            whiten=whiten,
            svd_solver=svd_solver,
            weighted=weighted,
            standardization=standardization,
            **kwargs
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CorePCA(**self.detector_kwargs)

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
