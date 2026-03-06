# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from pyimgano.features.protocols import FeatureExtractor, FittableFeatureExtractor
from pyimgano.features.registry import resolve_feature_extractor

from .base_detector import BaseDetector


class BaseVisionDetector(BaseDetector):
    """
    所有包装了经典机器学习算法的视觉异常检测器的抽象基类。
    """

    @abstractmethod
    def __init__(self, contamination=0.1, feature_extractor=None):
        super(BaseVisionDetector, self).__init__(contamination=contamination)
        # Compatibility: some utilities (e.g. `predict_proba`) expect `_classes` to exist.
        # In unsupervised detection this is always binary.
        self._set_n_classes(None)
        self._feature_cache = None

        if feature_extractor is None:
            # Provide a safe default so classical detectors work out-of-the-box.
            # This trades off accuracy for simplicity, but avoids surprising
            # TypeError/ValueError for new users and for quick benchmarks.
            from pyimgano.utils.image_ops import ImagePreprocessor

            feature_extractor = ImagePreprocessor(
                resize=(224, 224),
                output_tensor=False,
                error_mode="zeros",
            )

        # Allow JSON-friendly specs: {"name": "...", "kwargs": {...}}.
        feature_extractor = resolve_feature_extractor(feature_extractor)
        if not isinstance(feature_extractor, FeatureExtractor):
            raise TypeError("feature_extractor 必须实现 .extract(inputs) -> np.ndarray")
        self._base_feature_extractor = feature_extractor
        self.feature_extractor = feature_extractor
        self._feature_extractor_fitted = False

        self.detector = self._build_detector()

    def set_feature_cache(self, cache_dir: str | Path | None) -> None:
        """Enable/disable disk feature caching.

        Notes
        -----
        When enabled, caching is applied in a best-effort way:
        - If inputs are all paths (str/Path), cache extracted feature rows by file metadata.
        - If inputs are all numpy arrays, cache extracted feature rows by hashing array content.
        - Mixed inputs fall back to no caching.
        """

        if cache_dir is None:
            self._feature_cache = None
            self._array_feature_cache = None
            self.feature_extractor = self._base_feature_extractor
            return

        from pyimgano.cache.array_features import ArrayFeatureCache, CachedArrayFeatureExtractor
        from pyimgano.cache.features import (
            CachedFeatureExtractor,
            FeatureCache,
            fingerprint_feature_extractor,
        )

        cache_root = Path(cache_dir)
        fp = fingerprint_feature_extractor(self._base_feature_extractor)

        self._feature_cache = FeatureCache(
            cache_dir=cache_root / "paths",
            extractor_fingerprint=fp,
        )
        self._array_feature_cache = ArrayFeatureCache(
            cache_dir=cache_root / "arrays",
            extractor_fingerprint=fp,
        )

        extractor = CachedFeatureExtractor(
            base_extractor=self._base_feature_extractor,
            cache=self._feature_cache,
        )
        self.feature_extractor = CachedArrayFeatureExtractor(
            base_extractor=extractor,
            cache=self._array_feature_cache,
        )

    @abstractmethod
    def _build_detector(self):
        """
        定义并返回一个具体的、经典的异常检测器实例。
            例如: return sklearn.neighbors.LocalOutlierFactor(...)
        """
        pass

    def fit(self, X, y=None):
        """
        使用正常的、无缺陷的图像数据来拟合检测器。

        Parameters
        ----------
        X : list of str, or numpy array
            输入的训练样本，通常是图像文件路径的列表。
        """
        # 1. 使用插件化的特征提取器，将图像转换为特征向量
        extractor = self.feature_extractor
        if extractor is None or not isinstance(extractor, FeatureExtractor):
            raise TypeError("feature_extractor 必须实现 .extract(inputs) -> np.ndarray")

        # Optional: allow extractors to learn normalization/projection from the training set.
        if (not self._feature_extractor_fitted) and isinstance(extractor, FittableFeatureExtractor):
            extractor.fit(X, y=y)
            self._feature_extractor_fitted = True

        features = self.feature_extractor.extract(X)
        # 2. 使用特征向量来训练内部的经典检测器
        self.detector.fit(features)
        # 3. 将训练分数同步到 self.decision_scores_，以便父类处理
        self.decision_scores_ = self.detector.decision_scores_
        # 4. 调用基类的方法，自动计算阈值和标签
        self._process_decision_scores()
        # Compatibility: enable `predict_proba()` by initializing `_classes`.
        self._set_n_classes(y)

        return self

    def decision_function(self, X):
        # 1. 从新图像中提取特征
        features = self.feature_extractor.extract(X)
        # 2. 用训练好的经典检测器计算异常分数
        return self.detector.decision_function(features)
