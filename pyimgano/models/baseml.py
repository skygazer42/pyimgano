# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod

from pyimgano.utils.optional_deps import optional_import


_pyod_base, _pyod_error = optional_import("pyod.models.base")
if _pyod_base is not None:
    BaseDetector = _pyod_base.BaseDetector  # type: ignore[attr-defined]
else:
    class BaseDetector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Optional dependency 'pyod' is required for classical detectors.\n"
                "Install it via:\n  pip install 'pyod'\n"
                f"Original error: {_pyod_error}"
            ) from _pyod_error


class BaseVisionDetector(BaseDetector):
    """
    所有包装了经典机器学习算法的视觉异常检测器的抽象基类。
    """

    @abstractmethod
    def __init__(self, contamination=0.1, feature_extractor=None):
        super(BaseVisionDetector, self).__init__(contamination=contamination)

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
        if not hasattr(feature_extractor, 'extract'):
            raise TypeError("feature_extractor 必须有一个名为 'extract' 的方法。")
        self.feature_extractor = feature_extractor

        self.detector = self._build_detector()

    @abstractmethod
    def _build_detector(self):
        """
    定义并返回一个具体的、经典的异常检测器实例。
        例如: from pyod.models.lof import LOF; return LOF()
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
        features = self.feature_extractor.extract(X)
        # 2. 使用特征向量来训练内部的经典检测器
        self.detector.fit(features)
        # 3. 将训练分数同步到 self.decision_scores_，以便父类处理
        self.decision_scores_ = self.detector.decision_scores_
        # 4. 调用 PyOD 基类的方法，自动计算阈值和标签
        self._process_decision_scores()

        return self
    def decision_function(self, X):

        # 1. 从新图像中提取特征
        features = self.feature_extractor.extract(X)
        # 2. 用训练好的经典检测器计算异常分数
        return self.detector.decision_function(features)
