# -*- coding: utf-8 -*-
"""
Vision CBLOF - 基于聚类的视觉异常检测器
遵循 BaseVisionDetector 架构，不依赖外部检测库的基础类
"""

import logging

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array
from tqdm import tqdm

from ..utils.fitted import require_fitted
from ..utils.param_check import check_parameter

# 只从本地基类导入
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

# ===================================================================
#                     日志
# ===================================================================

logger = logging.getLogger(__name__)

# ===================================================================
#                     特征提取器类
# ===================================================================


class ImageFeatureExtractor:
    """
    图像特征提取器 - 实现extract方法供BaseVisionDetector使用

    Parameters
    ----------
    method : str, optional (default='combined')
        特征提取方法：'color', 'texture', 'deep', 'combined'

    reduce_dim : bool, optional (default=True)
        是否进行PCA降维

    n_components : int, optional (default=50)
        PCA降维后的维度
    """

    def __init__(self, method="combined", reduce_dim=True, n_components=50):
        self.method = method
        self.reduce_dim = reduce_dim
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=0) if reduce_dim else None
        self.is_fitted = False

    def extract(self, x):
        """
        提取图像特征 - BaseVisionDetector要求的接口

        Parameters
        ----------
        X : list of str or numpy array
            图像文件路径列表或特征数组

        Returns
        -------
        features : numpy array
            提取的特征矩阵
        """
        # 如果已经是特征矩阵，直接处理
        if isinstance(x, np.ndarray):
            if self.is_fitted:
                x = self.scaler.transform(x)
                if self.reduce_dim and self.pca is not None:
                    x = self.pca.transform(x)
            return x

        # 从图像路径提取特征：物化一次，便于计数与复用迭代器
        paths = list(x)
        logger.info("CBLOF: extracting image features for %d inputs", len(paths))
        features = []

        for img_path in tqdm(paths):
            try:
                feat = self._extract_single_image_features(img_path)
                features.append(feat)
            except Exception as e:
                logger.warning("CBLOF: failed to extract features for %s: %s", img_path, e)
                # 添加零特征
                if len(features) > 0:
                    features.append(np.zeros_like(features[0]))
                else:
                    features.append(np.zeros(100))

        features = np.array(features)

        # 标准化和降维
        if not self.is_fitted:
            features = self.scaler.fit_transform(features)
            if self.reduce_dim and features.shape[1] > self.n_components:
                features = self.pca.fit_transform(features)
                logger.info(
                    "CBLOF: PCA reduced to %d dims (kept variance %.2f%%)",
                    int(features.shape[1]),
                    float(self.pca.explained_variance_ratio_.sum() * 100.0),
                )
            self.is_fitted = True
        else:
            features = self.scaler.transform(features)
            if self.reduce_dim and self.pca is not None:
                features = self.pca.transform(features)

        return features

    def _extract_single_image_features(self, image_path):
        """提取单张图像的特征"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        features = []

        if self.method in ["color", "combined"]:
            features.extend(self._extract_color_features(img))

        if self.method in ["texture", "combined"]:
            features.extend(self._extract_texture_features(img))

        if self.method == "deep":
            features.extend(self._extract_deep_features(img))

        return np.array(features)

    def _extract_color_features(self, img):
        """提取颜色特征"""
        features = []

        # RGB直方图
        for i in range(3):
            hist, _ = np.histogram(img[:, :, i], bins=16, range=(0, 256))
            hist = hist.astype(float) / (hist.sum() + 1e-6)
            features.extend(hist)

        # 颜色统计
        for i in range(3):
            channel = img[:, :, i]
            features.extend(
                [
                    channel.mean() / 255.0,
                    channel.std() / 255.0,
                    np.percentile(channel, 25) / 255.0,
                    np.percentile(channel, 75) / 255.0,
                ]
            )

        # HSV特征
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            features.extend([hsv[:, :, i].mean() / 255.0, hsv[:, :, i].std() / 255.0])

        return features

    def _extract_texture_features(self, img):
        """提取纹理特征"""
        features = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 梯度特征
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        features.extend(
            [grad_mag.mean() / 255.0, grad_mag.std() / 255.0, (grad_mag > 50).sum() / grad_mag.size]
        )

        # 简单纹理统计
        features.extend([gray.mean() / 255.0, gray.std() / 255.0])

        # 频域特征
        fft = np.fft.fft2(cv2.resize(gray, (64, 64)))
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        center = 32
        low_freq = magnitude[center - 8 : center + 8, center - 8 : center + 8].sum()
        total_freq = magnitude.sum()
        features.append(low_freq / (total_freq + 1e-6))

        return features

    def _extract_deep_features(self, img):
        """提取深度特征（简化版）"""
        img_small = cv2.resize(img, (16, 16))
        features = img_small.flatten() / 255.0
        return features[:100]


# ===================================================================
#                     核心CBLOF算法（独立实现）
# ===================================================================


@register_model(
    "core_cblof",
    tags=("classical", "core", "features", "clustering", "cblof"),
    metadata={
        "description": "Core CBLOF detector on feature matrices (native implementation)",
        "input": "features",
        "paper": "He et al., SDM 2003",
        "year": 2003,
    },
)
class CoreCBLOF(BaseDetector):
    """
    核心CBLOF算法 - 独立实现，不依赖PyOD

    Parameters
    ----------
    n_clusters : int
        聚类数量
    contamination : float
        污染率
    alpha : float
        大簇样本占比阈值
    beta : float
        簇大小比例阈值
    use_weights : bool
        是否使用簇大小作为权重
    random_state : int
        随机种子
    """

    def __init__(
        self,
        n_clusters=8,
        contamination=0.1,
        alpha=0.9,
        beta=5,
        use_weights=False,
        random_state=None,
    ):
        super().__init__(contamination=contamination)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.use_weights = use_weights
        self.random_state = random_state

        # 将在fit时设置
        self.clustering_estimator_ = None
        self.cluster_labels_ = None
        self.cluster_centers_ = None
        self.cluster_sizes_ = None
        self.n_clusters_ = None
        self.large_cluster_labels_ = None
        self.small_cluster_labels_ = None
        self.decision_scores_ = None
        self.threshold_ = None
        self.labels_ = None

    def fit(self, x, y=None):
        """训练CBLOF模型"""
        x = check_array(x)
        self._set_n_classes(y)
        n_samples, _ = x.shape

        # 参数验证
        check_parameter(
            self.alpha, low=0, high=1, param_name="alpha", include_left=False, include_right=False
        )
        check_parameter(self.beta, low=1, param_name="beta", include_left=False)

        # 执行聚类
        self.clustering_estimator_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )
        self.clustering_estimator_.fit(x)

        # 获取聚类结果
        self.cluster_labels_ = self.clustering_estimator_.labels_
        self.cluster_centers_ = self.clustering_estimator_.cluster_centers_
        self.cluster_sizes_ = np.bincount(self.cluster_labels_)
        self.n_clusters_ = len(self.cluster_sizes_)

        if self.n_clusters_ != self.n_clusters:
            logger.info(
                "CBLOF actual cluster count %d differs from requested %d",
                self.n_clusters_,
                self.n_clusters,
            )

        # 区分大小簇
        self._set_small_large_clusters(n_samples)

        # 计算异常分数
        self.decision_scores_ = self._compute_scores(x, self.cluster_labels_).ravel()
        self._process_decision_scores()

        return self

    def decision_function(self, x):
        """计算异常分数"""
        require_fitted(self, ["cluster_centers_", "threshold_"])
        x = check_array(x)

        # 预测聚类标签
        labels = self.clustering_estimator_.predict(x)

        # 计算异常分数
        return self._compute_scores(x, labels)

    def _set_small_large_clusters(self, n_samples):
        """区分大簇和小簇"""
        # 按簇大小排序（从大到小）
        sorted_indices = np.argsort(self.cluster_sizes_)[::-1]

        alpha_list = []
        beta_list = []

        for i in range(1, self.n_clusters_):
            # α条件：前i个簇的样本数占比
            temp_sum = np.sum(self.cluster_sizes_[sorted_indices[:i]])
            if temp_sum >= n_samples * self.alpha:
                alpha_list.append(i)

            # β条件：相邻簇大小比例
            ratio = self.cluster_sizes_[sorted_indices[i - 1]] / (
                self.cluster_sizes_[sorted_indices[i]] + 1e-10
            )
            if ratio >= self.beta:
                beta_list.append(i)

        # 找到同时满足条件的分割点
        intersection = np.intersect1d(alpha_list, beta_list)

        if len(intersection) > 0:
            threshold = intersection[0]
        elif len(alpha_list) > 0:
            threshold = alpha_list[0]
        elif len(beta_list) > 0:
            threshold = beta_list[0]
        else:
            threshold = 1
            logger.info("CBLOF falling back to default large/small cluster split threshold=1")

        self.large_cluster_labels_ = sorted_indices[:threshold]
        self.small_cluster_labels_ = sorted_indices[threshold:]

        logger.info(
            "CBLOF cluster split: large=%d small=%d (n_clusters=%d)",
            len(self.large_cluster_labels_),
            len(self.small_cluster_labels_),
            self.n_clusters_,
        )

    def _compute_scores(self, x, labels):
        """计算异常分数"""
        scores = np.zeros(x.shape[0])

        # 小簇中的样本：计算到最近大簇中心的距离
        small_mask = np.isin(labels, self.small_cluster_labels_)
        if small_mask.any():
            large_centers = self.cluster_centers_[self.large_cluster_labels_]
            dist_to_large = cdist(x[small_mask], large_centers)
            scores[small_mask] = np.min(dist_to_large, axis=1)

        # 大簇中的样本：计算到所属簇中心的距离
        large_mask = np.isin(labels, self.large_cluster_labels_)
        if large_mask.any():
            for label in self.large_cluster_labels_:
                label_mask = labels == label
                if label_mask.any():
                    center = self.cluster_centers_[label]
                    scores[label_mask] = np.linalg.norm(x[label_mask] - center, axis=1)

        # 使用簇大小作为权重
        if self.use_weights:
            scores = scores * self.cluster_sizes_[labels]

        return scores


# ===================================================================
#                     VisionCBLOF（主类）
# ===================================================================


@register_model(
    "vision_cblof",
    tags=("vision", "classical", "clustering"),
    metadata={
        "description": "基于 CBLOF 的视觉异常检测器",
        "paper": "He et al., SDM 2003",
        "year": 2003,
    },
)
class VisionCBLOF(BaseVisionDetector):
    """
    基于CBLOF算法的视觉异常检测器

    继承自BaseVisionDetector，符合统一的接口规范

    Parameters
    ----------
    contamination : float, optional (default=0.1)
        数据集中异常样本的比例

    feature_extractor : object, optional
        特征提取器实例，必须有extract方法

    n_clusters : int, optional (default=8)
        聚类数量

    alpha : float, optional (default=0.9)
        区分大小簇的系数（大簇样本占比）

    beta : float, optional (default=5)
        区分大小簇的系数（簇大小比例）

    use_weights : bool, optional (default=False)
        是否使用簇大小作为权重

    feature_method : str, optional (default='combined')
        特征提取方法（仅在未提供feature_extractor时使用）

    reduce_dim : bool, optional (default=True)
        是否PCA降维（仅在未提供feature_extractor时使用）

    n_components : int, optional (default=50)
        PCA维度（仅在未提供feature_extractor时使用）

    random_state : int, optional
        随机种子

    Examples
    --------
    >>> from vision_cblof import VisionCBLOF
    >>> # 使用默认特征提取器
    >>> detector = VisionCBLOF(n_clusters=8, contamination=0.1)
    >>> detector.fit(train_image_paths)
    >>> scores = detector.decision_function(test_image_paths)
    >>> labels = detector.predict(test_image_paths)
    """

    def __init__(
        self,
        contamination=0.1,
        feature_extractor=None,
        n_clusters=8,
        alpha=0.9,
        beta=5,
        use_weights=False,
        feature_method="combined",
        reduce_dim=True,
        n_components=50,
        random_state=None,
    ):
        # 保存CBLOF特定参数
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.use_weights = use_weights
        self.random_state = random_state

        # 如果未提供特征提取器，创建默认的
        if feature_extractor is None:
            feature_extractor = ImageFeatureExtractor(
                method=feature_method, reduce_dim=reduce_dim, n_components=n_components
            )
            logger.info(
                "CBLOF: using default feature extractor (method=%s, pca=%s)",
                feature_method,
                bool(reduce_dim),
            )

        # 调用父类构造函数
        super(VisionCBLOF, self).__init__(
            contamination=contamination, feature_extractor=feature_extractor
        )

    def _build_detector(self):
        """
        构建核心检测器实例
        BaseVisionDetector要求的接口
        """
        return CoreCBLOF(
            n_clusters=self.n_clusters,
            contamination=self.contamination,
            alpha=self.alpha,
            beta=self.beta,
            use_weights=self.use_weights,
            random_state=self.random_state,
        )

    def visualize_clusters(self):
        """可视化聚类结果"""
        from pyimgano.utils.optional_deps import require

        plt = require("matplotlib.pyplot", extra="viz", purpose="VisionCBLOF visualization")

        if not hasattr(self.detector, "cluster_labels_"):
            logger.warning("CBLOF: visualize_clusters called before fit()")
            return

        # 获取聚类信息
        n_clusters = self.detector.n_clusters_
        cluster_sizes = self.detector.cluster_sizes_
        large_clusters = self.detector.large_cluster_labels_

        # 创建可视化
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 簇大小分布
        colors = ["green" if i in large_clusters else "orange" for i in range(n_clusters)]

        ax1.bar(range(n_clusters), cluster_sizes, color=colors)
        ax1.set_xlabel("簇索引")
        ax1.set_ylabel("簇大小")
        ax1.set_title("簇大小分布")
        ax1.legend(["大簇", "小簇"])

        # 异常分数分布
        if hasattr(self, "decision_scores_"):
            ax2.hist(self.decision_scores_, bins=30, edgecolor="black")
            ax2.axvline(
                self.detector.threshold_,
                color="r",
                linestyle="--",
                label=f"阈值={self.detector.threshold_:.3f}",
            )
            ax2.set_xlabel("异常分数")
            ax2.set_ylabel("频数")
            ax2.set_title("异常分数分布")
            ax2.legend()

        plt.tight_layout()
        plt.show()

    def get_cluster_info(self):
        """获取聚类详细信息"""
        if not hasattr(self.detector, "cluster_labels_"):
            return {"status": "模型未训练"}

        info = {
            "算法": "Vision-CBLOF",
            "聚类数": self.detector.n_clusters_,
            "大簇": list(self.detector.large_cluster_labels_),
            "小簇": list(self.detector.small_cluster_labels_),
            "簇大小": {
                i: int(self.detector.cluster_sizes_[i]) for i in range(self.detector.n_clusters_)
            },
            "α参数": self.alpha,
            "β参数": self.beta,
            "使用权重": self.use_weights,
            "污染率": self.contamination,
            "阈值": float(self.detector.threshold_),
        }

        # 添加异常检测结果统计
        if hasattr(self.detector, "labels_"):
            n_anomalies = self.detector.labels_.sum()
            n_total = len(self.detector.labels_)
            info["训练集异常数"] = f"{n_anomalies}/{n_total} ({n_anomalies / n_total * 100:.1f}%)"

        return info


# ===================================================================
#                          使用示例
# ===================================================================

if __name__ == "__main__":
    print("Vision-CBLOF 异常检测器示例")
    print("=" * 60)

    # 示例1: 使用默认配置
    print("\n示例1: 默认配置")
    detector = VisionCBLOF(n_clusters=8, contamination=0.1, feature_method="combined")
    print("创建成功，特征方法: combined")

    # 示例2: 自定义特征提取器
    print("\n示例2: 自定义特征提取器")
    custom_extractor = ImageFeatureExtractor(method="texture", reduce_dim=False)

    detector2 = VisionCBLOF(feature_extractor=custom_extractor, n_clusters=5, alpha=0.8, beta=3)
    print("创建成功，特征方法: texture, 不使用PCA")

    # 示例3: 使用模拟数据演示完整流程
    print("\n示例3: 模拟数据演示")
    print("-" * 40)

    # 创建模拟特征提取器
    class MockExtractor:
        def extract(self, x):
            # 模拟正常数据和异常数据
            rng = np.random.default_rng(42)
            n_samples = len(x) if isinstance(x, list) else x.shape[0]
            n_features = 20

            # 90%正常数据（聚类分布）
            n_normal = int(n_samples * 0.9)
            normal_data = rng.standard_normal((n_normal, n_features))

            # 10%异常数据（离群点）
            n_anomaly = n_samples - n_normal
            anomaly_data = rng.standard_normal((n_anomaly, n_features)) * 3 + 5

            data = np.vstack([normal_data, anomaly_data])
            return data

    # 创建检测器
    mock_detector = VisionCBLOF(feature_extractor=MockExtractor(), n_clusters=3, contamination=0.1)

    # 训练
    train_paths = [f"img_{i}.jpg" for i in range(100)]
    print("训练模型...")
    mock_detector.fit(train_paths)

    # 测试
    test_paths = [f"test_{i}.jpg" for i in range(20)]
    scores = mock_detector.decision_function(test_paths)
    labels = mock_detector.predict(test_paths)

    print("\n检测结果:")
    print(f"  异常分数范围: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  检测到异常: {labels.sum()}/{len(labels)}")

    # 显示聚类信息
    info = mock_detector.get_cluster_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("示例完成")
