import os

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm

from .registry import register_model

ONE_CLASS_CNN_FEATURE_EXTRACTION = "one_class_cnn CNN feature extraction"



@register_model(
    "one_class_cnn",
    tags=("vision", "classical", "svm"),
    metadata={"description": "基于多特征的一类 SVM 图像检测器"},
)
class ImageAnomalyDetector:
    def __init__(
        self,
        feature_type="combined",
        nu=0.05,
        cnn_pretrained: bool = False,
        random_state: int | None = 0,
    ):
        """
        初始化异常检测器
        feature_type: 'histogram', 'hog', 'combined', 'cnn'
        nu: 异常比例上界（0.05表示最多5%被认为是异常）
        """
        self.feature_type = feature_type
        self.cnn_pretrained = bool(cnn_pretrained)
        self.ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=nu)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=128, random_state=0)  # 降维到128维
        self.is_trained = False

        self.pca.random_state = random_state

    def _iter_image_paths(self, x):
        if isinstance(x, (str, os.PathLike)):
            raw = str(x)
            if os.path.isdir(raw):
                return sorted(
                    os.path.join(raw, name)
                    for name in os.listdir(raw)
                    if name.endswith(".jpg") or name.endswith(".png")
                )
            return [raw]
        return [str(item) for item in x]

    def extract_features(self, image_path):
        """提取图像特征"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []

        if self.feature_type in ["histogram", "combined"]:
            # 1. 颜色直方图特征
            hist_b = cv2.calcHist([img], [0], None, [64], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [64], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [64], [0, 256])
            hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])

            # 归一化
            hist_features = np.concatenate(
                [
                    hist_b.flatten() / (hist_b.sum() + 1e-6),
                    hist_g.flatten() / (hist_g.sum() + 1e-6),
                    hist_r.flatten() / (hist_r.sum() + 1e-6),
                    hist_gray.flatten() / (hist_gray.sum() + 1e-6),
                ]
            )
            features.extend(hist_features)

        if self.feature_type in ["hog", "combined"]:
            # 2. HOG特征（边缘和形状）
            from pyimgano.utils.optional_deps import require

            skfeature = require(
                "skimage.feature", extra="skimage", purpose="HOG features (one_class_cnn)"
            )
            hog = skfeature.hog
            resized = cv2.resize(gray, (128, 128))
            hog_features = hog(
                resized,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                feature_vector=True,
            )
            features.extend(hog_features)

        if self.feature_type == "combined":
            # 3. 纹理特征（LBP）
            from pyimgano.utils.optional_deps import require

            skfeature = require(
                "skimage.feature", extra="skimage", purpose="LBP features (one_class_cnn)"
            )
            local_binary_pattern = skfeature.local_binary_pattern
            resized = cv2.resize(gray, (128, 128))
            lbp = local_binary_pattern(resized, P=8, R=1, method="uniform")
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= lbp_hist.sum() + 1e-6
            features.extend(lbp_hist)

            # 4. 统计特征
            features.extend(
                [
                    gray.mean(),
                    gray.std(),
                    cv2.Laplacian(gray, cv2.CV_64F).var(),  # 模糊度
                    img.mean(),
                    img.std(),
                ]
            )

        return np.array(features)

    def extract_cnn_features(self, image_path):
        """使用预训练CNN提取特征（需要深度学习框架）"""
        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose=ONE_CLASS_CNN_FEATURE_EXTRACTION)
        transforms = require(
            "torchvision.transforms", extra="torch", purpose=ONE_CLASS_CNN_FEATURE_EXTRACTION
        )
        models = require(
            "torchvision.models", extra="torch", purpose=ONE_CLASS_CNN_FEATURE_EXTRACTION
        )

        # 加载预训练模型
        try:
            weights = models.ResNet18_Weights.DEFAULT if self.cnn_pretrained else None
            model = models.resnet18(weights=weights)
        except Exception:  # pragma: no cover - fallback for older torchvision
            model = models.resnet18(pretrained=bool(self.cnn_pretrained))
        model.eval()

        # 移除最后的分类层
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

        # 图像预处理
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # 读取和预处理图像
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0)

        # 提取特征
        with torch.no_grad():
            features = feature_extractor(img_tensor)
            features = features.squeeze().numpy()

        return features

    def train(self, image_folder):
        """训练模型"""
        print("开始提取特征...")

        image_paths = self._iter_image_paths(image_folder)

        # 提取所有图片的特征
        features_list = []
        for img_path in tqdm(image_paths, desc="提取特征"):
            try:
                if self.feature_type == "cnn":
                    feat = self.extract_cnn_features(img_path)
                else:
                    feat = self.extract_features(img_path)
                features_list.append(feat)
            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")
                continue

        x = np.array(features_list)
        print(f"提取了 {x.shape[0]} 个样本的特征，每个样本 {x.shape[1]} 维")

        # 数据预处理
        print("标准化特征...")
        x_scaled = self.scaler.fit_transform(x)

        # PCA降维（可选，用于加速和去噪）
        if x.shape[1] > 128:
            print("PCA降维...")
            max_components = max(1, min(128, x_scaled.shape[0], x_scaled.shape[1]))
            self.pca = PCA(n_components=max_components, random_state=self.pca.random_state)
            x_reduced = self.pca.fit_transform(x_scaled)
        else:
            x_reduced = x_scaled

        # 训练 One-Class SVM
        print("训练 One-Class SVM...")
        self.ocsvm.fit(x_reduced)

        self.is_trained = True
        print("训练完成！")

        # 计算训练集上的阈值（用于自适应阈值）
        train_scores = self.ocsvm.decision_function(x_reduced)
        self.threshold_percentile_95 = np.percentile(train_scores, 5)  # 95%的正常样本的阈值

        return self

    def _predict_single(self, image_path):
        """预测单张图片是否异常"""
        if not self.is_trained:
            raise ValueError("模型尚未训练！")

        # 提取特征
        if self.feature_type == "cnn":
            feat = self.extract_cnn_features(image_path)
        else:
            feat = self.extract_features(image_path)

        # 预处理
        feat_scaled = self.scaler.transform([feat])
        if hasattr(self, "pca") and feat_scaled.shape[1] > 128:
            feat_reduced = self.pca.transform(feat_scaled)
        else:
            feat_reduced = feat_scaled

        # 预测
        prediction = self.ocsvm.predict(feat_reduced)[0]
        score = self.ocsvm.decision_function(feat_reduced)[0]

        # 计算异常概率（归一化分数）
        anomaly_score = -score  # 分数越低越异常，转换为越高越异常

        return {
            "prediction": "normal" if prediction == 1 else "anomaly",
            "is_normal": prediction == 1,
            "anomaly_score": anomaly_score,
            "decision_value": score,
            "confidence": abs(score - self.threshold_percentile_95),
        }

    def fit(self, x, y=None):
        del y
        self.train(x)
        return self

    def decision_function(self, x):
        image_paths = self._iter_image_paths(x)
        scores = []
        for image_path in image_paths:
            result = self._predict_single(image_path)
            scores.append(float(result["anomaly_score"]))
        return np.asarray(scores, dtype=np.float64)

    def predict(self, x):
        if isinstance(x, (str, os.PathLike)) and os.path.isfile(str(x)):
            return self._predict_single(str(x))

        image_paths = self._iter_image_paths(x)
        labels = []
        for image_path in image_paths:
            result = self._predict_single(image_path)
            labels.append(0 if result["is_normal"] else 1)
        return np.asarray(labels, dtype=np.int64)

    def batch_predict(self, test_folder):
        """批量预测"""
        results = []

        for filename in os.listdir(test_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(test_folder, filename)
                try:
                    result = self._predict_single(img_path)
                    result["filename"] = filename
                    results.append(result)
                except Exception as e:
                    print(f"预测 {filename} 时出错: {e}")

        return results

    def save_model(self, save_path):
        """保存模型"""
        model_data = {
            "ocsvm": self.ocsvm,
            "scaler": self.scaler,
            "pca": self.pca if hasattr(self, "pca") else None,
            "feature_type": self.feature_type,
            "threshold_percentile_95": self.threshold_percentile_95,
            "is_trained": self.is_trained,
        }
        joblib.dump(model_data, save_path)
        print(f"模型已保存到: {save_path}")

    def load_model(self, load_path):
        """加载模型"""
        model_data = joblib.load(load_path)
        self.ocsvm = model_data["ocsvm"]
        self.scaler = model_data["scaler"]
        self.pca = model_data["pca"]
        self.feature_type = model_data["feature_type"]
        self.threshold_percentile_95 = model_data["threshold_percentile_95"]
        self.is_trained = model_data["is_trained"]
        print(f"模型已从 {load_path} 加载")


# 使用示例
if __name__ == "__main__":
    # 1. 创建检测器
    detector = ImageAnomalyDetector(
        feature_type="combined", nu=0.05
    )  # 使用组合特征  # 假设最多5%的训练数据可能是异常

    # 2. 训练模型
    train_folder = "/Computer/data/temp11/程序正常"
    detector.train(train_folder)

    # 3. 保存模型
    detector.save_model("anomaly_detector_model.pkl")

    # 4. 测试单张图片
    test_image = "/Computer/data/temp11/程序正常/0a6c84ef7e1942529f46fea50ed5dfab.jpg"
    result = detector.predict(test_image)
    print(f"\n预测结果: {result}")

    # 5. 批量测试（如果有测试文件夹）
    # test_results = detector.batch_predict("/path/to/test/folder")
