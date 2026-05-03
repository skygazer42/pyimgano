---
title: 模型注册表
---

# 完整模型注册表

=== "中文"

    pyimgano 通过统一的模型注册表管理所有检测器。目前共注册 **278** 个模型名称（包含别名），覆盖经典统计方法、深度学习模型和视觉语言模型。所有模型通过 `create_model()` 统一创建。

=== "English"

    pyimgano manages all detectors through a unified model registry. Currently **278** model names are registered (including aliases), covering classical statistical methods, deep learning models, and vision-language models. All models are created via `create_model()`.

---

## 模型发现

### CLI 发现

```bash
# 列出所有模型
pyimgano-benchmark --list-models

# 按标签过滤
pyimgano-benchmark --list-models --tags classical
pyimgano-benchmark --list-models --tags deep,pixel_map
pyimgano-benchmark --list-models --tags numpy

# 查看模型详情
pyimgano-benchmark --model-info vision_patchcore --json

# 目标导向发现
pyim --goal first-run --json
pyim --goal cpu-screening --json
pyim --goal pixel-localization --json
pyim --goal deployable --json

# 按目标和选择配置文件
pyim --list models --objective latency --selection-profile cpu-screening --topk 5
pyim --list models --objective localization --selection-profile balanced --topk 5

# 检查模型依赖
pyimgano-doctor --recommend-extras --for-model vision_patchcore --json
```

### Python 发现

```python
from pyimgano.models import list_models, create_model

# 列出所有模型
all_models = list_models()
print(f"Total: {len(all_models)}")

# 按标签过滤
classical = list_models(tags=["classical"])
deep = list_models(tags=["deep"])
pixel_map = list_models(tags=["pixel_map"])
numpy_models = list_models(tags=["numpy"])

# 创建模型
detector = create_model("vision_ecod", contamination=0.1)
```

---

## 标签系统

=== "中文"

    每个模型注册时附带标签，用于分类和发现。

=== "English"

    Each model is registered with tags for classification and discovery.

| 标签 | 说明 |
|------|------|
| `classical` | 经典统计/机器学习方法 |
| `deep` | 深度学习方法（需要 `torch`） |
| `vision` | 视觉模型包装器（接受图像输入） |
| `core` | 核心检测器（接受特征矩阵输入） |
| `pixel_map` | 支持像素级异常图输出 |
| `numpy` | numpy-first 接口（工业集成友好） |
| `ensemble` | 集成方法 |
| `neighbors` | 基于近邻的方法 |
| `density` | 密度估计方法 |
| `clustering` | 聚类方法 |
| `projection` | 投影/子空间方法 |
| `parameter-free` | 无需参数调优 |
| `autoencoder` | 自编码器 |
| `memory_bank` | 记忆库方法（PatchCore 系列） |
| `pipeline` | 特征提取 + 检测器管线 |
| `industrial` | 工业场景优化管线 |
| `embeddings` | 使用嵌入特征 |
| `template` | 模板匹配方法 |
| `backend` | 外部后端封装（anomalib 等） |
| `onnx` | ONNX Runtime 嵌入 |
| `torchscript` | TorchScript 嵌入 |
| `sota` | 当前 SOTA 水平 |

!!! note "自动标签"
    如果模型类定义了 `predict_anomaly_map()` 或 `get_anomaly_map()` 方法，
    注册时会自动添加 `pixel_map` 标签，无需手动指定。

---

## 经典模型 (Classical)

### 统计与无参数方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_ecod` | classical, core, parameter-free | ECOD 经验累积分布异常检测 |
| `core_copod` | classical, core, parameter-free | COPOD 基于 Copula 的异常检测 |
| `core_mad` | classical, core, robust | MAD 中位绝对偏差基线 |
| `core_rzscore` | classical, core, robust | 稳健 z-score (中位数 + MAD) |
| `core_mahalanobis` | classical, core, distance | Mahalanobis 距离基线 |
| `core_mahalanobis_shrinkage` | classical, core, distance, shrinkage | Ledoit-Wolf 协方差收缩 Mahalanobis |
| `core_cosine_mahalanobis` | classical, core, distance, cosine | L2 归一化嵌入上的 Mahalanobis |
| `core_mcd` | classical, core, robust | 最小协方差行列式 (MCD) |
| `core_elliptic_envelope` | classical, core, gaussian | 稳健协方差 Mahalanobis 异常基线 |
| `core_hbos` | classical, core, histogram | 基于直方图的异常评分 |
| `vision_ecod` | classical, vision | ECOD 视觉包装器 |
| `vision_copod` | classical, vision | COPOD 视觉包装器 |

### 近邻方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_knn` | classical, core, neighbors | kNN 异常检测 |
| `core_knn_cosine` | classical, core, neighbors, cosine | 余弦 kNN 距离异常检测 |
| `core_knn_cosine_calibrated` | classical, core, neighbors, calibration | 校准余弦 kNN |
| `core_lof` | classical, core, neighbors, density | 局部异常因子 (LOF) |
| `core_loop` | classical, core, neighbors, probability | 局部异常概率 (LoOP) |
| `core_ldof` | classical, core, neighbors | 局部距离异常因子 (LDOF) |
| `core_cof` | classical, core, neighbors | 连通异常因子 (COF) |
| `core_odin` | classical, core, neighbors, graph | 基于 kNN 图入度的 ODIN |
| `core_lid` | classical, core, neighbors | 局部内在维度 (LID) |
| `core_oddoneout` | classical, core, neighbors | Odd-One-Out 邻居比较 |
| `vision_knn` | classical, vision | kNN 视觉包装器 |
| `vision_lof` | classical, vision | LOF 视觉包装器 |

### 树方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_iforest` | classical, core, ensemble | 隔离森林 (Isolation Forest) |
| `core_rrcf` | classical, core, forest | 随机割集森林 (RRCF) |
| `core_hst` | classical, core, tree, online | 半空间树 (Half-Space Trees) |
| `core_extra_trees_density` | classical, core, trees, density | 随机树嵌入叶稀有度密度基线 |
| `vision_iforest` | classical, vision | 隔离森林视觉包装器 |

### 子空间与投影方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_pca` | classical, core, linear | PCA 重构误差异常检测 |
| `core_pca_md` | classical, core, pca, distance | PCA + Mahalanobis 距离 |
| `core_kpca` | classical, core, kernel | Kernel PCA 异常检测 |
| `core_loda` | classical, core, projection, density | LODA 轻量在线检测 |
| `core_sod` | classical, core, subspace | 子空间异常检测 (SOD) |
| `core_random_projection_knn` | classical, core, projection | 随机投影 + kNN |

### 聚类方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_cblof` | classical, core, clustering | 基于聚类的局部异常因子 |
| `core_kmeans` | classical, core, clustering | KMeans 质心距离基线 |
| `core_dbscan` | classical, core, clustering, density | DBSCAN 核心集距离检测 |
| `core_gmm` | classical, core, density | 高斯混合模型检测 |
| `core_dtc` | classical, core, distance | 质心距离基线 (DTC) |

### 集成方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_feature_bagging` | classical, core, ensemble | 特征子抽样集成 |
| `core_feature_bagging_spec` | classical, core, ensemble | JSON 配置友好特征集成 |
| `core_lscp` | classical, core, ensemble | LSCP 局部选择集成 |
| `core_suod` | classical, core, ensemble | SUOD 评分集成 |
| `core_score_ensemble` | classical, core, ensemble, score | 评分集成包装器 |
| `vision_score_ensemble` | classical, vision, ensemble | 视觉评分集成 |

### SVM 与密度方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_ocsvm` | classical, core, svm | 单类 SVM (One-Class SVM) |
| `core_kde` | classical, core, density | 核密度估计 (KDE) |
| `core_kde_ratio` | classical, core, density | 双带宽 KDE 密度对比 |
| `core_sos` | classical, core, probabilistic | 随机异常选择 (SOS) |

### 其他经典方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_sampling` | classical, core, distance | 采样距离异常检测 |
| `core_rgraph` | classical, core, graph | 图随机游走异常检测 |
| `core_rod` | classical, core | 旋转异常检测 (ROD) |
| `core_qmcd` | classical, core, robust | QMCD 差异检测 |
| `core_imdd` | classical, core | IMDD/LMDD 偏差检测 |
| `core_lmdd` | classical, core | LMDD 偏差检测 |
| `core_loci` | classical, core, density | LOCI 异常检测 |
| `core_inne` | classical, core, isolation | INNE 隔离异常检测 |
| `core_dcorr` | classical, core, projection | 距离相关影响检测 |
| `core_mst_outlier` | classical, core, graph | MST 图异常基线 |
| `core_neighborhood_entropy` | classical, core, graph | kNN 邻域熵异常评分 |
| `core_studentized_residual` | classical, core, pca | PCA 学生化残差基线 |
| `core_cook_distance` | classical, core, pca | Cook 距离影响评分 |
| `core_knn_degree` | classical, core, graph | kNN 度数异常检测 |
| `core_padim_lite` | classical, core, gaussian | PaDiM-lite 高斯嵌入基线 |
| `core_patchcore_lite` | classical, core, memory_bank | PatchCore-lite 核心集 + NN |
| `core_patchcore_online` | classical, core, memory_bank, online | 增量 PatchCore |
| `core_crossmad` | classical, core, prototype | CrossMAD 原型距离检测 |
| `core_score_standardizer` | classical, core, calibration | 评分标准化包装器 |

---

## 深度学习模型 (Deep)

### 记忆库方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `vision_patchcore` | deep, vision, memory_bank, pixel_map | PatchCore（CVPR 2022） |
| `vision_softpatch` | deep, vision, memory_bank, pixel_map | SoftPatch 鲁棒记忆库 |
| `vision_spade` | deep, vision, neighbors, pixel_map | SPADE 子图像异常检测 |

### 师生网络方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `vision_stfpm` | deep, vision, pixel_map | STFPM 师生特征金字塔匹配 |
| `efficient_ad` | deep, vision, distillation | EfficientAD 师生嵌入蒸馏 |
| `vision_reverse_distillation` | deep, vision, distillation | 反向蒸馏 |
| `vision_reverse_dist` | deep, vision, distillation | 反向蒸馏（别名） |

### 流模型方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `vision_fastflow` | deep, vision, pixel_map | FastFlow 快速正则化流 |
| `vision_cflow` | deep, vision, pixel_map | CFlow 条件正则化流 |

### 重构方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `vision_draem` | deep, vision, pixel_map | DRAEM 重构+判别 |
| `ae_resnet_unet` | deep, autoencoder, reconstruction | ResNet UNet 自编码器 |
| `core_torch_autoencoder` | deep, core, autoencoder | MLP 自编码器 |

### 自监督方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `vision_cutpaste` | deep, vision, self-supervised | CutPaste（CVPR 2021） |
| `cutpaste` | deep, vision, self-supervised | CutPaste（别名） |

### 基础模型方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `vision_simplenet` | deep, vision, pixel_map | SimpleNet（CVPR 2023） |
| `vision_anomalydino` | deep, vision, pixel_map | AnomalyDINO (DINOv2) |
| `vision_mambaad` | deep, vision, pixel_map | MambaAD 序列建模 |

### 其他深度方法

| 名称 | 标签 | 说明 |
|------|------|------|
| `core_deep_svdd` | deep, core, one-class | DeepSVDD 单类深度检测 |
| `vision_deep_svdd` | deep, vision, one-class | DeepSVDD 视觉包装器 |
| `vision_padim` | deep, vision, pixel_map | PaDiM 概率嵌入 |
| `vision_padim_lite` | deep, vision, pixel_map | PaDiM-lite 轻量嵌入 |
| `vision_dfm` | deep, vision | DFM 深度特征建模 |
| `vision_patchcore_lite` | deep, vision, memory_bank | PatchCore-lite 轻量版 |
| `vision_patchcore_online` | deep, vision, memory_bank, online | 增量 PatchCore |

---

## 视觉语言模型 (VLM)

| 名称 | 标签 | 说明 |
|------|------|------|
| `vision_winclip` | vision, deep, clip, pixel_map | WinCLIP 零样本/少样本检测 |
| `winclip` | vision, deep, clip, pixel_map | WinCLIP（别名） |
| `vision_openclip_patch_map` | vision, deep, clip, pixel_map | OpenCLIP patch 模板距离检测 |
| `vision_promptad` | vision, deep, few-shot, sota | PromptAD 提示学习少样本检测 |
| `vision_anomalydino` | vision, deep, dinov2, pixel_map | AnomalyDINO (DINOv2 嵌入) |

---

## 像素级模板基线 (Pixel Template)

| 名称 | 标签 | 说明 |
|------|------|------|
| `ssim_template_map` | vision, numpy, pixel_map | SSIM 模板相似度异常图 |
| `ssim_struct_map` | vision, numpy, pixel_map | 结构 SSIM（边缘图） |
| `vision_template_ncc_map` | vision, numpy, pixel_map | 局部 NCC 模板异常图 |
| `vision_phase_correlation_map` | vision, numpy, pixel_map | 相位相关对齐 + 差异图 |
| `vision_pixel_gaussian_map` | vision, numpy, pixel_map | 逐像素高斯 z-score 图 |
| `vision_pixel_mad_map` | vision, numpy, pixel_map | 逐像素 MAD z-score 图 |
| `vision_pixel_mean_absdiff_map` | vision, numpy, pixel_map | 逐像素均值绝对差异图 |

---

## 工业管线模型 (Industrial Pipelines)

=== "中文"

    工业管线模型组合特征提取器（structural / resnet18 / TorchScript / ONNX）和 core 检测器，提供端到端的视觉异常检测管线。

=== "English"

    Industrial pipeline models combine feature extractors (structural / resnet18 / TorchScript / ONNX) with core detectors, providing end-to-end visual anomaly detection pipelines.

### Structural 系列

| 名称 | 核心检测器 | 说明 |
|------|-----------|------|
| `vision_structural_ecod` | `core_ecod` | 结构特征 + ECOD |
| `vision_structural_copod` | `core_copod` | 结构特征 + COPOD |
| `vision_structural_knn` | `core_knn` | 结构特征 + KNN |
| `vision_structural_lof` | `core_lof` | 结构特征 + LOF |
| `vision_structural_iforest` | `core_iforest` | 结构特征 + IForest |
| `vision_structural_extra_trees_density` | `core_extra_trees_density` | 结构特征 + ExtraTrees |
| `vision_structural_mcd` | `core_mcd` | 结构特征 + MCD |
| `vision_structural_pca_md` | `core_pca_md` | 结构特征 + PCA-MD |
| `vision_structural_lid` | `core_lid` | 结构特征 + LID |
| `vision_structural_mst_outlier` | `core_mst_outlier` | 结构特征 + MST |

### ResNet18 嵌入系列

| 名称 | 核心检测器 | 说明 |
|------|-----------|------|
| `vision_resnet18_ecod` | `core_ecod` | ResNet18 嵌入 + ECOD |
| `vision_resnet18_copod` | `core_copod` | ResNet18 嵌入 + COPOD |
| `vision_resnet18_iforest` | `core_iforest` | ResNet18 嵌入 + IForest |
| `vision_resnet18_knn` | `core_knn` | ResNet18 嵌入 + KNN |
| `vision_resnet18_knn_cosine` | `core_knn_cosine` | ResNet18 嵌入 + 余弦 KNN |
| `vision_resnet18_knn_cosine_calibrated` | `core_knn_cosine_calibrated` | ResNet18 嵌入 + 校准余弦 KNN |
| `vision_resnet18_cosine_mahalanobis` | `core_cosine_mahalanobis` | ResNet18 嵌入 + 余弦 Mahalanobis |
| `vision_resnet18_lof` | `core_lof` | ResNet18 嵌入 + LOF |
| `vision_resnet18_mcd` | `core_mcd` | ResNet18 嵌入 + MCD |
| `vision_resnet18_pca_md` | `core_pca_md` | ResNet18 嵌入 + PCA-MD |
| `vision_resnet18_lid` | `core_lid` | ResNet18 嵌入 + LID |
| `vision_resnet18_mst_outlier` | `core_mst_outlier` | ResNet18 嵌入 + MST |
| `vision_resnet18_extra_trees_density` | `core_extra_trees_density` | ResNet18 嵌入 + ExtraTrees |
| `vision_resnet18_oddoneout` | `core_oddoneout` | ResNet18 嵌入 + OddOneOut |
| `vision_resnet18_mahalanobis_shrinkage` | `core_mahalanobis_shrinkage` | ResNet18 嵌入 + 收缩 Mahalanobis |
| `vision_resnet18_torch_ae` | `core_torch_autoencoder` | ResNet18 嵌入 + Autoencoder |

### ONNX Runtime 系列

=== "中文"

    需要提供 ONNX 模型检查点路径。适合无 PyTorch 环境的部署场景。

=== "English"

    Requires an ONNX model checkpoint path. Suitable for deployment without PyTorch.

| 名称 | 核心检测器 |
|------|-----------|
| `vision_onnx_ecod` | `core_ecod` |
| `vision_onnx_copod` | `core_copod` |
| `vision_onnx_iforest` | `core_iforest` |
| `vision_onnx_knn_cosine` | `core_knn_cosine` |
| `vision_onnx_knn_cosine_calibrated` | `core_knn_cosine_calibrated` |
| `vision_onnx_cosine_mahalanobis` | `core_cosine_mahalanobis` |
| `vision_onnx_lof` | `core_lof` |
| `vision_onnx_mcd` | `core_mcd` |
| `vision_onnx_pca_md` | `core_pca_md` |
| `vision_onnx_lid` | `core_lid` |
| `vision_onnx_mst_outlier` | `core_mst_outlier` |
| `vision_onnx_extra_trees_density` | `core_extra_trees_density` |
| `vision_onnx_oddoneout` | `core_oddoneout` |

### TorchScript 系列

| 名称 | 核心检测器 |
|------|-----------|
| `vision_torchscript_ecod` | `core_ecod` |
| `vision_torchscript_copod` | `core_copod` |
| `vision_torchscript_iforest` | `core_iforest` |
| `vision_torchscript_knn_cosine` | `core_knn_cosine` |
| `vision_torchscript_knn_cosine_calibrated` | `core_knn_cosine_calibrated` |
| `vision_torchscript_cosine_mahalanobis` | `core_cosine_mahalanobis` |
| `vision_torchscript_lof` | `core_lof` |
| `vision_torchscript_mcd` | `core_mcd` |
| `vision_torchscript_pca_md` | `core_pca_md` |
| `vision_torchscript_lid` | `core_lid` |
| `vision_torchscript_extra_trees_density` | `core_extra_trees_density` |
| `vision_torchscript_oddoneout` | `core_oddoneout` |
| `vision_torchscript_mst_outlier` | `core_mst_outlier` |

---

## Anomalib 后端模型

=== "中文"

    需要安装 `pyimgano[anomalib]`。这些模型封装 anomalib 训练的检查点用于推理。

=== "English"

    Requires `pyimgano[anomalib]`. These models wrap anomalib-trained checkpoints for inference.

| 名称 | 对应算法 | 说明 |
|------|---------|------|
| `vision_anomalib_checkpoint` | 通用 | 通用 anomalib 检查点加载器 |
| `vision_patchcore_anomalib` | PatchCore | CVPR 2022 记忆库方法 |
| `vision_padim_anomalib` | PaDiM | 概率嵌入分布 |
| `vision_stfpm_anomalib` | STFPM | 师生特征金字塔 |
| `vision_draem_anomalib` | DRAEM | 重构+判别 |
| `vision_fastflow_anomalib` | FastFlow | 正则化流 |
| `vision_reverse_distillation_anomalib` | Reverse Distillation | 反向蒸馏 |
| `vision_efficientad_anomalib` | EfficientAD | 高效师生蒸馏 |
| `vision_cflow_anomalib` | CFlow | 条件正则化流 |
| `vision_dfm_anomalib` | DFM | 深度特征建模 |
| `vision_dinomaly_anomalib` | Dinomaly | DINOv2 重构 |
| `vision_cfa_anomalib` | CFA | 耦合超球特征适应 |
| `vision_csflow_anomalib` | CS-Flow | 跨尺度正则化流 |
| `vision_dfkde_anomalib` | DFKDE | 深度特征 KDE |
| `vision_dsr_anomalib` | DSR | 双子空间重投影 |
| `vision_ganomaly_anomalib` | GANomaly | GAN 异常检测 |
| `vision_rkde_anomalib` | R-KDE | 区域 KDE |
| `vision_uflow_anomalib` | U-Flow | U 形正则化流 |
| `vision_winclip_anomalib` | WinCLIP | CLIP 零样本 |
| `vision_fre_anomalib` | FRE | 快速重构误差 |
| `vision_supersimplenet_anomalib` | SuperSimpleNet | 统一有/无监督 |
| `vision_vlmad_anomalib` | VLM-AD | 视觉语言模型 |

---

## 特殊 / 工具模型

| 名称 | 标签 | 说明 |
|------|------|------|
| `vision_embedding_core` | vision, pipeline | 嵌入 + 核心检测器通用组合路径 |
| `vision_feature_pipeline` | vision, pipeline | 自定义特征提取管线 |
| `vision_score_standardizer` | vision, calibration | 视觉分数标准化封装 |
| `core_score_standardizer` | core, calibration | Core 分数标准化封装 |

---

## 自定义模型注册

=== "中文"

    通过 `@register_model` 装饰器注册自定义模型，即可在 CLI 和注册表中使用。

=== "English"

    Register custom models via the `@register_model` decorator to make them available in CLI and registry.

```python
import numpy as np
from pyimgano.models.registry import register_model

@register_model(
    "my_custom_detector",
    tags=("vision", "classical", "custom"),
    metadata={"description": "My custom industrial detector"},
)
class MyCustomDetector:
    def __init__(self, *, contamination: float = 0.1):
        self.contamination = float(contamination)
        self.threshold_ = None

    def fit(self, X, y=None):
        scores = self.decision_function(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def decision_function(self, X):
        # 实现异常评分逻辑
        return np.zeros(len(list(X)), dtype=np.float32)

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(int)
```

!!! tip "像素级支持"

    如需支持像素级异常图，额外实现：

    ```python
    def get_anomaly_map(self, image):
        """返回 (H, W) 异常图，值越高越异常"""
        ...

    # 或批量版本
    def predict_anomaly_map(self, X):
        """返回 (N, H, W) 异常图"""
        ...
    ```

!!! info "可选依赖"

    如果模型依赖可选库，使用懒加载并提供安装提示：

    ```python
    def fit(self, X, y=None):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "This model requires PyTorch. "
                "Install with: pip install 'pyimgano[torch]'"
            )
        ...
    ```

---

## 注册 API 参考

```python
from pyimgano.models.registry import (
    register_model,    # 装饰器：注册模型
    create_model,      # 创建模型实例
    list_models,       # 列出模型名称
    model_info,        # 查看模型元数据
)
```

| 函数 | 说明 |
|------|------|
| `register_model(name, *, tags, metadata, overwrite)` | 装饰器，注册模型构造器到全局注册表 |
| `create_model(name, *args, **kwargs)` | 通过注册名创建模型实例 |
| `list_models(*, tags)` | 列出模型名称（可按标签筛选），返回排序列表 |
| `model_info(name)` | 返回模型的 `ModelEntry`（含 tags 和 metadata） |

!!! tip "最佳实践"

    === "中文"

        - 为视觉模型添加 `vision` 标签，core 模型添加 `core` 标签
        - 实现 `get_anomaly_map()` 的模型会自动获得 `pixel_map` 标签
        - `metadata` 中建议包含 `description`、`paper`、`year` 字段
        - 模型名应使用 `snake_case`，视觉模型建议以 `vision_` 为前缀

    === "English"

        - Add `vision` tag for vision models, `core` tag for core models
        - Models implementing `get_anomaly_map()` automatically get the `pixel_map` tag
        - Include `description`, `paper`, `year` in `metadata`
        - Use `snake_case` for model names; prefix vision models with `vision_`
