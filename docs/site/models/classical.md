---
title: 经典模型
---

# 经典模型 (Classical Models)

=== "中文"

    经典模型基于统计方法和机器学习算法，无需 GPU，训练和推理速度极快。
    适合 CPU 部署、快速基线验证和大规模批量筛查场景。

=== "English"

    Classical models are based on statistical methods and ML algorithms. No GPU required, with extremely fast training and inference.
    Ideal for CPU deployment, rapid baseline validation, and large-scale batch screening.

---

## 统计方法 (Statistical)

=== "中文"

    基于经验分布或统计量直接计算异常分数，通常无需参数调优。

=== "English"

    Compute anomaly scores directly from empirical distributions or statistics. Typically parameter-free.

| 模型名 | 注册名 | 特点 | 推荐场景 |
|:---|:---|:---|:---|
| ECOD | `core_ecod` / `vision_ecod` | 经验 CDF，无参数，极快 | 首选基线 |
| COPOD | `core_copod` / `vision_copod` | Copula 分布，无参数 | 高维数据 |
| MAD | `core_mad` / `vision_mad` | 中位数绝对偏差，鲁棒 | 鲁棒筛查 |
| RZScore | `core_rzscore` / `vision_rzscore` | 鲁棒 Z 分数 | 单特征异常 |
| Mahalanobis | `core_mahalanobis` | 马氏距离 | 多元高斯假设 |
| Elliptic Envelope | `core_elliptic_envelope` | 鲁棒协方差估计 | 高斯分布数据 |

!!! tip "推荐起步"
    `vision_ecod` 是最佳默认选择：无参数、极快、性能稳定。

### ECOD 示例

```python
from pyimgano import create_model

model = create_model("vision_ecod", contamination=0.05)
model.fit(train_images)
scores = model.decision_function(test_images)
predictions = model.predict(test_images)  # 1=异常, 0=正常
```

---

## 近邻方法 (Nearest Neighbor)

=== "中文"

    基于 k 近邻距离或局部密度估计异常程度。对局部异常模式敏感。

=== "English"

    Estimate anomaly degree based on k-nearest-neighbor distances or local density. Sensitive to local anomaly patterns.

| 模型名 | 注册名 | 特点 | 推荐场景 |
|:---|:---|:---|:---|
| KNN | `core_knn` / `vision_knn` | k 近邻距离 | 通用近邻基线 |
| LOF | `core_lof` / `vision_lof` | 局部离群因子 | 局部密度异常 |
| LoOP | `core_loop` / `vision_loop` | 概率化 LOF | 概率输出需求 |
| LDOF | `core_ldof` / `vision_ldof` | 局部距离离群因子 | 子空间异常 |
| ODIN | `core_odin` / `vision_odin` | 度中心性 | 图结构数据 |
| COF | `core_cof` / `vision_cof` | 链接离群因子 | 链式分布异常 |

### KNN 示例

```python
from pyimgano import create_model

model = create_model("vision_knn", n_neighbors=5, contamination=0.1)
model.fit(train_images)
scores = model.decision_function(test_images)
```

---

## 子空间 / 投影方法 (Subspace & Projection)

=== "中文"

    将数据投影到子空间，在低维空间中检测异常。适合高维数据降维后分析。

=== "English"

    Project data into subspaces and detect anomalies in lower dimensions. Suitable for high-dimensional data.

| 模型名 | 注册名 | 特点 | 推荐场景 |
|:---|:---|:---|:---|
| PCA | `core_pca` / `vision_pca` | 重构误差 | 线性结构数据 |
| PCA-MD | `core_pca_md` / `vision_pca_md` | PCA + 马氏距离 | 多元高斯 |
| LODA | `core_loda` / `vision_loda` | 轻量在线检测 | 流式/在线场景 |
| KPCA | `core_kpca` / `vision_kpca` | 核 PCA | 非线性结构 |

---

## 树方法 (Tree-based)

=== "中文"

    基于随机树的隔离或密度估计。天然适合高维数据，无分布假设。

=== "English"

    Tree-based isolation or density estimation. Naturally suited for high-dimensional data without distribution assumptions.

| 模型名 | 注册名 | 特点 | 推荐场景 |
|:---|:---|:---|:---|
| Isolation Forest | `core_iforest` / `vision_iforest` | 隔离深度 | 通用树基线 |
| RRCF | `core_rrcf` / `vision_rrcf` | 鲁棒随机割森林 | 流式数据 |
| HST | `core_hst` / `vision_hst` | 半空间树 | 流式/在线 |
| ExtraTreesDensity | `core_extra_trees_density` / `vision_extra_trees_density` | 极端随机树密度 | 密度估计 |

### IForest 示例

```python
from pyimgano import create_model

model = create_model("vision_iforest",
                     n_estimators=200,
                     contamination=0.05)
model.fit(train_images)
scores = model.decision_function(test_images)
```

---

## 聚类方法 (Clustering)

=== "中文"

    基于聚类结构判断异常：远离聚类中心或落入小簇的样本更可能是异常。

=== "English"

    Detect anomalies based on clustering structure: samples far from cluster centers or in small clusters are more likely anomalous.

| 模型名 | 注册名 | 特点 | 推荐场景 |
|:---|:---|:---|:---|
| CBLOF | `core_cblof` / `vision_cblof` | 基于聚类的局部离群 | 多簇数据 |
| KMeans | `core_kmeans` / `vision_kmeans` | K 均值距离 | 球形分布 |
| DBSCAN | `core_dbscan` / `vision_dbscan` | 密度聚类 | 任意形状簇 |
| GMM | `core_gmm` / `vision_gmm` | 高斯混合模型 | 多模态分布 |

---

## 集成方法 (Ensemble)

=== "中文"

    组合多个基检测器提升鲁棒性。适合无法确定最佳单一模型的场景。

=== "English"

    Combine multiple base detectors for improved robustness. Useful when the best single model is unknown.

| 模型名 | 注册名 | 特点 | 推荐场景 |
|:---|:---|:---|:---|
| FeatureBagging | `core_feature_bagging` | 特征子集集成 | 高维鲁棒 |
| LSCP | `core_lscp` | 局部选择组合 | 自适应集成 |
| SUOD | `core_suod` / `vision_suod` | 可扩展集成 | 大规模数据 |
| ScoreEnsemble | `core_score_ensemble` | 分数融合 | 多模型融合 |

---

## SVM 与密度方法 (SVM & Density)

| 模型名 | 注册名 | 特点 | 推荐场景 |
|:---|:---|:---|:---|
| OCSVM | `core_ocsvm` / `vision_ocsvm` | 单分类 SVM | 非线性边界 |
| KDE | `core_kde` / `vision_kde` | 核密度估计 | 低维密度 |
| HBOS | `core_hbos` / `vision_hbos` | 直方图离群 | 极快近似 |
| SOS | `core_sos` / `vision_sos` | 随机离群选择 | 中等规模 |

---

## 模板基线 (Template Baselines)

=== "中文"

    模板基线是面向工业场景的像素级检测方法。它们通过对比测试图与训练集模板，直接输出像素级异常热力图。
    **无需 GPU**，纯 NumPy/OpenCV 实现。

=== "English"

    Template baselines are pixel-level detection methods designed for industrial scenarios. They compare test images against training set templates and directly output pixel-level anomaly maps.
    **No GPU required** -- pure NumPy/OpenCV implementation.

| 注册名 | 方法 | 说明 |
|:---|:---|:---|
| `ssim_template_map` | SSIM 模板匹配 | 1 - SSIM 作为异常图 |
| `ssim_struct_map` | 结构 SSIM | 边缘图上的 SSIM |
| `vision_pixel_gaussian_map` | 逐像素高斯 | 均值 + 标准差 z-score |
| `vision_pixel_mad_map` | 逐像素 MAD | 鲁棒中位数绝对偏差 |
| `vision_pixel_mean_absdiff_map` | 逐像素均值差 | 均值模板绝对差 |
| `vision_template_ncc_map` | NCC 模板匹配 | 归一化互相关相似度 |
| `vision_phase_correlation_map` | 相位相关 | 频域对齐 + 绝对差 |

### SSIM 模板基线示例

```python
from pyimgano import create_model

model = create_model("ssim_template_map",
                     contamination=0.01,
                     resize_hw=(384, 512))
model.fit(reference_images)

# 像素级异常图
anomaly_map = model.get_anomaly_map(test_image)  # shape: (H, W), [0, 1]

# 图像级分数
scores = model.decision_function(test_images)
```

---

## 性能提示

!!! note "调优建议"

    === "中文"

        1. **先跑 `vision_ecod`**：作为零参数基线，性能通常出乎意料地好
        2. **`contamination` 参数**：设为预期异常比例 (默认 0.1)。工业场景常设 0.01-0.05
        3. **特征提取**：`vision_*` 模型内置特征提取；`core_*` 模型接受特征矩阵
        4. **大数据量**：优先 ECOD/COPOD/IForest（线性/对数复杂度），避免 KNN/LOF（二次复杂度）
        5. **工业流水线**：`vision_structural_*` 系列封装了特征提取 + core 检测器的完整管线

    === "English"

        1. **Start with `vision_ecod`**: as a zero-parameter baseline, often surprisingly good
        2. **`contamination` parameter**: set to expected anomaly ratio (default 0.1). Industrial scenarios often use 0.01-0.05
        3. **Feature extraction**: `vision_*` models have built-in feature extraction; `core_*` models accept feature matrices
        4. **Large datasets**: prefer ECOD/COPOD/IForest (linear/log complexity), avoid KNN/LOF (quadratic)
        5. **Industrial pipelines**: `vision_structural_*` series wrap feature extraction + core detector
