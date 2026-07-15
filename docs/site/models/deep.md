---
title: 深度学习模型
---

# 深度学习模型 (Deep Learning Models)

=== "中文"

    PyImgAno 同时包含原生核心实现、论文适配、紧凑变体、实验代理和外部检查点封装。
    模型名字本身不代表论文复现；使用前请查看 `paper_fidelity`。

=== "English"

    PyImgAno includes native core implementations, paper adaptations, compact
    variants, experimental proxies, and external checkpoint wrappers. A model
    name alone does not claim a paper reproduction; inspect `paper_fidelity`.

```python
from pyimgano.models import model_info

print(model_info("vision_patchcore")["metadata"])
```

## 论文关系 (Fidelity)

| 值 | 含义 |
|---|---|
| `core-aligned` | 实现论文定义性的结构、目标和评分流程；不保证复现实验数值 |
| `paper-adaptation` | 保留核心目标，但适配了输入或部署场景 |
| `partial` | 实现部分算法或紧凑变体 |
| `inspired` | 仅为实验代理，不是论文实现 |
| `external-backend` | 加载上游实现训练的检查点 |
| `not-applicable` | 通用基线，不声明论文复现 |

## 原生核心实现

| 注册名 | 方法 |
|---|---|
| `vision_patchcore` | 局部补丁聚合、coreset、近邻重加权 |
| `vision_padim` | 固定通道采样和逐位置高斯分布 |
| `vision_stfpm` | 多层师生特征匹配和乘积异常图 |
| `vision_cflow` | 三尺度条件归一化流与位置编码 |
| `vision_devnet` | 2021 图像网络、两尺度 top-K MIL 与偏差损失（无定位接口） |
| `vision_reverse_distillation` | WRN50-2 教师、OCBE 与反向 WRN 解码器 |
| `vision_simplenet` | 论文 3×3 补丁嵌入、特征适配器与噪声判别器 |
| `vision_spade` | 图像检索与深层金字塔对应 |
| `vision_cutpaste` | CutPaste 三分类自监督 |

!!! warning "预训练权重"
    多数模型默认 `pretrained=False`，用于离线安全和测试。结构正确不等于实验有效；
    复现实验还需要论文权重、数据划分、预处理、训练计划和评估协议。

## 最小示例

```python
from pyimgano import create_model

model = create_model(
    "vision_patchcore",
    pretrained=True,
    coreset_sampling_ratio=0.1,
    device="cuda",
)
model.fit(normal_images)
scores = model.decision_function(test_images)
maps = model.predict_anomaly_map(test_images)
```

## 非完整论文实现

- `paper-adaptation`: `core_deep_svdd` 与 `vision_deep_svdd` 已实现论文的
  one-class/soft-boundary 目标、无偏置最终线性编码器、中心初始化和半径评分，
  但仍是通用特征 MLP 而非论文的 MNIST/CIFAR-10 LeNet；`vision_devnet`
  已对齐 2021 图像论文的端到端 ResNet-18、1x1 patch 打分、两尺度
  top-10% MIL、偏差损失与训练默认值，
  但未暴露平滑输入梯度定位图；`vision_differnet` 已对齐论文检测网络与
  4/64 变换数，但未暴露论文的梯度定位路径；`vision_memae` 已对齐论文
  CIFAR-10 RGB 网络和记忆寻址，但工业图像接口仍是场景适配；
  `vision_draem` 的网络与训练计划已对齐但默认纹理合成仍为简化路径；
  `vision_fastflow` 已对齐论文的 ResNet18/WRN50-2 前三阶段、原生通道、
  8 步二维仿射流、论文卷积安排、二维似然和多尺度概率图，但论文未提供
  作者代码且未写明全部稳定化、概率图归一化、图像级聚合和旋转细节，
  本地离线默认也不下载 ImageNet 权重；
  `vision_fcdd` 已对齐 MVTec 的截断 VGG11-BN、pseudo-Huber/HSC 目标、
  confetti 参数、优化计划和感受野高斯热图，但离线默认不加载 ImageNet 权重，
  且在传入的正常样本上估计类别归一化范围。
- `core-aligned`: `vision_cflow` 已对齐作者 ResNet 路径的 layer2--layer4
  特征金字塔、二维位置条件、每尺度八个条件流块、归一化似然目标和多尺度概率图；
  论文指标仍要求 ImageNet 权重、类别输入尺寸与完整评估协议。
- `partial`: `vision_dfm`, `vision_softpatch`。
- `inspired`: `vision_ast`, `vision_promptad`, `vision_realnet`, `vision_inctrl`,
  `vision_glad`, `vision_oneformore`, `vision_panda`, `vision_regad`,
  `vision_riad`, `vision_winclip`。
- 外部路径：使用对应的 `vision_*_anomalib` 检查点封装；例如
  `vision_winclip_anomalib` 或 `vision_fastflow_anomalib`。

完整逐项清单见 [Model Index](../../MODEL_INDEX.md) 和
[Neural Model Fidelity](../../SOTA_ALGORITHMS.md)。
