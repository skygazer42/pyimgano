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
from pyimgano.models.registry import model_info

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
| `vision_padim` | 论文 R18/WR50-2 通道采样、逐位置高斯分布与马氏距离图 |
| `vision_stfpm` | 论文 ResNet-18 三层师生特征匹配和全分辨率乘积异常图 |
| `vision_cflow` | 三尺度条件归一化流与位置编码 |
| `vision_devnet` | 2021 图像网络、两尺度 top-K MIL 与偏差损失（无定位接口） |
| `vision_reverse_distillation` | WRN50-2 教师、OCBE 与反向 WRN 解码器 |
| `vision_simplenet` | 论文 3×3 补丁嵌入、特征适配器与噪声判别器 |
| `vision_spade` | WRN50-2 ImageNet-V1 图像检索与平方 L2 深层金字塔对应 |
| `vision_cutpaste` | ResNet-18 CutPaste 三分类自监督与高斯密度评分 |
| `vision_efficientad` | 论文 S/M PDN、双头学生、64 维 AE 与双异常图（需外部教师权重） |

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
  但仍是通用特征 MLP 而非论文的 MNIST/CIFAR-10 LeNet；`vision_dfm`
  已实现单层特征、4 倍平均池化、99.5% PCA、MLE 高斯与完整负对数似然，
  但正常类接口、224px 输入和离线权重默认值属于本项目适配；`vision_devnet`
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
  且在传入的正常样本上估计类别归一化范围；`vision_ast` 已对齐论文的
  MVTec AD RGB 路径，包括 EfficientNet-B5 第 36 层特征、32 维位置条件、
  四块 RealNVP teacher、四残差块 student、两阶段训练和 RGB 均值距离评分，
  但默认不下载 ImageNet 权重，且未实现论文的 3D/前景掩码路径；
  `vision_riad` 已对齐三组互补区域遮罩、论文 U-Net、L2/SSIM/MSGMS
  联合目标和四种区域尺寸的 MSGMS 集成，但作者未发布参考代码，因此仍标为适配；
  `vision_efficientad` / `efficient_ad` 已对齐补充材料的 S/M PDN、384/768
  通道师生路径、64 维瓶颈 AE、三项损失、70,000 步计划和分位数双图评分。
  论文未发布官方教师权重或作者仓库，因此严格模式要求显式提供
  `teacher_checkpoint` 与 `imagenet_dir`，不会用随机 ResNet 冒充。
- `core-aligned`: `vision_cflow` 已对齐作者 ResNet 路径的 layer2--layer4
  特征金字塔、二维位置条件、每尺度八个条件流块、归一化似然目标和多尺度概率图；
  论文指标仍要求 ImageNet 权重、类别输入尺寸与完整评估协议；`vision_softpatch`
  已对齐 WRN50-2 layer2/layer3 patch 特征、逐位置 LOF(k=6)、15% 去噪、
  10% greedy coreset、记忆权重乘最近邻距离与最大 patch 图像分数；
  `vision_panda` 已对齐图像级 PANDA-Early：ImageNet ResNet152、仅微调
  layer3/layer4、2,300 个 minibatch 的论文 SGD 参数及平方 L2 2-NN 评分。
  该入口不包含需要 Fisher 的 EWC、SES 多检查点或独立 SPADE 分割路径。
- `paper-adaptation`: `vision_realnet` 已对齐 WideResNet50-2 四层特征、
  64-batch AFS、四个独立重建 U-Net、max/mean RRS、联合重建/分割损失及
  1,000 epoch 默认值。训练必须显式提供离线 SDAS/SIA 异常图与掩码配对，
  且不复现作者用有标签验证集选择最佳 checkpoint 的评测流程。
- `paper-adaptation`: `vision_regad` 已对齐 ECCV 2022 的 ResNet-18 三段
  STN、卷积式 SimSiam 注册损失、50 epoch momentum SGD、支持集增强、
  逐位置高斯与 Mahalanobis 异常图。`fit` 必须提供源类别标签及目标正常
  支持集；`set_support` 可在不微调网络的情况下切换目标类别。论文数值仍需
  leave-one-category-out 与十轮支持集协议，本 API 不用带标签测试 AUC 选 checkpoint。
- `paper-adaptation`: `vision_promptad` 已对齐冻结的 LAION-400M
  ViT-B/16+ VV-CLIP、语义拼接、零间隔 EAM、MAP/LAP 分布对齐、双层正常
  视觉记忆及论文调和融合；图像级与像素级提示需按论文分别训练。
- `paper-adaptation`: `vision_winclip` / `winclip` 已对齐 ViT-B/16+ 完整
  组合提示集、2x2/3x3 masked-token 窗口、调和多尺度图、WinCLIP+ 三套
  视觉记忆，以及补充材料的 240px 预处理和非方形切片策略。
- `inspired`: `vision_inctrl`, `vision_glad`, `vision_oneformore`。
- 外部路径：使用对应的 `vision_*_anomalib` 检查点封装；例如
  `vision_winclip_anomalib` 或 `vision_fastflow_anomalib`。

完整逐项清单见 [Model Index](../../MODEL_INDEX.md) 和
[Neural Model Fidelity](../../SOTA_ALGORITHMS.md)。
