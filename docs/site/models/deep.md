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

- `paper-adaptation`: `vision_differnet` 已对齐论文检测网络与 4/64 变换数，
  但未暴露论文的梯度定位路径；`vision_memae` 已对齐论文 CIFAR-10 RGB
  网络和记忆寻址，但工业图像接口仍是场景适配；`vision_draem` 的网络与
  训练计划已对齐但默认纹理合成仍为简化路径。
- `partial`: `vision_fastflow`, `vision_cflow`, `vision_dfm`, `vision_fcdd`,
  `vision_softpatch`。
- `inspired`: `vision_ast`, `vision_promptad`, `vision_realnet`, `vision_inctrl`,
  `vision_glad`, `vision_oneformore`, `vision_panda`, `vision_regad`,
  `vision_riad`, `vision_winclip`。
- 外部路径：使用对应的 `vision_*_anomalib` 检查点封装；例如
  `vision_winclip_anomalib` 或 `vision_fastflow_anomalib`。

完整逐项清单见 [Model Index](../../MODEL_INDEX.md) 和
[Neural Model Fidelity](../../SOTA_ALGORITHMS.md)。
