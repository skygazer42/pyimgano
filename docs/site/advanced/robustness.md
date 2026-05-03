---
title: 鲁棒性测试
---

# 鲁棒性测试

=== "中文"

    `pyimgano-robust-benchmark` 在受控干扰（corruptions）条件下评估模型的稳定性。
    它会对测试图像施加不同类型和强度的干扰，然后测量检测性能的退化程度。
    核心约束：全程使用**单一固定阈值**，模拟真实部署场景。

=== "English"

    `pyimgano-robust-benchmark` evaluates model stability under controlled corruptions.
    It applies various types and severities of corruptions to test images, then measures
    the degradation in detection performance.
    Core constraint: a **single fixed threshold** is used throughout, simulating real deployment.

---

## CLI 用法 / CLI Usage

```bash
# 完整鲁棒性评测
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --preset industrial-balanced \
  --pretrained \
  --device cuda \
  --pixel-normal-quantile 0.999 \
  --pixel-calibration-fraction 0.2 \
  --corruptions lighting,jpeg,blur,glare,geo_jitter \
  --severities 1 2 3 4 5 \
  --save-run \
  --output-dir /tmp/pyimgano_robust_run \
  --output runs/robust_result.json
```

```bash
# 快速冒烟测试
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --device cpu \
  --no-pretrained \
  --corruptions lighting \
  --severities 1 \
  --limit-train 32 \
  --limit-test 64
```

### 完整参数 / Full Flags

| 参数 / Flag | 说明 / Description |
|---|---|
| `--model` | 模型名称 |
| `--dataset` | 数据集类型（mvtec, visa, custom） |
| `--root` | 数据集路径 |
| `--category` | 数据集类别 |
| `--preset` | 模型预设（如 `industrial-balanced`） |
| `--corruptions` | 干扰类型列表（逗号分隔） |
| `--severities` | 严重度等级（1-5） |
| `--output-dir` | 结果输出目录 |
| `--output` | JSON 结果输出路径 |
| `--metric` | 评估指标 |
| `--threshold` | 固定阈值（所有干扰共用） |
| `--pixel-normal-quantile` | 像素阈值校准分位数（默认 0.999） |
| `--pixel-calibration-fraction` | 校准使用的正常像素比例 |
| `--latency` | 测量推理延迟 |
| `--limit-train` / `--limit-test` | 限制训练/测试样本数 |
| `--save-run` | 持久化运行产物 |
| `--plugins` | 启用插件模型 |
| `--no-pretrained` | 离线模式（默认） |
| `--pretrained` | 使用预训练权重 |
| `--no-pixel-segf1` | 禁用像素 SegF1（无分割掩码时） |
| `--input-mode` | `numpy`（默认）或 `paths` |

!!! warning "输入模式 / Input Mode"

    === "中文"

        干扰需要 `--input-mode numpy`（默认），向模型输入 RGB/u8/HWC numpy 图像。
        仅接受文件路径的经典模型请用 `--input-mode paths`，此模式跳过干扰评测。

    === "English"

        Corruptions require `--input-mode numpy` (default), feeding models RGB/u8/HWC numpy images.
        For classical baselines that only accept file paths, use `--input-mode paths` (corruptions are skipped).

---

## 干扰类型 / Corruption Types

| 干扰 / Corruption | 说明 / Description |
|---|---|
| `lighting` | 曝光/对比度/gamma 随机变化 + 轻微通道增益漂移 |
| `jpeg` | JPEG 编解码伪影（块效应/振铃） |
| `blur` | 高斯模糊 |
| `glare` | 高光反射/眩光模拟 |
| `geo_jitter` | 轻微仿射抖动（平移、旋转，图像与掩码一致变换） |

### 严重度等级 / Severity Levels

=== "中文"

    每种干扰有 5 个严重度等级，1 为最轻，5 为最重。
    等级定义确保跨干扰类型的可比性。每种干扰在给定 `--seed`、名称和严重度下是**确定性的**。

=== "English"

    Each corruption has 5 severity levels, where 1 is lightest and 5 is heaviest.
    Level definitions ensure comparability across corruption types. Each corruption is **deterministic** for a given `--seed`, name, and severity.

---

## 评估指标 / Metrics

### Pixel SegF1

=== "中文"

    像素级分割 F1 分数（SegF1），评估异常图的像素定位精度。
    像素阈值 `t` 从正常像素校准一次（使用分位数），全程固定。

=== "English"

    Pixel-level segmentation F1 score (SegF1), evaluating anomaly map pixel localization accuracy.
    Pixel threshold `t` is calibrated once from normal pixels (using a quantile) and kept fixed throughout.

### Background FPR

=== "中文"

    背景误报率（Background False Positive Rate），衡量正常区域被误判为异常的比例。

=== "English"

    Background False Positive Rate, measuring the proportion of normal regions
    misclassified as anomalous.

### 单一固定阈值约束 / Single Fixed Threshold

!!! warning "阈值一致性"

    === "中文"

        鲁棒性基准要求所有干扰条件使用同一个固定阈值（`--threshold`）。
        这确保评估的是模型在不同条件下的稳定性，而非针对每种干扰重新调参。
        `--pixel-normal-quantile` 控制灵敏度权衡：更高分位数 → 更少误报但可能遗漏小缺陷。

    === "English"

        The robustness benchmark requires a single fixed threshold (`--threshold`)
        across all corruption conditions. This ensures we evaluate model stability
        across conditions, not per-corruption tuning.
        `--pixel-normal-quantile` controls the sensitivity tradeoff: higher quantile → fewer false positives but may miss small defects.

---

## 延迟测量 / Latency Measurement

=== "中文"

    使用 `--latency` 标志启用推理延迟测量。报告包含每种干扰条件下的
    平均推理时间（ms）和吞吐量（images/s），以及延迟稳定性比率。

=== "English"

    Enable inference latency measurement with the `--latency` flag. The report includes
    average inference time (ms) and throughput (images/s) for each corruption condition,
    plus latency stability ratios.

---

## 输出格式 / Output Schema

```json
{
  "dataset": "mvtec",
  "category": "bottle",
  "model": "vision_patchcore",
  "robustness_summary": {
    "clean_auroc": 0.99,
    "mean_corruption_auroc": 0.94,
    "worst_corruption_auroc": 0.88,
    "mean_corruption_drop_auroc": 0.05,
    "worst_corruption_drop_auroc": 0.11,
    "clean_latency_ms_per_image": 12.3,
    "mean_corruption_latency_ms_per_image": 14.1,
    "worst_corruption_latency_ratio": 1.37
  },
  "robustness": {
    "pixel_threshold_strategy": "normal_pixel_quantile",
    "pixel_normal_quantile": 0.999,
    "clean": { "latency_ms_per_image": 12.3, "results": { "..." } },
    "corruptions": {
      "lighting": {
        "severity_1": { "..." },
        "severity_2": { "..." }
      }
    }
  }
}
```

=== "中文"

    `robustness_summary` 支持直接用于质控门禁：

    - **精度保持** — `mean_corruption_*`, `worst_corruption_*`
    - **退化幅度** — `mean_corruption_drop_*`, `worst_corruption_drop_*`
    - **延迟稳定性** — `*_latency_ms_per_image`, `*_latency_ratio`

=== "English"

    `robustness_summary` supports direct use in quality gates:

    - **Accuracy retention** — `mean_corruption_*`, `worst_corruption_*`
    - **Degradation magnitude** — `mean_corruption_drop_*`, `worst_corruption_drop_*`
    - **Latency stability** — `*_latency_ms_per_image`, `*_latency_ratio`

---

## 运行管理 / Run Management

### 保存运行 / Save Runs

=== "中文"

    `--save-run` 持久化运行产物，包括 `report.json`、`config.json`、`environment.json`，
    以及 `artifacts/robustness_conditions.csv` 和 `artifacts/robustness_summary.json`。
    CSV 包含 `drop_*` 列（相对于 clean 基线的退化），无需离线重算。

=== "English"

    `--save-run` persists run artifacts including `report.json`, `config.json`, `environment.json`,
    plus `artifacts/robustness_conditions.csv` and `artifacts/robustness_summary.json`.
    The CSV includes `drop_*` columns (degradation relative to clean baseline), no offline recomputation needed.

### 查询与对比 / Query and Compare

```bash
# 列出鲁棒性运行
pyimgano-runs list --root runs --kind robustness --json

# 查找协议兼容的运行
pyimgano-runs list --root runs --kind robustness \
  --same-robustness-protocol-as runs/robust_a --json

# 最新兼容运行
pyimgano-runs latest --root runs --kind robustness \
  --same-robustness-protocol-as runs/robust_a --json

# 对比两次运行
pyimgano-runs compare runs/robust_a runs/robust_b \
  --baseline runs/robust_a \
  --require-same-robustness-protocol --json
```

=== "中文"

    `--require-same-robustness-protocol` 确保对比的两次运行使用相同的干扰模式、干扰集、严重度、输入模式和 resize，否则快速失败。
    对比结果中 `robustness_protocol_comparison` 字段记录协议一致性检查详情。

=== "English"

    `--require-same-robustness-protocol` ensures both runs use the same corruption mode, corruption set, severities, input mode, and resize, or fails fast.
    The comparison result's `robustness_protocol_comparison` field records protocol consistency check details.

---

## Python API — CorruptionsDataset

=== "中文"

    `CorruptionsDataset` 提供 torch 风格的数据集接口，可在训练/评估流水线中
    直接应用确定性干扰。

=== "English"

    `CorruptionsDataset` provides a torch-style dataset interface for applying
    deterministic corruptions directly in training/evaluation pipelines.

```python
from pyimgano.datasets import CorruptionsDataset

dataset = CorruptionsDataset(
    base_dataset=my_dataset,
    corruptions=["lighting", "blur"],
    severity=3,
    seed=42,
)

for image, label in dataset:
    # image has corruption applied
    ...
```

---

## 合成式干扰辅助 / Synthesis-Style Corruption Helper

=== "中文"

    `apply_synthesis_preset` 将合成异常预设作为干扰使用，
    用于测试模型对未见过的缺陷类型的鲁棒性。

=== "English"

    `apply_synthesis_preset` uses synthesis anomaly presets as corruptions,
    testing model robustness to unseen defect types.

```python
from pyimgano.robustness import apply_synthesis_preset

corrupted = apply_synthesis_preset(image, preset="scratch", severity=3)
```

---

## 调优建议 / Tuning Tips

=== "中文"

    - `--pixel-normal-quantile` 控制灵敏度权衡：更高分位数 → 更少背景误报（低 `bg_fpr`），但可能遗漏小缺陷
    - 生产环境正常样本有噪声时，先用 `vision_softpatch` + `industrial-balanced` 预设评估
    - 无分割掩码的数据集请用 `--no-pixel-segf1`

=== "English"

    - `--pixel-normal-quantile` controls the sensitivity tradeoff: higher quantile → fewer background false positives (lower `bg_fpr`), but may miss small defects
    - For noisy production normals, evaluate with `vision_softpatch` and the `industrial-balanced` preset first
    - For datasets without segmentation masks, use `--no-pixel-segf1`
