---
title: FAQ
---

# FAQ / 常见问题

---

## 快速分诊 / Quick Triage

| 症状 / Symptom | 首选命令 / First Command | 可能原因 / Likely Cause | 参考文档 / Doc |
|---|---|---|---|
| `ModuleNotFoundError` | `pip install pyimgano[full]` | 缺少可选依赖 | [安装指南](getting-started/installation.md) |
| GPU 不可用 | `python -c "import torch; print(torch.cuda.is_available())"` | CUDA/PyTorch 版本不匹配 | [安装指南](getting-started/installation.md) |
| 清单校验失败 | `pyimgano-train preflight --config ...` | 清单格式错误 | [配置参考](reference/configuration.md) |
| 推理误报过多 | `pyimgano-infer --defects ...` | 阈值过低或预处理不一致 | [校准与阈值](guide/calibration.md) |
| Bundle 验证失败 | `pyimgano-weights audit-bundle --bundle ...` | 权重哈希不匹配 | [权重管理](advanced/weights.md) |
| 拼接接缝伪影 | 检查 `--tile-overlap` | tiling overlap 不足 | [推理指南](guide/inference.md) |

---

## 环境问题 / Environment FAQ

### 缺少可选依赖 / Missing Extras

=== "中文"

    pyimgano 的可选功能按 extras 分组。安装对应 extras 即可：

=== "English"

    Optional features are grouped by extras. Install the corresponding extras:

```bash
# 完整安装
pip install pyimgano[full]

# 仅深度学习功能
pip install pyimgano[torch]

# 仅合成功能
pip install pyimgano[synthesis]
```

### GPU 未检测到 / GPU Not Detected

=== "中文"

    1. 确认 PyTorch 安装了 CUDA 版本：`python -c "import torch; print(torch.version.cuda)"`
    2. 确认 NVIDIA 驱动：`nvidia-smi`
    3. pyimgano 的经典模型和像素基线不需要 GPU

=== "English"

    1. Verify PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
    2. Verify NVIDIA driver: `nvidia-smi`
    3. Classical models and pixel baselines in pyimgano do not require GPU

### 何时修复环境 vs 配置 / When to Fix Env vs Config

=== "中文"

    - **ImportError / ModuleNotFoundError** → 环境问题，安装缺少的包
    - **推理结果异常但无报错** → 配置问题，检查预处理参数和模型参数
    - **CUDA out of memory** → 配置问题，减小 `batch_size` 或 `image_size`

=== "English"

    - **ImportError / ModuleNotFoundError** → Environment issue, install missing packages
    - **Abnormal inference results without errors** → Config issue, check preprocessing and model params
    - **CUDA out of memory** → Config issue, reduce `batch_size` or `image_size`

---

## 数据问题 / Data FAQ

### 清单格式 / Manifest Format

=== "中文"

    pyimgano 使用 JSONL（每行一条 JSON 记录）作为数据清单格式：

=== "English"

    pyimgano uses JSONL (one JSON record per line) as the data manifest format:

```jsonl
{"image_path": "normal/001.png", "label": "normal"}
{"image_path": "normal/002.png", "label": "normal"}
{"image_path": "anomaly/001.png", "label": "anomaly", "mask_path": "masks/001.png"}
```

### 自定义数据集布局 / Custom Dataset Layout

=== "中文"

    推荐的目录结构：

=== "English"

    Recommended directory structure:

```
my_dataset/
├── manifest.jsonl
├── train/
│   └── normal/
│       ├── 001.png
│       └── ...
├── test/
│   ├── normal/
│   └── anomaly/
└── masks/        # 可选：异常掩码
    ├── 001.png
    └── ...
```

### 掩码覆盖率 / Mask Coverage

=== "中文"

    - 掩码应为单通道二值图（0=正常, 255=异常）
    - 分辨率须与原图一致
    - 像素级指标（SegF1）需要掩码标注

=== "English"

    - Masks should be single-channel binary images (0=normal, 255=anomaly)
    - Resolution must match the original image
    - Pixel-level metrics (SegF1) require mask annotations

---

## 推理问题 / Inference FAQ

### 误报过多 / Too Many False Positives

=== "中文"

    1. **检查阈值**：使用 `pyimgano-infer --defects` 查看缺陷裁剪图，确认阈值是否合理
    2. **检查预处理**：确保训练和推理使用相同的预处理流水线
    3. **使用校准**：参考[校准与阈值](guide/calibration.md)进行阈值校准
    4. **增加训练数据**：正常样本越多，模型的正常分布估计越准确

=== "English"

    1. **Check threshold**: Use `pyimgano-infer --defects` to review defect crops
    2. **Check preprocessing**: Ensure training and inference use the same preprocessing pipeline
    3. **Use calibration**: See [Calibration](guide/calibration.md) for threshold calibration
    4. **Add training data**: More normal samples improve the normal distribution estimate

### 拼接接缝伪影 / Tiling Seams

=== "中文"

    使用 tiling 推理时可能出现接缝伪影。增大 `--tile-overlap` 可以缓解此问题。

=== "English"

    Tiling inference may produce seam artifacts. Increase `--tile-overlap` to mitigate.

### 格式不匹配 / Format Mismatch

=== "中文"

    确保输入图像的色彩空间（RGB/BGR）、数据类型（uint8/float32）和通道顺序
    与模型训练时一致。推荐统一使用 RGB uint8 格式。

=== "English"

    Ensure the input image color space (RGB/BGR), data type (uint8/float32), and channel order
    match the training configuration. RGB uint8 is recommended as the standard format.

---

## 部署问题 / Deployment FAQ

### infer_config vs training config

=== "中文"

    - **training config**：训练时的完整配置，包含数据集路径、训练参数等
    - **infer_config**：bundle 内的推理配置，仅包含推理所需的预处理参数和阈值
    - 二者由导出过程自动生成，通常无需手动修改 infer_config

=== "English"

    - **training config**: Complete config for training, including dataset paths and training params
    - **infer_config**: Inference config inside the bundle, only preprocessing params and thresholds
    - Both are auto-generated during export; typically no manual editing of infer_config is needed

### Bundle 验证 / Bundle Validation

```bash
pyimgano-weights audit-bundle --bundle ./output/bundle/
```

=== "中文"

    部署前始终运行 bundle 审计，验证文件完整性和配置一致性。

=== "English"

    Always run bundle audit before deployment to verify file integrity and config consistency.

---

## 性能问题 / Performance FAQ

### 选择什么算法 / Which Algorithm

=== "中文"

    | 场景 / Scenario | 推荐模型 / Recommended |
    |---|---|
    | 快速原型 | 像素级基线（`ssim_template_map` 等） |
    | 通用工业检测 | `vision_resnet18_ecod` |
    | 无 GPU 环境 | 经典模型（`vision_hbos`, `vision_iforest`） |
    | 高精度要求 | `vision_embedding_core` + 调优 |
    | 缺陷定位 | 像素级模型（`*_map`） |

=== "English"

    | Scenario | Recommended |
    |---|---|
    | Quick prototype | Pixel baselines (`ssim_template_map`, etc.) |
    | General industrial inspection | `vision_resnet18_ecod` |
    | No GPU | Classical models (`vision_hbos`, `vision_iforest`) |
    | High accuracy | `vision_embedding_core` + tuning |
    | Defect localization | Pixel-map models (`*_map`) |

### 需要多少训练数据 / How Much Training Data

=== "中文"

    - 像素级基线：1 张参考图（模板模型）或 20+ 张（统计模型）
    - 经典模型：50-200 张正常图像通常足够
    - 嵌入+核心：100+ 张正常图像推荐
    - 更多数据通常会提升性能，但存在收益递减

=== "English"

    - Pixel baselines: 1 reference (template) or 20+ images (statistical)
    - Classical models: 50-200 normal images usually sufficient
    - Embedding + Core: 100+ normal images recommended
    - More data generally improves performance with diminishing returns

### GPU 使用 / GPU Usage

=== "中文"

    仅深度学习模型（`torchvision_backbone`, 重建模型）使用 GPU。
    经典模型和像素基线在 CPU 上运行，适合边缘部署。

=== "English"

    Only deep learning models (`torchvision_backbone`, reconstruction models) use GPU.
    Classical models and pixel baselines run on CPU, suitable for edge deployment.

---

## 常见错误 / Common Errors

### `RuntimeError: No model named 'xxx' in registry`

=== "中文"

    模型名拼写错误，或插件未加载。使用 `pyimgano-benchmark --list-models` 查看可用模型。
    插件模型需要 `--plugins` 标志。

=== "English"

    Model name typo or plugin not loaded. Use `pyimgano-benchmark --list-models` to list
    available models. Plugin models require the `--plugins` flag.

### `ValueError: Image size mismatch`

=== "中文"

    输入图像分辨率与模型期望不匹配。检查配置中的 `image_size` 参数。

=== "English"

    Input image resolution doesn't match model expectations. Check the `image_size` in config.

### `FileNotFoundError: weights/...`

=== "中文"

    权重文件缺失。确认 bundle 完整性或重新导出。使用 `pyimgano-weights audit-bundle` 检查。

=== "English"

    Weight file missing. Verify bundle integrity or re-export.
    Use `pyimgano-weights audit-bundle` to check.

---

## 获取帮助 / Getting Help

=== "中文"

    - **文档**：你正在阅读的这份文档
    - **GitHub Issues**：[github.com/skygazer42/pyimgano/issues](https://github.com/skygazer42/pyimgano/issues) — 报告 bug 和提出功能请求
    - **CHANGELOG**：[CHANGELOG.md](https://github.com/skygazer42/pyimgano/blob/main/CHANGELOG.md) — 查看版本变更

=== "English"

    - **Documentation**: The docs you are reading now
    - **GitHub Issues**: [github.com/skygazer42/pyimgano/issues](https://github.com/skygazer42/pyimgano/issues) — report bugs and feature requests
    - **CHANGELOG**: [CHANGELOG.md](https://github.com/skygazer42/pyimgano/blob/main/CHANGELOG.md) — view version changes
