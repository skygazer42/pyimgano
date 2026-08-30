---
title: 安装指南
---

# 安装指南

## 基础安装

```bash
pip install pyimgano
```

!!! info "Python 版本"

    pyimgano 需要 **Python 3.9+**。推荐使用 Python 3.10 或 3.11。

=== "中文"

    基础安装即可使用经典模型、CPU 基线与全部 CLI 命令。深度学习模型、ONNX 导出等功能需要安装对应的可选依赖。

=== "English"

    The base install provides classical models, CPU baselines, and all CLI commands. Deep learning models, ONNX export, and other advanced features require optional extras.

---

## 可选依赖 (Extras)

| Extra | 安装命令 | 功能 |
|-------|---------|------|
| `torch` | `pip install "pyimgano[torch]"` | 深度学习模型 / torchvision 骨干网络 |
| `onnx` | `pip install "pyimgano[onnx]"` | ONNX Runtime 推理 / ONNX 导出 |
| `openvino` | `pip install "pyimgano[openvino]"` | OpenVINO 推理 |
| `skimage` | `pip install "pyimgano[skimage]"` | SSIM / 相位相关 / scikit-image 基线 |
| `numba` | `pip install "pyimgano[numba]"` | Numba 加速基线 |
| `clip` | `pip install "pyimgano[clip]"` | OpenCLIP 后端 |
| `faiss` | `pip install "pyimgano[faiss]"` | 快速 kNN (memory-bank 方法) |
| `anomalib` | `pip install "pyimgano[anomalib]"` | anomalib 检查点封装 |
| `deploy` | `pip install "pyimgano[deploy]"` | 部署运行时 (ONNX + OpenVINO) |
| `benchmark` | `pip install "pyimgano[benchmark]"` | 基准测试全量依赖 |
| `all` | `pip install "pyimgano[all]"` | 全部依赖 (开发 / 文档 / 可视化 + 所有后端) |

!!! tip "组合安装"

    可同时安装多个 extras：`pip install "pyimgano[torch,onnx,skimage]"`

---

## GPU 环境配置

=== "Linux"

    ```bash
    # CUDA 12.x (推荐)
    pip install "pyimgano[torch]"
    # torch 默认会安装支持 CUDA 的版本

    # 验证 GPU
    python -c "import torch; print(torch.cuda.is_available())"
    ```

=== "macOS"

    ```bash
    # Apple Silicon (MPS)
    pip install "pyimgano[torch]"

    # 验证 MPS
    python -c "import torch; print(torch.backends.mps.is_available())"
    ```

=== "Windows"

    ```bash
    # CUDA 12.x
    pip install "pyimgano[torch]"

    # 验证 GPU
    python -c "import torch; print(torch.cuda.is_available())"
    ```

    !!! warning "Windows 注意事项"

        部分 extras（如 `faiss`）在 Windows 上可能需要额外配置。建议使用 WSL2 以获得最佳体验。

---

## 从源码安装

```bash
git clone https://github.com/skygazer42/pyimgano.git
cd pyimgano
pip install -e ".[dev]"
```

=== "中文"

    源码安装适用于贡献者或需要最新开发版本的用户。`[dev]` 包含测试、lint 和文档构建依赖。

=== "English"

    Source install is for contributors or users who need the latest development version. `[dev]` includes test, lint, and documentation build dependencies.

---

## 验证安装

```python
import pyimgano
print(f"pyimgano version: {pyimgano.__version__}")
```

```bash
# 环境检查 (推荐)
pyimgano-doctor
pyimgano-doctor --json

# 检查特定 extras
pyimgano-doctor --require-extras torch,skimage --json

# 检查加速器
pyimgano-doctor --accelerators --json
```

!!! success "验证通过"

    如果 `pyimgano-doctor` 显示绿色状态，说明安装成功。

---

## 推荐安装组合

| 场景 | 推荐安装 |
|------|---------|
| CPU 模板检测基线 | `pip install pyimgano`（+ `[skimage]` 用于 SSIM/相位相关） |
| GPU 异常热图 (PatchCore / SoftPatch / DINO) | `pip install "pyimgano[torch]"` |
| 部署运行时 (ONNX) | `pip install "pyimgano[onnx]"` |
| 部署运行时 (OpenVINO) | `pip install "pyimgano[openvino]"` |
| 语义驱动基线 | `pip install "pyimgano[clip]"` |
| 完整基准测试 | `pip install "pyimgano[benchmark]"` |

---

## 下一步

安装完成后，前往 [5 分钟体验](quickstart.md) 运行你的第一次异常检测。
