# 可选依赖

=== "中文"

    pyimgano 采用模块化的可选依赖（extras）机制，按需安装功能组件，避免不必要的依赖开销。

=== "English"

    pyimgano uses a modular optional dependency (extras) mechanism, allowing on-demand installation of feature components to avoid unnecessary dependency overhead.

---

## 快速安装表

| Extra | 安装命令 | 用途 |
|-------|----------|------|
| `torch` | `pip install pyimgano[torch]` | PyTorch 深度学习模型 |
| `onnx` | `pip install pyimgano[onnx]` | ONNX 导出与推理 |
| `openvino` | `pip install pyimgano[openvino]` | OpenVINO 推理加速 |
| `skimage` | `pip install pyimgano[skimage]` | scikit-image 图像处理 |
| `numba` | `pip install pyimgano[numba]` | Numba JIT 加速 |
| `clip` | `pip install pyimgano[clip]` | CLIP 视觉-语言特征 |
| `faiss` | `pip install pyimgano[faiss]` | FAISS 向量检索加速 |
| `anomalib` | `pip install pyimgano[anomalib]` | Anomalib 模型集成 |
| `deploy` | `pip install pyimgano[deploy]` | 部署工具链（ONNX + 验证） |
| `benchmark` | `pip install pyimgano[benchmark]` | 基准测试依赖 |
| `dev` | `pip install pyimgano[dev]` | 开发依赖（测试、lint 等） |
| `all` | `pip install pyimgano[all]` | 安装全部可选依赖 |

=== "中文"

    可以组合安装多个 extras：

=== "English"

    Multiple extras can be installed together:

```bash
pip install pyimgano[torch,onnx,deploy]
```

---

## 环境诊断

### pyimgano-doctor

```bash
# 完整环境报告
pyimgano-doctor

# 推荐缺失的 extras
pyimgano-doctor --recommend-extras

# 断言指定 extra 已安装（CI/CD 中使用）
pyimgano-doctor --require-extras torch,onnx

# 检查加速器
pyimgano-doctor --accelerators
```

=== "中文"

    `pyimgano-doctor` 会扫描环境，报告已安装/缺失的依赖、可用的加速器，并给出安装建议。

=== "English"

    `pyimgano-doctor` scans the environment, reports installed/missing dependencies, available accelerators, and provides installation recommendations.

!!! tip "CI/CD 集成"

    在流水线中使用 `--require-extras` 确保必要依赖已安装：

    ```bash
    pyimgano-doctor --require-extras torch,deploy --json || exit 1
    ```

---

## 推荐组合

### 工业部署

```bash
pip install pyimgano[torch,deploy]
```

=== "中文"

    覆盖训练、导出和部署包验证的完整流程。

=== "English"

    Covers the complete workflow of training, export, and deploy bundle validation.

### 基准测试与评估

```bash
pip install pyimgano[torch,benchmark,skimage]
```

### 全功能研发

```bash
pip install pyimgano[all]
```

### 轻量推理（仅 ONNX）

```bash
pip install pyimgano[onnx]
```

=== "中文"

    不需要 PyTorch，仅使用 ONNX Runtime 进行推理，适合边缘设备和轻量环境。

=== "English"

    No PyTorch needed. Uses only ONNX Runtime for inference — suitable for edge devices and lightweight environments.

---

## 套件/模型依赖

=== "中文"

    不同模型和基准套件可能依赖不同的 extras。使用以下命令检查：

=== "English"

    Different models and benchmark suites may depend on different extras. Check with:

```bash
# 查看模型所需依赖
pyimgano-benchmark --model-info patchcore

# 查看套件所需依赖
pyimgano-benchmark --suite industrial_core --dry-run
```

!!! note "常见依赖关系"

    - 基于 embedding 的模型（PatchCore, PaDiM 等）→ `torch`
    - CLIP 类模型 → `torch`, `clip`
    - FAISS 加速的 k-NN → `faiss`
    - 像素级评估 → `skimage`

---

## 开发安装

```bash
# 克隆仓库
git clone https://github.com/your-org/pyimgano.git
cd pyimgano

# 安装开发依赖
pip install -e ".[dev]"

# 或安装全部依赖（开发 + 所有功能）
pip install -e ".[all,dev]"
```

=== "中文"

    `dev` extra 包含测试框架（pytest）、代码检查（ruff/mypy）和文档构建工具。

=== "English"

    The `dev` extra includes testing frameworks (pytest), linting tools (ruff/mypy), and documentation build tools.
