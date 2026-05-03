# 模型导出

=== "中文"

    pyimgano 支持将训练好的模型导出为 ONNX、TorchScript 和 OpenVINO 格式，以便在不同运行时和硬件平台上进行推理。

=== "English"

    pyimgano supports exporting trained models to ONNX, TorchScript, and OpenVINO formats for inference across different runtimes and hardware platforms.

## ONNX 导出

```bash
pyimgano-export-onnx --train-dir runs/my_model --output embed.onnx
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train-dir` | 训练输出目录 | *必填* |
| `--output` | 输出文件路径 | `embed.onnx` |
| `--opset` | ONNX opset 版本 | `17` |
| `--dynamic-batch` | 启用动态 batch 维度 | `False` |
| `--no-pretrained` | 不下载预训练权重（离线安全） | `False` |
| `--simplify` | 运行 ONNX Simplifier | `False` |

!!! note "安装要求"

    需要安装 `onnx` extra：`pip install pyimgano[onnx]`

## TorchScript 导出

```bash
pyimgano-export-torchscript --train-dir runs/my_model --output embed.ts
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train-dir` | 训练输出目录 | *必填* |
| `--output` | 输出文件路径 | `embed.ts` |
| `--no-pretrained` | 不下载预训练权重（离线安全） | `False` |
| `--trace` | 使用 tracing 而非 scripting | `False` |

## OpenVINO 导出

=== "中文"

    OpenVINO 导出基于 Intel 的模型优化工具包（Model Optimizer），适用于 Intel CPU/GPU/VPU 推理加速。

=== "English"

    OpenVINO export uses Intel's Model Optimizer toolkit for accelerated inference on Intel CPU/GPU/VPU hardware.

```bash
# 先导出 ONNX，再转换为 OpenVINO IR
pyimgano-export-onnx --train-dir runs/my_model --output embed.onnx
mo --input_model embed.onnx --output_dir openvino_out/
```

!!! note "安装要求"

    需要安装 `openvino` extra：`pip install pyimgano[openvino]`

## 导出验证

=== "中文"

    导出后应验证模型输出的一致性。可使用推理命令对比原始模型和导出模型的结果。

=== "English"

    After export, verify output consistency by comparing inference results between the original and exported models.

```bash
# 使用导出的 ONNX 模型运行推理
pyimgano-infer --model onnx --train-dir runs/my_model --input test_images/
```

## 离线安全默认值

=== "中文"

    在受限网络环境中，使用 `--no-pretrained` 标志避免运行时下载预训练权重。建议在 CI/CD 或生产打包流程中始终使用此选项。

=== "English"

    In restricted network environments, use the `--no-pretrained` flag to avoid runtime downloads of pretrained weights. Recommended for CI/CD and production packaging workflows.

```bash
pyimgano-export-onnx --train-dir runs/my_model --output embed.onnx --no-pretrained
pyimgano-export-torchscript --train-dir runs/my_model --output embed.ts --no-pretrained
```

!!! warning "重要"

    如果模型依赖预训练特征提取器，请确保权重文件已在训练阶段缓存到本地，或通过 `pyimgano-weights` 命令预先下载。
