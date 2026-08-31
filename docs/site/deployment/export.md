# 训练产物导出与第三方导入

=== "中文"

    `pyimgano-export` 将**已拟合 detector** 从持久化 run 恢复并封装成可迁移、可校验的
    artifact。导出不会重新调用 `fit()`。artifact 包含运行时状态、输入/输出契约、
    artifact-local 推理策略、哈希和验证证据。

=== "English"

    `pyimgano-export` restores a **fitted detector** from a persisted run and packages a
    relocatable, verified artifact. Export never refits the detector. The artifact carries
    runtime state, tensor contracts, artifact-local policy, hashes, and verification evidence.

## 从已完成 run 导出

```bash
pyimgano-export \
    --from-run runs/<run_dir> \
    --format native \
    --out ./exports
```

重复 `--format` 可请求多个已认证格式：

```bash
pyimgano-export \
    --from-run runs/<run_dir> \
    --format native \
    --format onnx \
    --out ./exports
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--from-run` | 包含持久化训练 checkpoint 的完成 run | 必填 |
| `--format` | `native` / `onnx` / `torchscript` / `openvino`，可重复 | `native` |
| `--out` | artifact 输出根目录 | `<run>/artifacts/exported` |
| `--category` | 多 category run 的显式选择器 | 无 |
| `--verification-level` | `reference-parity` / `end-to-end` | `reference-parity` |
| `--non-strict` | 允许仅发布通过验证的格式 | false |
| `--trust-checkpoint` | 显式允许已校验但需要可执行反序列化的 checkpoint | false |
| `--overwrite` | 替换已存在的目标 | false |

默认事务是严格且原子的：所有请求格式均必须支持并通过验证，否则不发布最终目录。
schema v1 不支持“未验证”导出；`end-to-end` 只会加强强制 reference parity。

训练时也可直接请求相同的 canonical export service：

```bash
pyimgano-train \
    --config my_certified_config.json \
    --export-format native \
    --export-verification-level reference-parity
```

## 支持能力不是按格式猜测

每个 model 与 fitted state 的 `capabilities.trained_export` 由显式 adapter registry 决定：

```bash
pyimgano-benchmark --model-info <model_name> --json
```

若 capability cell 为 unsupported，CLI 会返回原因和 remediation；不会把
“能消费 ONNX”误报为“能导出已训练 ONNX”。

当前 schema-v1 认证矩阵按 model、格式和 layout 明确区分：

| Model | Native | ONNX | TorchScript | OpenVINO |
|---|---|---|---|---|
| `ae_resnet_unet` | supported / `native_detector` | conditional / `single_graph` | conditional / `single_graph` | conditional / `single_graph` |
| `vision_onnx_ecod` | unsupported | conditional / `composite` | unsupported | unsupported |
| `vision_torchscript_ecod` | unsupported | unsupported | conditional / `composite` | unsupported |
| `vision_patchcore` | unsupported | unsupported | unsupported | unsupported |

`conditional` cell 需要具体 fitted detector、完整已验证 checkpoint、声明的依赖，以及
（ECOD composite）认证时绑定的精确本地 embedding graph。`vision_onnx_ecod` 只保留 ONNX
源图及 external-data closure；`vision_torchscript_ecod` 只保留 TorchScript 源图。schema v1
不会在两者之间转换。两种 composite 均封装 non-executable fitted ECOD core、不会 refit，且
只提供 image score，不提供 anomaly map。自定义 extractor、变化后的 graph 或不完整 core
都会拒绝导出。上面的 `my_certified_config.json` 占位配置仍指全格式参考目标
`model.name=ae_resnet_unet`。

发布 E2E 认证矩阵仍是 Ubuntu x86_64、Python 3.10、CPU；ONNX 使用
`CPUExecutionProvider`。其它平台兼容性不由该 release gate 认证。

| 格式 | 执行后端 | Runtime extra | Export extra | 关键边界 |
|---|---|---|---|---|
| Native | `pyimgano` | 基础/模型依赖 | 基础/模型依赖 | 需要认证 state codec 或显式 trust checkpoint |
| ONNX | `onnxruntime` | `onnx-runtime` | `onnx-export` | raw ONNX 必须先提供语义契约 |
| TorchScript | `torchscript` | `torch` | `torch` | single/composite graph 加载均需显式 trust |
| OpenVINO | `openvino` | `openvino-runtime` | `openvino-export` | 使用 device；不接受 ONNX provider flags |

完整创建与执行环境可安装：

```bash
pip install "pyimgano[deploy]"
```

仅运行 ONNX/OpenVINO 的容器不需要 Torch：

```bash
pip install "pyimgano[onnx-runtime]"
pip install "pyimgano[openvino-runtime]"
```

`onnx` 与 `openvino` 在 0.10 发布线中保留为兼容别名。

## TorchScript 可执行信任边界

TorchScript 的 `single_graph` 与 `composite` artifact 都会调用可执行 graph loader，因此默认
拒绝加载。CLI 必须显式传入 `--trust-checkpoint`：

```bash
pyimgano-infer --artifact ./exports/bottle/torchscript \
    --trust-checkpoint --input ./test_images --save-jsonl ./results.jsonl
```

Python API 对应为 `load_artifact(path, trust_checkpoint=True)`。PyTorch 官方
[`torch.jit.load` 文档](https://docs.pytorch.org/docs/stable/generated/torch.jit.load.html)
警告恶意模型可能在反序列化时执行任意代码。只有独立确认来源与完整性后才能开启；hash
只能发现内容变化，不能把不可信 graph 变安全。

## 导入第三方 ONNX

raw `.onnx` 文件不是自描述 anomaly artifact。必须用 versioned contract 明确预处理与
score/map 语义：

```json
{
  "schema_family": "pyimgano-onnx-import",
  "schema_version": 1,
  "input": {
    "name": "input",
    "dtype": "float32",
    "layout": "NCHW",
    "color_space": "RGB",
    "size": [224, 224],
    "dynamic_batch": true,
    "dynamic_spatial": false,
    "resize": {"mode": "stretch", "interpolation": "bilinear"},
    "scale": {"divisor": 255.0},
    "normalize": {
      "mean": [0.0, 0.0, 0.0],
      "std": [1.0, 1.0, 1.0]
    }
  },
  "outputs": {
    "score": {
      "name": "score",
      "transform": "identity",
      "score_order": "higher_is_more_anomalous"
    }
  }
}
```

```bash
pyimgano-artifact import \
    --format onnx \
    --model ./model.onnx \
    --contract ./onnx-contract.json \
    --out ./imported-artifact

pyimgano-infer --artifact ./imported-artifact \
    --input ./test_images --save-jsonl ./results.jsonl
```

Importer 会校验 graph/contract、约束 external-data 路径，并在 fresh ONNX Runtime session
中执行 smoke test。第三方模型没有 PyImgAno reference implementation，因此其诚实验证等级是
`runtime_smoke`。未传 `--policy` 时得到 score-only policy；可用
`pyimgano-artifact bind-policy` 创建带已校验 operating policy 的新 immutable artifact。

## 旧 backbone exporter

`pyimgano-export-onnx` 与 `pyimgano-export-torchscript` 是兼容保留的
embedding/backbone exporter。它们不会封装 fitted detector，也不会生成
`artifact_manifest.json`。训练后部署请使用 `pyimgano-export`。
