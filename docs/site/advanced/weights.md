---
title: 权重管理
---

# 权重管理

=== "中文"

    pyimgano 提供完整的模型权重生命周期管理工具，包括清单验证、哈希校验、
    模型卡片验证和 bundle 审计。默认离线安全，不会隐式下载权重。

=== "English"

    pyimgano provides complete model weight lifecycle management tools, including
    manifest validation, hash verification, model card validation, and bundle auditing.
    Offline-safe by default with no implicit weight downloads.

---

## pyimgano-weights CLI

```bash
# 查看可用子命令
pyimgano-weights --help
```

### validate — 验证权重清单

```bash
pyimgano-weights validate ./weights_manifest.json

# 检查文件存在性和哈希完整性
pyimgano-weights validate ./weights_manifest.json --check-files --check-hashes --json
```

=== "中文"

    验证清单 JSON 格式、字段完整性和引用文件的存在性。`--check-hashes` 同时校验 SHA-256 哈希。

=== "English"

    Validates manifest JSON format, field completeness, and referenced file existence. `--check-hashes` also verifies SHA-256 hashes.

### template — 生成清单模板

```bash
pyimgano-weights template manifest > weights_manifest.json
```

### hash — 计算权重哈希

```bash
pyimgano-weights hash ./model.onnx
# 输出: sha256:abcdef1234567890...
```

=== "中文"

    计算 SHA-256 哈希，用于填入权重清单的 `hash` / `sha256` 字段。

=== "English"

    Computes SHA-256 hash for the `hash` / `sha256` field in the weights manifest.

### validate-model-card — 模型卡片验证

```bash
# 基本验证
pyimgano-weights validate-model-card ./model_card.json --json

# 验证卡片 + 引用的检查点文件
pyimgano-weights validate-model-card ./model_card.json \
  --check-files --check-hashes --json

# 交叉验证: 模型卡片 + 权重清单
pyimgano-weights validate-model-card ./model_card.json \
  --manifest ./weights_manifest.json \
  --check-files --check-hashes --json
```

=== "中文"

    - `weights.path` 默认相对于模型卡片文件解析
    - 使用 `--base-dir DIR` 可指定其他解析根目录
    - `--manifest` 交叉检查模型卡片与清单条目的一致性

=== "English"

    - `weights.path` is resolved relative to the model card file by default
    - Use `--base-dir DIR` to resolve relative paths against a different root
    - `--manifest` cross-checks the model card against a manifest entry

### audit-bundle — Bundle 审计

```bash
pyimgano-weights audit-bundle ./deploy_bundle --check-hashes --json
```

=== "中文"

    审计 deploy bundle 的完整性：权重文件哈希、推理配置一致性、必要文件存在性。
    如果 bundle 中包含 `model_card.json` 或 `weights_manifest.json`，审计会一并检查。

=== "English"

    Audits deploy bundle integrity: weight file hashes, inference config consistency,
    and required file existence. If `model_card.json` or `weights_manifest.json` exist in the bundle, they are validated too.

---

## 权重清单格式 / Weights Manifest Format

```json title="weights_manifest.json"
{
  "schema_version": 1,
  "entries": [
    {
      "name": "patchcore_wrn50_imagenet1k",
      "path": "checkpoints/patchcore_wrn50.pt",
      "sha256": "abcdef1234567890...",
      "license": "MIT",
      "source": "internal registry or upstream training run",
      "models": ["vision_patchcore"],
      "runtime": "torch"
    }
  ]
}
```

| 字段 / Field | 类型 / Type | 说明 / Description |
|---|---|---|
| `schema_version` | int | 清单格式版本 |
| `entries[].name` | string | 权重条目名称 |
| `entries[].path` | string | 相对路径 |
| `entries[].sha256` | string | SHA-256 哈希 |
| `entries[].license` | string | 许可证 / 再分发边界 |
| `entries[].source` | string | 来源（上游训练运行或内部注册表） |
| `entries[].models` | array | 关联的模型注册名 |
| `entries[].runtime` | string | 运行时路径（`torch`, `onnx`, `openvino`） |

---

## Deploy Bundle 清单 / Deploy Bundle Manifest

=== "中文"

    导出部署包时，pyimgano 在 `deploy_bundle/bundle_manifest.json` 中记录 bundle 的精确内容。
    这与 `weights_manifest.json` 是分开的两个概念：

    - `weights_manifest.json` — 跨环境的可复用检查点清单
    - `bundle_manifest.json` — 单个 deploy bundle 中的精确文件记录

=== "English"

    When exporting a deployment bundle, pyimgano records the exact bundle contents in `deploy_bundle/bundle_manifest.json`.
    This is separate from `weights_manifest.json`:

    - `weights_manifest.json` — Reusable checkpoint inventory across environments
    - `bundle_manifest.json` — Exact file record within one deploy bundle

```bash
# 导出带有 bundle manifest 的部署包
pyimgano-train --config ./train.json --export-deploy-bundle
```

=== "中文"

    bundle manifest 记录：相对目标路径、文件大小、sha256 摘要、源运行元数据和环境指纹，
    以及核心文件的显式角色（`infer_config.json`、`calibration_card.json`、`model_card.json`、`weights_manifest.json`）。

=== "English"

    The bundle manifest records: relative destination paths, file sizes, sha256 digests, source run metadata and environment fingerprint,
    plus explicit roles for core files (`infer_config.json`, `calibration_card.json`, `model_card.json`, `weights_manifest.json`).

---

## 哈希验证 / Hash Verification

=== "中文"

    部署前务必验证权重文件哈希，确保传输或存储过程中未发生损坏。

=== "English"

    Always verify weight file hashes before deployment to ensure no corruption
    during transfer or storage.

```bash
# 验证单个文件
pyimgano-weights hash ./model.onnx

# 验证整个清单 (自动校验所有文件哈希)
pyimgano-weights validate ./weights_manifest.json --check-files --check-hashes
```

---

## 离线安全原则 / Offline-Safe by Default

!!! info "v0.8.0 离线安全策略"

    === "中文"

        自 v0.8.0 起，所有 CLI 和模型默认不下载预训练权重：

        - CLI 默认 `--no-pretrained`
        - `torchvision_backbone` 默认 `pretrained=False`
        - `torchscript_embed` 完全离线（用户提供检查点）

        需要预训练权重时，须显式传入 `--pretrained` 或 `pretrained=True`。

    === "English"

        Since v0.8.0, all CLIs and models default to no pretrained weight downloads:

        - CLI defaults to `--no-pretrained`
        - `torchvision_backbone` defaults to `pretrained=False`
        - `torchscript_embed` is fully offline (user provides checkpoint)

        When pretrained weights are needed, explicitly pass `--pretrained` or `pretrained=True`.

---

## 推荐模式 / Recommended Patterns

### 1. 显式传递检查点路径 / Pass Checkpoint Paths Explicitly

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /datasets/mvtec_ad \
  --category bottle \
  --model vision_patchcore_anomalib \
  --checkpoint-path /checkpoints/patchcore.ckpt \
  --device cuda
```

```python
from pyimgano.models import create_model

det = create_model(
    "vision_anomalib_checkpoint",
    checkpoint_path="/checkpoints/model.ckpt",
    device="cuda",
)
```

### 2. 控制缓存目录 / Control Cache Directories

=== "中文"

    某些模型/骨干网络通过上游库下载权重（PyTorch、HuggingFace、OpenCLIP）。
    在生产环境中应控制缓存位置：

=== "English"

    Some models/backbones download weights via upstream libraries (PyTorch, HuggingFace, OpenCLIP).
    Control cache locations in production:

```bash
export TORCH_HOME=/mnt/models/torch_cache
export HF_HOME=/mnt/models/hf_cache
```

### 3. 版本化存储 / Versioned Storage

=== "中文"

    1. **版本化存储** — 将权重文件存入带版本标签的对象存储
    2. **清单随 bundle 分发** — 每个 bundle 包含完整清单
    3. **部署前验证** — 用 `audit-bundle` 验证 bundle 完整性
    4. **哈希固定** — 在 CI/CD 中自动验证哈希一致性
    5. **回滚策略** — 保留至少 2 个历史版本的权重

=== "English"

    1. **Versioned storage** — Store weight files in versioned object storage
    2. **Manifest with bundle** — Each bundle includes a complete manifest
    3. **Pre-deploy validation** — Use `audit-bundle` to verify bundle integrity
    4. **Hash pinning** — Automatically verify hash consistency in CI/CD
    5. **Rollback strategy** — Retain at least 2 historical weight versions

---

## 反模式 / Anti-Patterns

!!! danger "避免以下做法"

    === "中文"

        - 不要将多 GB 检查点烘焙到 Docker 镜像中（除非有充分理由）
        - 不要将权重提交到 git 仓库
        - 不要通过在 pyimgano wheel 中内嵌上游检查点来增大包体积

    === "English"

        - Don't bake multi-GB checkpoints into Docker images unless you have a strong reason
        - Don't commit weights into the git repo
        - Don't expand pyimgano wheel size by vendoring upstream checkpoints
