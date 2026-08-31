# 部署包 (Deploy Bundle)

=== "中文"

    部署包（Deploy Bundle）是 pyimgano 的标准化部署产物，将模型、配置和校准信息打包为一个可验证、可审计的单元。

=== "English"

    A Deploy Bundle is pyimgano's standardized deployment artifact, packaging model, configuration, and calibration information into a verifiable, auditable unit.

## Bundle 结构

```
my_bundle/
├── bundle_manifest.json      # 包清单：版本、哈希、文件列表
├── infer_config.json          # 兼容旧 reader 的推理配置镜像
├── calibration_card.json      # 校准卡：阈值来源、数据集统计
├── handoff_report.json        # 交付报告：训练指标、验证结果
└── artifacts/
    ├── export_index.json
    └── <category>/<format>/
        ├── artifact_manifest.json
        ├── infer_config.json  # artifact-local authoritative policy
        ├── model/ or state/
        └── verification/
```

| 文件 | 用途 |
|------|------|
| `bundle_manifest.json` | 清单、完整性校验和 runtime `artifact_refs` 索引 |
| `infer_config.json` | 旧 reader 的兼容镜像；vNext runtime 使用 artifact-local policy |
| `calibration_card.json` | 阈值校准的来源和统计信息 |
| `handoff_report.json` | 训练过程的质量指标和验证结果 |

## 创建 Bundle

=== "中文"

    在训练时通过 `--export-deploy-bundle` 标志一次性生成完整的部署包。

=== "English"

    Generate a complete deploy bundle during training with the `--export-deploy-bundle` flag.

```bash
pyimgano-train --config my_config.json \
    --export-format native \
    --export-infer-config \
    --export-deploy-bundle
```

当前可执行的全格式参考配置必须使用 `model.name=ae_resnet_unet`。不要给现有
`vision_patchcore` starter 追加 `--export-format`：其 trained-export cells 当前为 unsupported。

!!! tip "推荐工作流"

    `--export-format` 先调用 canonical fitted-detector exporter；bundle assembly 随后复制完整
    artifact root，并将每个 manifest 写入 `bundle_manifest.json.artifact_refs`。格式可重复，
    但必须由该 model 的 export adapter 明确支持。

## 验证 Bundle

### 结构与配置验证

```bash
# 验证 bundle 完整性和配置一致性
pyimgano bundle validate my_bundle/
```

### 推理配置验证

```bash
# 单独验证 infer_config.json
pyimgano-validate-infer-config my_bundle/infer_config.json
```

### 权重哈希审计

```bash
# 验证模型文件哈希是否与清单一致
pyimgano weights audit-bundle my_bundle/
```

!!! warning "CI/CD 集成"

    建议在部署流水线中将 `bundle validate` 和 `weights audit-bundle` 作为必要的门控步骤。

## 运行 Bundle

```bash
pyimgano bundle run my_bundle/ \
    --image-dir ./test_images/ \
    --max-anomaly-rate 0.05 \
    --max-reject-rate 0.02 \
    --max-error-rate 0.01 \
    --min-processed 100
```

若 bundle 含多个 artifact，必须加 selector，不能静默选择：

```bash
pyimgano-bundle run my_bundle/ \
    --artifact-category bottle \
    --artifact-format onnx \
    --artifact-backend onnxruntime \
    --onnx-providers CPUExecutionProvider \
    --image-dir ./test_images/ \
    --output-dir ./bundle_run
```

带 `artifact_refs` 的 bundle 总是加载选中 artifact 的 authoritative policy。没有
runtime artifact references 的旧 bundle 继续使用根目录 `infer_config.json` fallback。
若选择的 artifact 含 TorchScript graph（single 或 composite），`run` / `watch` 还必须在
完成 provenance review 后传入 `--trust-checkpoint`；默认不会执行该 graph。

### 运行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image-dir` | 待检测图片目录 | *必填* |
| `--max-anomaly-rate` | 最大异常率阈值 | `None` |
| `--max-reject-rate` | 最大拒绝率阈值 | `None` |
| `--max-error-rate` | 最大错误率阈值 | `None` |
| `--min-processed` | 最少处理图片数 | `None` |

### 输出文件

运行完成后生成两个输出文件：

**`results.jsonl`** — 逐图推理结果（JSON Lines 格式）：

```json
{"image": "img_001.png", "score": 0.12, "label": "normal", "anomaly_map": "..."}
{"image": "img_002.png", "score": 0.87, "label": "anomaly", "anomaly_map": "..."}
```

**`run_report.json`** — 运行汇总报告：

```json
{
  "batch_verdict": "PASS",
  "batch_gate_summary": {
    "anomaly_rate": 0.03,
    "reject_rate": 0.01,
    "error_rate": 0.00,
    "total_processed": 200,
    "gates_passed": true
  }
}
```

=== "中文"

    `batch_verdict` 取值为 `PASS` 或 `FAIL`，由 batch gate 参数（`--max-anomaly-rate` 等）决定。

=== "English"

    `batch_verdict` is `PASS` or `FAIL`, determined by batch gate parameters (`--max-anomaly-rate`, etc.).

## pyimgano-bundle CLI 概览

```bash
# 查看帮助
pyimgano bundle --help

# 验证
pyimgano bundle validate <bundle_dir>

# 一次性运行
pyimgano bundle run <bundle_dir> --image-dir <path> [gate flags]

# 热文件夹长驻运行（v0.9.0+）
pyimgano-bundle watch <bundle_dir> --watch-dir <inbox> --output-dir <out> [--once]
```

---

## 热文件夹运行时 (Hot-folder Watch)

=== "中文"

    `pyimgano-bundle watch` 在 v0.9.0 引入，将已验证的部署包变成长驻服务：轮询 `--watch-dir`、
    等待文件 `--settle-seconds` 稳定后写入聚合产物，无需重新训练或导出即可处理增量数据。
    适用于产线相机投递、夜班数据补检、与 PLC/MES 系统的轻量集成。

=== "English"

    `pyimgano-bundle watch` (v0.9.0+) turns a validated deploy bundle into a long-running
    service. It polls `--watch-dir`, waits `--settle-seconds` for files to stabilize, and
    appends aggregate artifacts without rerunning training or export — ideal for production
    camera ingest, off-hour reprocessing, and lightweight PLC/MES integration.

### 基本用法

```bash
# 一次性消化当前 backlog 后退出
pyimgano-bundle watch ./deploy_bundle \
    --watch-dir ./inbox \
    --output-dir ./bundle_watch \
    --once --json

# 长驻轮询（生产模式）
pyimgano-bundle watch ./deploy_bundle \
    --watch-dir ./inbox \
    --output-dir ./bundle_watch \
    --poll-seconds 1.0 \
    --settle-seconds 2.0
```

### 输出产物

| 文件 | 说明 |
|------|------|
| `results.jsonl` | 增量逐图推理结果 |
| `watch_report.json` | 滚动汇总，包含投递摘要、批次裁决 |
| `watch_state.json` | 文件指纹/状态 ledger（避免重复处理） |
| `watch_events.jsonl` | 事件日志（含 webhook 投递关联元数据） |
| `masks/`, `overlays/`, `defects_regions.jsonl` | 按需启用的可视化与缺陷区域产物 |

!!! tip "幂等性"

    `watch_state.json` 记录每个文件的指纹与状态，已成功或确认失败的指纹不会被重复处理，
    直到文件内容变化。

### Batch Gate

```bash
pyimgano-bundle watch ./deploy_bundle \
    --watch-dir ./inbox \
    --output-dir ./bundle_watch \
    --max-anomaly-rate 0.05 \
    --max-error-rate 0.01 \
    --min-processed 50
```

Batch gate 仅对**当前轮询周期**裁决，不影响 bundle 契约。

---

## Webhook 投递

=== "中文"

    `watch` 模式可把每个稳定结果以 JSON POST 到下游系统。投递失败会在后续轮询自动重试，
    且不会重新执行推理。每次投递携带稳定的 `delivery_id` 与 `delivery_attempt`，下游可据此做幂等。

=== "English"

    `watch` can POST each stable result as JSON to a downstream system. Failed deliveries
    are retried on later polling cycles without rerunning inference. Each delivery carries
    a stable `delivery_id` and `delivery_attempt` for downstream idempotency.

### 基本用法

```bash
pyimgano-bundle watch ./deploy_bundle \
    --watch-dir ./inbox \
    --output-dir ./bundle_watch \
    --webhook-url https://example.com/anomaly-callback \
    --webhook-timeout-seconds 5 \
    --webhook-retry-min-seconds 30
```

### 认证：Bearer Token

```bash
# 直接传入（仅用于开发）
--webhook-bearer-token "$TOKEN"

# 从环境变量解析（推荐用于容器/CI）
--webhook-bearer-token-env PYIMGANO_WEBHOOK_TOKEN

# 从挂载文件解析（推荐用于 Kubernetes Secret）
--webhook-bearer-token-file /run/secrets/webhook_token
```

### 签名：HMAC-SHA256

| 头部 | 含义 |
|------|------|
| `X-PyImgAno-Timestamp` | 投递时刻（UTC 秒数） |
| `X-PyImgAno-Signature` | `HMAC-SHA256(<timestamp>.<raw_json_body>)` |
| `X-PyImgAno-Delivery-Id` | 稳定的投递 id（与 payload `delivery_id` 一致） |
| `X-PyImgAno-Delivery-Attempt` | 投递重试计数 |

```bash
pyimgano-bundle watch ./deploy_bundle ... \
    --webhook-signing-secret-file /run/secrets/webhook_hmac \
    --webhook-header "X-Tenant=plant-a"
```

下游验签示例（Python）：

```python
import hmac, hashlib

def verify(timestamp: str, raw_body: bytes, signature: str, secret: bytes) -> bool:
    expected = hmac.new(
        secret,
        f"{timestamp}.".encode() + raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

### 重试与可观察性

`watch_report.json` 自动聚合投递健康度，便于运维分诊：

```json
{
  "delivery_summary": {
    "delivered": 187,
    "pending_retry": 3,
    "failed": 1,
    "next_delivery_attempt_after_min": "2026-04-30T08:12:30Z",
    "last_delivery_error": "HTTP 503 from downstream",
    "last_delivery_error_path": "results/2026-04-30/img_204.png",
    "last_delivery_success_path": "results/2026-04-30/img_207.png",
    "last_delivery_success_at": "2026-04-30T08:13:02Z"
  }
}
```

!!! warning "失败重试不会重跑推理"

    投递失败仅会在后续轮询周期重试 webhook 投递，不会重新对同一指纹运行模型，
    避免下游因瞬时 5xx 触发昂贵的重新推理。

---

## 容器化部署

仓库根目录提供两份就绪文件：

- `Dockerfile.bundle-watch` — 基于 `python:3.11-slim`，以 `pyimgano-bundle watch` 为 ENTRYPOINT
- `compose.bundle-watch.yml` — 单服务 compose 模板，挂载 `deploy_bundle/`、`inbox/`、`output/`

```bash
# 构建并启动
docker compose -f compose.bundle-watch.yml up --build
```

挂载约定（参考 `compose.bundle-watch.yml`）：

| 主机目录 | 容器路径 | 挂载模式 |
|---------|---------|---------|
| `./runtime/deploy_bundle` | `/runtime/deploy_bundle` | 只读 |
| `./runtime/inbox` | `/runtime/inbox` | 读写（投递目录） |
| `./runtime/output` | `/runtime/output` | 读写（产物目录） |

!!! tip "Secret 注入"

    生产环境推荐通过 `--webhook-bearer-token-file` 与 `--webhook-signing-secret-file`
    指向容器内 secret 挂载点（`/run/secrets/...`），避免 token 出现在镜像、命令行或日志中。
