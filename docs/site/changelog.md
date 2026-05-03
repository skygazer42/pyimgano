---
title: 更新日志
---

# 更新日志 / Changelog

=== "中文"

    完整的更新日志请查看项目根目录的 [CHANGELOG.md](https://github.com/skygazer42/pyimgano/blob/main/CHANGELOG.md)。
    以下是最近版本的主要更新摘要。

=== "English"

    For the full changelog, see [CHANGELOG.md](https://github.com/skygazer42/pyimgano/blob/main/CHANGELOG.md)
    in the project root. Below is a summary of recent highlights.

---

## v0.9.1 — 2026-05-03

### 发布对齐 / Release Alignment

=== "中文"

    - 将 post-0.9.0 的 main 分支内容准备为新的 `0.9.1` 包版本，避免 PyPI 重复上传已存在的 `0.9.0` 文件
    - 发布 workflow 新增版本/tag 审计：GitHub Release tag 必须匹配 `pyproject.toml` 与 `pyimgano.__version__`
    - 发布文档明确：仅 push git tag 不会触发 PyPI 上传，必须发布 GitHub Release

=== "English"

    - Prepared post-0.9.0 main-branch changes as package version `0.9.1`, avoiding attempts to overwrite existing PyPI `0.9.0` files
    - Added a version/tag audit to the release workflow so GitHub Release tags must match `pyproject.toml` and `pyimgano.__version__`
    - Clarified that pushing a git tag alone does not upload to PyPI; publishing a GitHub Release is required

---

## v0.9.0 — 2026-04-03

### 热文件夹运行时 / Hot-folder Runtime

=== "中文"

    - 新增 `pyimgano-bundle watch`：将已验证的 deploy bundle 部署为长驻服务，轮询 `--watch-dir` 中的输入文件并追加聚合产物
    - 不需重跑训练/导出即可生成增量 `results.jsonl` / `watch_report.json` / `watch_state.json` / `watch_events.jsonl`
    - 提供容器化运行示例 `Dockerfile.bundle-watch` 与 `compose.bundle-watch.yml`

=== "English"

    - New `pyimgano-bundle watch` runs a validated deploy bundle as a long-running service that polls `--watch-dir` and appends aggregate artifacts
    - Streams incremental `results.jsonl` / `watch_report.json` / `watch_state.json` / `watch_events.jsonl` without rerunning training or export
    - Container-ready examples `Dockerfile.bundle-watch` and `compose.bundle-watch.yml`

### Webhook 投递 / Webhook Delivery

=== "中文"

    - `bundle watch --webhook-url URL` 支持把每个稳定结果以 JSON 推送到下游系统
    - 支持 Bearer token 认证（直接传入 / 环境变量 / 文件三种方式）
    - HMAC 签名头 `X-PyImgAno-Timestamp` 与 `X-PyImgAno-Signature`，签名为 `HMAC-SHA256(<timestamp>.<raw_json_body>)`
    - 每次投递携带稳定的 `delivery_id` / `delivery_attempt` 元数据；可配置失败重试退避 `--webhook-retry-min-seconds`
    - `watch_report.json` 聚合 `pending_retry`、最早重试时间、最近一次错误与最近一次成功，便于运维快速分诊

=== "English"

    - `bundle watch --webhook-url URL` POSTs each processed result to downstream systems
    - Bearer token auth via direct value, environment variable, or mounted file
    - HMAC signing headers `X-PyImgAno-Timestamp` / `X-PyImgAno-Signature` (signed body = `HMAC-SHA256(<ts>.<raw_json>)`)
    - Stable `delivery_id` / `delivery_attempt` metadata in payload + headers; configurable retry backoff via `--webhook-retry-min-seconds`
    - `watch_report.json` aggregates pending retries, earliest retry horizon, last error, and last successful delivery for fast operator triage

### 操作流程引导 / Operator Workflow Guidance

=== "中文"

    - 根 CLI、`doctor` 与 `pyim` 全面强化引导：结构化 train/infer/runs 后续命令
    - 训练发现路径：`--list-recipes`、`--recipe-info`、`--dry-run`、`--preflight`
    - `pyim --goal ...` 现在返回带结构化 train 后续命令的 recipe 推荐
    - 可观察性：dataset-readiness 元数据贯穿 train / workbench / runs / publication 路径

=== "English"

    - Root CLI, `doctor`, and `pyim` now expose structured train/infer/runs follow-up commands
    - Training discovery: `--list-recipes`, `--recipe-info`, `--dry-run`, `--preflight`
    - `pyim --goal ...` returns recipe picks carrying structured train follow-up commands
    - Dataset-readiness metadata propagates across train / workbench / runs / publication so operator gates retain dataset-trust context

### 修复 / Fixed

=== "中文"

    - 安全 checkpoint 加载现在能正确处理 numpy 后端 payload 的兼容恢复路径
    - 抑制 DevNet 无监督使用与 TorchScript 弃用路径的噪声警告，让生产日志更干净
    - 不支持的 AnomalyDINO checkpoint 不再破坏已保存的运行

=== "English"

    - Safe checkpoint loading now handles numpy-backed payloads in compatibility restore paths
    - Suppressed noisy DevNet unsupervised + TorchScript deprecation warnings for cleaner production logs
    - Unsupported AnomalyDINO checkpoints no longer break saved runs

---

## v0.8.0 — 2026-03-31

### 合成异常 / Synthesis

=== "中文"

    - 新增 `pyimgano.synthesis` 合成异常生成包
    - `AnomalySynthesizer` 支持 17+ 缺陷预设（scratch, stain, pit, glare, rust, oil, crack, brush, spatter, tape 等）
    - 新增 `DefectBank` 真实缺陷裁剪粘贴合成
    - `pyimgano-synthesize` CLI 支持 `--from-manifest`、`--preview`、`--presets`、`--roi-mask` 等参数
    - `SyntheticAnomalyDataset` 支持在线合成和严重度课程控制

=== "English"

    - New `pyimgano.synthesis` synthetic anomaly generation package
    - `AnomalySynthesizer` with 17+ defect presets (scratch, stain, pit, glare, rust, oil, crack, brush, spatter, tape, etc.)
    - New `DefectBank` for real defect crop-paste synthesis
    - `pyimgano-synthesize` CLI with `--from-manifest`, `--preview`, `--presets`, `--roi-mask`
    - `SyntheticAnomalyDataset` with online synthesis and severity curriculum

### 鲁棒性 / Robustness

=== "中文"

    - 鲁棒性基准干扰可选生成掩码
    - 新增合成式干扰辅助函数 `apply_synthesis_preset`
    - 新增 `CorruptionsDataset` 用于 torch 流水线中的确定性干扰

=== "English"

    - Robustness benchmark corruptions with optional mask emission
    - New `apply_synthesis_preset` synthesis-style corruption helper
    - New `CorruptionsDataset` for deterministic corruptions in torch pipelines

### 预处理 / Preprocessing

=== "中文"

    - 新增 MSRCR-lite Retinex 光照归一化
    - 工业预处理预设 `retinex_illumination_normalization`

=== "English"

    - New MSRCR-lite Retinex illumination normalization
    - Industrial preprocessing preset `retinex_illumination_normalization`

### CLI 改进 / CLI Improvements

=== "中文"

    - CLI 默认离线安全（`--no-pretrained`），不再隐式下载权重
    - `pyimgano-infer` 支持 ONNX Runtime CPU 调优
    - 新增 `--onnx-session-options` 和 `--onnx-sweep` 参数

=== "English"

    - CLI defaults to offline-safe (`--no-pretrained`), no implicit weight downloads
    - `pyimgano-infer` supports ONNX Runtime CPU tuning
    - New `--onnx-session-options` and `--onnx-sweep` flags

### 特征提取器 / Feature Extractors

=== "中文"

    - 新增一等公民特征提取子系统（`pyimgano.features`）
    - 内置 14 种提取器：identity, hog, lbp, gabor_bank, color_hist, edge_stats, fft_lowfreq, patch_stats, multi, pca_projector, standard_scaler, normalize, torchvision_backbone, torchscript_embed
    - 支持特征流水线 spec、导出辅助和磁盘缓存

=== "English"

    - New first-class feature extractor subsystem (`pyimgano.features`)
    - 14 built-in extractors: identity, hog, lbp, gabor_bank, color_hist, edge_stats, fft_lowfreq, patch_stats, multi, pca_projector, standard_scaler, normalize, torchvision_backbone, torchscript_embed
    - Feature pipeline specs, export helpers, and disk caching

### 模型 / Models

=== "中文"

    - 移除 `pyod` 运行时依赖，经典检测器迁移为原生实现
    - 新增大量 `core_*` 和 `vision_*` 原生检测器
    - 新增像素级基线：`ssim_template_map`, `ssim_struct_map`, `vision_template_ncc_map`, `vision_phase_correlation_map` 等
    - 新增 `vision_embedding_core` 嵌入+核心组合路线
    - 新增预配置工业快捷模型：`vision_resnet18_ecod`, `vision_resnet18_iforest` 等

=== "English"

    - Removed `pyod` runtime dependency; classical detectors ported to native implementations
    - Many new `core_*` and `vision_*` native detectors
    - New pixel-map baselines: `ssim_template_map`, `ssim_struct_map`, `vision_template_ncc_map`, `vision_phase_correlation_map`, etc.
    - New `vision_embedding_core` embedding+core combination route
    - New preconfigured industrial shortcut models: `vision_resnet18_ecod`, `vision_resnet18_iforest`, etc.

---

=== "中文"

    查看完整更新日志和所有历史版本：[GitHub Releases](https://github.com/skygazer42/pyimgano/releases)

=== "English"

    See the full changelog and all historical versions: [GitHub Releases](https://github.com/skygazer42/pyimgano/releases)
