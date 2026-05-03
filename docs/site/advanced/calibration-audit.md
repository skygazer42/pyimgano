---
title: 阈值校准审计 / Calibration Audit
---

# 阈值校准审计 / Calibration Audit

=== "中文"

    pyimgano 把图像级阈值导出到 deploy artifacts，因此阈值本身需要可追溯的审计记录。
    紧凑的审计产物是 `artifacts/calibration_card.json`，回答三个问题：
    部署的阈值是什么？怎么得到的？属于哪个 split / category 上下文？

=== "English"

    pyimgano exports image-level thresholds into deploy artifacts, so the threshold
    itself needs an audit trail. The compact audit artifact `artifacts/calibration_card.json`
    answers three questions: what threshold is deployed, how was it derived, and which
    dataset split / category context did it come from?

参考完整文档：[docs/CALIBRATION_AUDIT.md](https://github.com/skygazer42/pyimgano/blob/main/docs/CALIBRATION_AUDIT.md)。

## 最小流程

```bash
# 1. 训练并导出 infer-config + calibration card
pyimgano-train --config my_config.json \
    --export-infer-config \
    --export-deploy-bundle

# 2. 审计 run 质量（含 calibration provenance 检查）
pyimgano runs quality runs/<run_dir> --json

# 3. 部署前再次 gate
pyimgano-validate-infer-config deploy_bundle/infer_config.json --strict
pyimgano bundle validate deploy_bundle/ --json
```

## calibration_card.json 关键字段

| 字段 | 含义 |
|------|------|
| `threshold` | 部署阈值数值 |
| `threshold_provenance.strategy` | 计算策略（如 `normal_quantile` / `pot` / `manual`） |
| `threshold_provenance.quantile` | 当 quantile 策略时使用的分位数 |
| `threshold_provenance.source` | 数据来源（split / category 名称） |
| `score_summary` | normal / anomaly 得分分布的轻量摘要 |
| `dataset_summary` | 样本计数、anomaly 比例、像素度量门 |

## 推荐审计动作

- **每次发版**：保留 `calibration_card.json` 与 `handoff_report.json`，与 git tag 关联
- **生产回归**：在 watch run 后比对 `watch_report.json` 中实际 anomaly_rate 与 card 中的 prior
- **跨 split 漂移**：若多 category 部署，逐个 category 校验 `threshold_provenance.source` 与 `score_summary` 一致性

!!! warning "禁止手改 card"

    `calibration_card.json` 是审计追溯起点，应当通过重新训练 / 配置变更重生成，
    而不是手工编辑。手工编辑会破坏 `bundle_manifest.json` 的哈希校验。
