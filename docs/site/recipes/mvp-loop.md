---
title: 工业 MVP Loop / Industrial MVP Loop
---

# 工业 MVP Loop / Industrial MVP Loop

=== "中文"

    一条最短可行链路：**合成 → 推理 → 缺陷提取**，让你在 30 分钟内拿到可验证的端到端
    产物，再决定要不要投入完整的真实数据采集流程。

=== "English"

    The shortest viable pipeline: **synthesize → infer → defects**. Get an end-to-end
    verifiable artifact in ~30 minutes before committing to full real-data collection.

完整文档：[docs/INDUSTRIAL_MVP_LOOP.md](https://github.com/skygazer42/pyimgano/blob/main/docs/INDUSTRIAL_MVP_LOOP.md)。

## 三步走

```bash
# 1. 合成一个最小数据集（含 normal + 合成异常 + masks + manifest）
pyimgano-synthesize \
    --normals-dir ./normals \
    --output-dir ./mvp_dataset \
    --presets scratch,stain,glare \
    --num-samples 50

# 2. 用 industrial-adapt 配方训练并直接导出 deploy bundle
pyimgano-train \
    --config examples/configs/industrial_adapt.json \
    --dataset.manifest_path ./mvp_dataset/manifest.jsonl \
    --export-deploy-bundle

# 3. 跑一次 bundle 推理 + 缺陷提取
pyimgano bundle run runs/<run_dir>/deploy_bundle/ \
    --image-dir ./mvp_dataset/test/ \
    --output-dir ./mvp_run/
```

## 验收要点

- `runs/<run_dir>/report.json` 含 AUROC / pixel SegF1 基线值
- `mvp_run/run_report.json` 中 `batch_verdict=PASS` 表明 batch gate 通过
- `mvp_run/results.jsonl` 可作为下游可视化 / FP 调优的输入

!!! tip "下一步"

    跑通 MVP 后，常见后续：
    1. 替换合成数据为真实采集图像（保留 manifest 结构）
    2. 用 `pyimgano-bundle watch` 接入产线投递目录
    3. 用 [误报调优指南](false-positive-debugging.md) 收紧产线 FP 表现
