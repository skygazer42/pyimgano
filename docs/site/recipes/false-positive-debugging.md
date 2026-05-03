---
title: 误报调优指南 / False Positive Debugging
---

# 误报调优指南 / False Positive Debugging

=== "中文"

    模型已经能识别异常，但产线上误报仍然过多 — 这是工业落地的常态。本节给出一套
    可复现的调优 loop，避免“调阈值压一切”的脆弱做法。完整论述见
    [docs/FALSE_POSITIVE_DEBUGGING.md](https://github.com/skygazer42/pyimgano/blob/main/docs/FALSE_POSITIVE_DEBUGGING.md)。

=== "English"

    Your model detects anomalies, but production keeps surfacing too many false positives —
    a normal industrial reality. This page distills a reproducible debugging loop that
    avoids fragile "just lower the threshold" fixes. Full discussion lives in
    [docs/FALSE_POSITIVE_DEBUGGING.md](https://github.com/skygazer42/pyimgano/blob/main/docs/FALSE_POSITIVE_DEBUGGING.md).

## 90% 场景速查清单

1. **先做 ROI 门控** — 仅检测可检测区域，别用阈值去对抗背景 FP
2. **抑制边界** — 工业相机边缘伪影常见，启用 `border_ignore_px`
3. **开启 overlay** — 调参前先看热图 + mask + region 三层可视化
4. **平滑 + 滞回阈值** — 去椒盐，仅保留与种子像素连通的点
5. **形状/面积过滤** — `min_area`、`min_solidity`、`max_aspect_ratio` 抑制条状/碎片噪声
6. **区域得分门控** — `min_score_max`、`min_score_mean` 拒绝弱响应

## 推荐 CLI 起点

```bash
pyimgano-infer \
    --infer-config deploy_bundle/infer_config.json \
    --input ./inbox \
    --defects \
    --defects-image-space \
    --save-overlays ./debug/overlays \
    --save-masks ./debug/masks \
    --save-jsonl ./debug/results.jsonl \
    --profile-json ./debug/timing.json
```

`infer_config.json` 中的 `defects.*` 默认值会被透传，但仍可通过命令行覆盖单项。

## 闭环

每次调整后：

1. 用 `pyimgano runs quality` 检查 run 质量门
2. 用 `pyimgano-evaluate` 在带标签集合上回归 AUROC / pixel SegF1
3. 用 `pyimgano bundle validate` 确认调优过的 infer-config 仍能通过 deploy 契约

!!! tip "FP40 预设"

    内置 recipe `industrial-adapt-fp40` 提供已校准的 FP 抑制默认值，
    适合作为基线再做小幅参数调优。
