---
title: 实战配方
---

# 实战配方

=== "中文"

    本节提供经过验证的端到端工作流配方，帮助你快速将 pyimgano 应用于真实场景。
    每个配方都包含完整的配置、CLI 命令和 Python 代码示例。

=== "English"

    This section provides battle-tested end-to-end workflow recipes to help you
    quickly apply pyimgano to real-world scenarios. Each recipe includes complete
    configuration, CLI commands, and Python code examples.

<div class="grid cards" markdown>

-   :material-factory:{ .lg .middle } **工业检测**

    ---

    === "中文"

        从配置到部署的完整工业检测流水线，包括合成数据 MVP 循环。

    === "English"

        Full industrial inspection pipeline from config to deployment, including
        synthetic data MVP loop.

    [:octicons-arrow-right-24: 工业检测](industrial-inspection.md)

-   :material-vector-combine:{ .lg .middle } **Embedding + Core**

    ---

    === "中文"

        深度嵌入特征 + 经典检测器的组合配方，灵活且高效。

    === "English"

        Deep embedding features + classical detector combination recipes.
        Flexible and efficient.

    [:octicons-arrow-right-24: Embedding + Core](embedding-core.md)

-   :material-grid:{ .lg .middle } **像素级基线**

    ---

    === "中文"

        零训练的像素级异常检测基线，适合快速原型验证和参考检测。

    === "English"

        Zero-training pixel-level anomaly detection baselines for quick
        prototyping and reference-based inspection.

    [:octicons-arrow-right-24: 像素级基线](pixel-baselines.md)

-   :material-rocket-launch:{ .lg .middle } **工业 MVP Loop**

    ---

    === "中文"

        合成 → 推理 → 缺陷提取的最短可行链路，30 分钟内拿到端到端产物。

    === "English"

        Synthesize → infer → defects in the shortest viable pipeline; an
        end-to-end artifact in ~30 minutes.

    [:octicons-arrow-right-24: 工业 MVP Loop](mvp-loop.md)

-   :material-bug-check:{ .lg .middle } **误报调优**

    ---

    === "中文"

        ROI / 边界抑制 / 区域得分门控，一套可复现的产线 FP 调优 loop。

    === "English"

        ROI gating, border suppression, and region-score filters — a
        reproducible loop for taming production false positives.

    [:octicons-arrow-right-24: 误报调优](false-positive-debugging.md)

</div>
