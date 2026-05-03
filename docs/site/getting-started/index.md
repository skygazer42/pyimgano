---
title: 快速开始
---

# 快速开始

=== "中文"

    从安装到首次异常检测，只需几分钟。选择下方路径开始。

=== "English"

    From installation to your first anomaly detection in minutes. Pick a path below.

---

<div class="grid" markdown>

<div class="card" markdown>

### :material-download: 安装指南

安装 pyimgano 及可选依赖，配置 GPU 环境。

[:octicons-arrow-right-24: 安装](installation.md)

</div>

<div class="card" markdown>

### :material-timer-sand: 5 分钟体验

Python API 与 CLI 两条路径，快速完成异常检测。

[:octicons-arrow-right-24: 快速开始](quickstart.md)

</div>

<div class="card" markdown>

### :material-play-circle: 首次运行

部署冒烟测试、引导式工作流与 `pyimgano-doctor` 环境检查。

[:octicons-arrow-right-24: 首次运行](first-run.md)

</div>

</div>

---

## 推荐路径

```mermaid
flowchart TD
    Start["pip install pyimgano"] --> A{"目标?"}
    A -->|"快速体验"| B["5 分钟体验"]
    A -->|"生产部署"| C["首次运行 → Deploy Smoke"]
    A -->|"深度学习模型"| D["安装 torch 等 extras"]

    B --> E["Python API / CLI"]
    C --> F["pyimgano-doctor → pyimgano-train → bundle"]
    D --> B

    style Start fill:#e3f2fd,stroke:#1565c0
    style E fill:#e8f5e9,stroke:#2e7d32
    style F fill:#fff3e0,stroke:#e65100
```
