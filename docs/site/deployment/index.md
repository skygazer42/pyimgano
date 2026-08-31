# 部署概览

=== "中文"

    pyimgano 提供完整的模型部署流水线——从训练完成到生产环境上线，每一步都有工具支撑和验证机制。

=== "English"

    pyimgano provides a complete model deployment pipeline — from training completion to production rollout, with tooling and validation at every step.

## 部署流程

```mermaid
graph LR
    A[训练<br/>pyimgano-train] --> B[验证导出<br/>pyimgano-export]
    B --> C[Artifact / Deploy Bundle]
    C --> D[验证<br/>bundle validate]
    D --> E[部署<br/>Production]

    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#9C27B0,color:#fff
    style E fill:#F44336,color:#fff
```

## 子章节

| 章节 | 说明 |
|------|------|
| [训练产物导出与导入](export.md) | fitted export、ECOD composite 认证矩阵、raw ONNX contract import 与 runtime support matrix |
| [部署包](bundle.md) | Deploy Bundle 的创建、验证与运行 |
| [工业快速路径](industrial.md) | 一份配置 → 一次运行 → 可审计产物集 |

!!! tip "快速开始"

    如果你的目标是最快速地从训练到部署，请直接参阅 [工业快速路径](industrial.md)。

!!! warning "TorchScript 是可执行信任边界"

    `single_graph` 与 `composite` TorchScript artifact 均需在确认来源后使用
    `--trust-checkpoint` 或 `load_artifact(..., trust_checkpoint=True)`。发布认证平台仍限定为
    Ubuntu x86_64、Python 3.10、CPU；详见[训练产物导出与导入](export.md)。
