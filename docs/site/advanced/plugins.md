---
title: 插件系统
---

# 插件系统

=== "中文"

    pyimgano 使用 Python entry-point 机制实现插件扩展。
    你可以通过插件注册自定义模型、特征提取器和预处理操作，无需修改 pyimgano 源码。

=== "English"

    pyimgano uses Python entry-point mechanism for plugin extensions.
    You can register custom models, feature extractors, and preprocessing operations
    via plugins without modifying pyimgano source code.

---

## 插件组 / Plugin Group

=== "中文"

    所有插件统一注册在 `pyimgano.plugins` 入口组下。

=== "English"

    All plugins register under the `pyimgano.plugins` entry point group.

---

## 编写插件 / Authoring a Plugin

### 1. 创建插件包 / Create Plugin Package

```
my-pyimgano-plugin/
├── pyproject.toml
└── my_plugin/
    ├── __init__.py
    └── my_model.py
```

### 2. 注册入口点 / Register Entry Point

```toml title="pyproject.toml"
[project]
name = "my-pyimgano-plugin"
version = "0.1.0"
dependencies = ["pyimgano>=0.8.0"]

[project.entry-points."pyimgano.plugins"]
my_plugin = "my_plugin:register"
```

### 3. 实现注册函数 / Implement Registration

=== "中文"

    有两种注册方式：函数式和装饰器式。

=== "English"

    There are two registration styles: function-based and decorator-based.

**函数式 / Function-based:**

```python title="my_plugin/__init__.py"
def register():
    """Called by pyimgano plugin loader."""
    from pyimgano.models import register_model
    from pyimgano.features import register_feature_extractor
    from .my_model import MyCustomModel
    from .my_extractor import MyCustomExtractor

    register_model("my_custom_model", MyCustomModel)
    register_feature_extractor("my_custom_extractor", MyCustomExtractor)
```

**装饰器式 / Decorator-based:**

```python title="my_plugin/__init__.py"
def register():
    """Called by pyimgano plugin loader."""
    from pyimgano.models.registry import register_model

    @register_model(
        "my_company_detector_v1",
        tags=("vision", "industrial", "plugin"),
        metadata={"description": "Internal detector v1"},
    )
    class MyCompanyDetector:
        ...
```

### 4. 实现模型 / Implement Model

```python title="my_plugin/my_model.py"
from pyimgano.models.base import BaseDetector
import numpy as np


class MyCustomModel(BaseDetector):
    """Example custom anomaly detector."""

    def fit(self, X, y=None):
        self.threshold_ = np.percentile(self._score(X), 95)
        return self

    def predict(self, X):
        scores = self._score(X)
        return (scores > self.threshold_).astype(int)

    def score_samples(self, X):
        return self._score(X)

    def _score(self, X):
        # Your scoring logic
        ...
```

---

## 加载插件 / Loading Plugins

=== "中文"

    pyimgano 在启动时自动发现并加载已安装的插件。也可以手动触发加载：

=== "English"

    pyimgano automatically discovers and loads installed plugins at startup.
    You can also trigger loading manually:

```python
from pyimgano.plugins import load_plugins

# 默认: 警告并继续
load_plugins()

# 严格模式: 加载失败时抛出异常
load_plugins(on_error="raise")

# 静默模式: 忽略错误
load_plugins(on_error="ignore")
```

| `on_error` | 说明 / Description |
|---|---|
| `"warn"` | 警告并继续（默认） |
| `"raise"` | 失败时立即抛出异常 |
| `"ignore"` | 静默忽略错误 |

---

## CLI 使用 / CLI Usage

```bash
# 列出包括插件模型在内的所有可用模型
pyimgano-benchmark --plugins --list-models

# 使用插件模型运行推理
pyimgano-infer \
  --model my_custom_model \
  --data ./data/test/ \
  --plugins
```

!!! note "--plugins 标志"

    === "中文"

        CLI 命令需要 `--plugins` 标志来显式启用插件加载。
        这确保了默认行为的可预测性，且避免导入重量级可选依赖。

    === "English"

        CLI commands require the `--plugins` flag to explicitly enable plugin loading.
        This ensures predictable default behavior and avoids importing heavy optional dependencies.

---

## 使用场景 / Use Cases

=== "中文"

    - **内部模型** — 注册公司内部专有的检测算法
    - **私有注册表** — 从内部 PyPI 源分发插件包
    - **实验模型** — 在不 fork 主仓库的情况下快速实验
    - **自定义提取器** — 注册特定领域的特征提取器
    - **运行时隔离** — 插件可依赖可选重量级运行时，核心 pyimgano 保持轻量

=== "English"

    - **Internal models** — Register proprietary detection algorithms
    - **Private registries** — Distribute plugin packages from internal PyPI mirrors
    - **Experimental models** — Quickly experiment without forking the main repo
    - **Custom extractors** — Register domain-specific feature extractors
    - **Runtime isolation** — Plugins can depend on heavy optional runtimes while core pyimgano stays lightweight
