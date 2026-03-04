# Plugins (Third‑Party Extensions)

`pyimgano` supports **opt‑in** third‑party extensions via Python **entry points**.

Why this matters for industrial use:

- You can ship internal models/pipelines in a separate package (private PyPI / artifact repo).
- Production users can `pip install` your plugin package without forking `pyimgano`.
- Core `pyimgano` stays dependency‑light; plugins can depend on optional runtimes.

---

## How it works

Plugins are loaded by calling:

```python
from pyimgano.plugins import load_plugins

load_plugins()  # loads entry point group: "pyimgano.plugins"
```

The default group is:

- `pyimgano.plugins`

Each entry point must resolve to a **zero‑argument callable**. That callable should register
models/features by calling:

- `pyimgano.models.registry.register_model(...)`
- `pyimgano.features.registry.register_feature_extractor(...)`

---

## CLI usage

Plugins are **not** loaded by default. To load them before discovery/benchmarking, pass:

```bash
pyimgano-benchmark --plugins --list-models
```

This flag is opt‑in because plugins may import optional heavy dependencies.

---

## Authoring a plugin package

In your plugin package `pyproject.toml`, add an entry point:

```toml
[project.entry-points."pyimgano.plugins"]
my_company_models = "my_company_pyimgano_plugin:register"
```

Then implement `my_company_pyimgano_plugin.py`:

```python
from __future__ import annotations

from pyimgano.models.registry import register_model


def register() -> None:
    @register_model(
        "my_company_detector_v1",
        tags=("vision", "industrial", "plugin"),
        metadata={"description": "Internal detector v1"},
    )
    class MyCompanyDetector:
        ...
```

Now install your plugin package and enable plugins:

```bash
pip install my-company-pyimgano-plugin
pyimgano-benchmark --plugins --list-models
```

---

## Error behavior

`load_plugins(on_error=...)` supports:

- `raise`: fail fast
- `warn`: warn and continue (default)
- `ignore`: silently continue

