# Package Boundary Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish a real detector runtime contract and remove the first reverse dependencies from CLI modules into core orchestration code.

**Architecture:** Add a shared detector protocol plus a small runtime normalization adapter, then extract model-option resolution into a reusable service module. Use those seams to stop `workbench`, `recipes`, and inference code from importing CLI internals while keeping the current CLI surface stable.

**Tech Stack:** Python, NumPy, pytest, `argparse`, dataclasses, runtime-checkable `Protocol`, existing `pyimgano` registry/capability helpers.

---

### Task 1: Add Detector Runtime Protocols

**Files:**
- Create: `pyimgano/models/protocols.py`
- Modify: `pyimgano/models/base_detector.py`
- Modify: `pyimgano/models/deep_contract.py`
- Test: `tests/test_detector_protocols.py`

**Step 1: Write the failing test**

Create `tests/test_detector_protocols.py`:

```python
from __future__ import annotations

import numpy as np

from pyimgano.models.base_detector import BaseDetector
from pyimgano.models.protocols import (
    DetectorProtocol,
    PixelMapDetectorProtocol,
    normalize_anomaly_maps,
)


class _DummyDetector(BaseDetector):
    input_mode = "numpy"

    def fit(self, X, y=None):
        self.decision_scores_ = np.asarray([0.1, 0.9], dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        n = len(list(X))
        return np.zeros((n,), dtype=np.float32)


class _DummyPixelDetector(_DummyDetector):
    def predict_anomaly_map(self, X):
        n = len(list(X))
        return [np.ones((4, 4), dtype=np.float32) for _ in range(n)]


def test_detector_runtime_protocol_accepts_native_detector():
    detector = _DummyDetector()
    assert isinstance(detector, DetectorProtocol)


def test_normalize_anomaly_maps_accepts_list_outputs():
    detector = _DummyPixelDetector()
    assert isinstance(detector, PixelMapDetectorProtocol)
    maps = normalize_anomaly_maps(detector.predict_anomaly_map([0, 1]), n_expected=2)
    assert maps.shape == (2, 4, 4)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector_protocols.py -v`

Expected: FAIL because `pyimgano.models.protocols` and `normalize_anomaly_maps(...)` do not exist yet.

**Step 3: Write minimal implementation**

Implement `pyimgano/models/protocols.py` with:

```python
from __future__ import annotations

from typing import Any, Iterable, Literal, Protocol, runtime_checkable

import numpy as np

InputMode = Literal["paths", "numpy", "features"]


@runtime_checkable
class DetectorProtocol(Protocol):
    input_mode: InputMode

    def fit(self, X: Iterable[Any], y: Any | None = None) -> "DetectorProtocol":
        ...

    def decision_function(self, X: Iterable[Any]) -> np.ndarray:
        ...

    def predict(self, X: Iterable[Any]) -> np.ndarray:
        ...


@runtime_checkable
class PixelMapDetectorProtocol(DetectorProtocol, Protocol):
    def predict_anomaly_map(self, X: Iterable[Any]) -> Any:
        ...


def normalize_scores(scores: Any, *, n_expected: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) != int(n_expected):
        raise ValueError(...)
    return arr


def normalize_anomaly_maps(maps: Any, *, n_expected: int) -> np.ndarray:
    if isinstance(maps, list):
        maps = np.stack([np.asarray(m, dtype=np.float32) for m in maps], axis=0)
    arr = np.asarray(maps, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != int(n_expected):
        raise ValueError(...)
    return arr
```

Also:

- make `BaseDetector` define `input_mode = "features"` as the conservative default
- update `pyimgano.models.deep_contract` to import and reuse `normalize_scores(...)` and `normalize_anomaly_maps(...)` instead of keeping partially overlapping helpers

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_detector_protocols.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/protocols.py pyimgano/models/base_detector.py pyimgano/models/deep_contract.py tests/test_detector_protocols.py
git commit -m "refactor: add detector runtime protocols"
```

---

### Task 2: Introduce a Runtime Adapter for Inference

**Files:**
- Create: `pyimgano/inference/runtime_adapter.py`
- Modify: `pyimgano/inference/api.py`
- Modify: `pyimgano/inference/runtime_wrappers.py`
- Test: `tests/test_inference_runtime_adapter.py`

**Step 1: Write the failing test**

Create `tests/test_inference_runtime_adapter.py`:

```python
from __future__ import annotations

import numpy as np

from pyimgano.inference.runtime_adapter import score_and_maps


class _TupleDetector:
    input_mode = "numpy"

    def decision_function(self, X):
        n = len(list(X))
        scores = np.arange(float(n), dtype=np.float32)
        maps = [np.full((3, 3), fill_value=i, dtype=np.float32) for i in range(n)]
        return scores, maps


class _SingleMapDetector:
    input_mode = "numpy"

    def decision_function(self, X):
        n = len(list(X))
        return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    def get_anomaly_map(self, X):
        return np.ones((3, 3), dtype=np.float32)


def test_score_and_maps_normalizes_tuple_return():
    X = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    scores, maps = score_and_maps(_TupleDetector(), X)
    assert scores.shape == (3,)
    assert maps is not None
    assert maps.shape == (3, 3, 3)


def test_score_and_maps_broadcasts_single_image_map_api():
    X = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    scores, maps = score_and_maps(_SingleMapDetector(), X)
    assert scores.shape == (2,)
    assert maps is not None
    assert maps.shape == (2, 3, 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference_runtime_adapter.py -v`

Expected: FAIL because `score_and_maps(...)` does not exist yet.

**Step 3: Write minimal implementation**

Create `pyimgano/inference/runtime_adapter.py` with:

```python
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pyimgano.models.protocols import normalize_anomaly_maps, normalize_scores


def score_and_maps(detector: Any, inputs: Sequence[Any]) -> tuple[np.ndarray, np.ndarray | None]:
    out = detector.decision_function(inputs)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        scores, maps = out
        return normalize_scores(scores, n_expected=len(inputs)), normalize_anomaly_maps(
            maps, n_expected=len(inputs)
        )

    scores = normalize_scores(out, n_expected=len(inputs))
    if hasattr(detector, "predict_anomaly_map"):
        maps = detector.predict_anomaly_map(inputs)
        return scores, normalize_anomaly_maps(maps, n_expected=len(inputs))
    if hasattr(detector, "get_anomaly_map"):
        maps = [np.asarray(detector.get_anomaly_map(item), dtype=np.float32) for item in inputs]
        return scores, normalize_anomaly_maps(maps, n_expected=len(inputs))
    return scores, None
```

Then:

- replace `_call_decision_function_with_optional_maps(...)` in `pyimgano/inference/api.py` with calls into `score_and_maps(...)`
- keep `runtime_wrappers.py` focused only on wrapper detection and unwrapping

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference_runtime_adapter.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/inference/runtime_adapter.py pyimgano/inference/api.py pyimgano/inference/runtime_wrappers.py tests/test_inference_runtime_adapter.py
git commit -m "refactor: centralize inference runtime adaptation"
```

---

### Task 3: Extract Model Option Resolution out of CLI

**Files:**
- Create: `pyimgano/services/model_options.py`
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `pyimgano/robust_cli.py`
- Modify: `pyimgano/recipes/builtin/micro_finetune_autoencoder.py`
- Test: `tests/test_model_options_service.py`
- Test: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Create `tests/test_model_options_service.py`:

```python
from pyimgano.services.model_options import resolve_model_options


def test_resolve_model_options_merges_user_preset_and_auto_kwargs():
    out = resolve_model_options(
        model_name="vision_padim",
        preset="industrial-fast",
        user_kwargs={"d_reduced": 16},
        auto_kwargs={"device": "cpu", "contamination": 0.1, "pretrained": False},
        checkpoint_path=None,
    )

    assert out["d_reduced"] == 16
    assert out["contamination"] == 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_options_service.py -v`

Expected: FAIL because `pyimgano.services.model_options` does not exist yet.

**Step 3: Write minimal implementation**

Create `pyimgano/services/model_options.py` with a stable API:

```python
from __future__ import annotations

from typing import Any

from pyimgano.cli_common import build_model_kwargs, merge_checkpoint_path
from pyimgano.cli import _resolve_preset_kwargs


def resolve_model_options(
    *,
    model_name: str,
    preset: str | None,
    user_kwargs: dict[str, Any],
    auto_kwargs: dict[str, Any],
    checkpoint_path: str | None,
) -> dict[str, Any]:
    merged_user = merge_checkpoint_path(user_kwargs, checkpoint_path=checkpoint_path)
    preset_kwargs = _resolve_preset_kwargs(preset, model_name)
    return build_model_kwargs(
        model_name,
        user_kwargs=merged_user,
        preset_kwargs=preset_kwargs,
        auto_kwargs=auto_kwargs,
    )
```

Then migrate call sites:

- `pyimgano/infer_cli.py`
- `pyimgano/workbench/runner.py`
- `pyimgano/robust_cli.py`
- `pyimgano/recipes/builtin/micro_finetune_autoencoder.py`

The first pass can leave thin compatibility wrappers in `cli.py` and `cli_common.py`; the critical goal is that core code no longer imports CLI helpers.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_model_options_service.py tests/test_cli_model_kwargs.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/services/model_options.py pyimgano/cli.py pyimgano/infer_cli.py pyimgano/workbench/runner.py pyimgano/robust_cli.py pyimgano/recipes/builtin/micro_finetune_autoencoder.py tests/test_model_options_service.py tests/test_cli_model_kwargs.py
git commit -m "refactor: extract model option resolution from cli"
```

---

### Task 4: Add a Thin Inference Service Seam

**Files:**
- Create: `pyimgano/services/inference_service.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_inference_service.py`
- Test: `tests/test_infer_cli_smoke.py`

**Step 1: Write the failing test**

Create `tests/test_inference_service.py`:

```python
from __future__ import annotations

import numpy as np

from pyimgano.services.inference_service import run_inference


class _DummyDetector:
    input_mode = "numpy"

    def decision_function(self, X):
        n = len(list(X))
        return np.arange(float(n), dtype=np.float32)


def test_run_inference_returns_structured_records():
    X = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    result = run_inference(detector=_DummyDetector(), inputs=X, input_format="rgb_u8_hwc")
    assert len(result.records) == 2
    assert result.records[1].score >= result.records[0].score
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference_service.py -v`

Expected: FAIL because `run_inference(...)` does not exist yet.

**Step 3: Write minimal implementation**

Create `pyimgano/services/inference_service.py` that:

- accepts `detector`, normalized inputs, and options
- delegates runtime normalization to `pyimgano.inference.runtime_adapter`
- returns a structured result dataclass instead of printing

Update `pyimgano/infer_cli.py` so `main(...)`:

- parses arguments
- calls the service
- formats service results for stdout/files

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_inference_service.py tests/test_infer_cli_smoke.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/services/inference_service.py pyimgano/infer_cli.py tests/test_inference_service.py tests/test_infer_cli_smoke.py
git commit -m "refactor: add inference service seam"
```

---

### Task 5: Remove CLI Dependencies from Workbench and Recipes

**Files:**
- Create: `pyimgano/services/workbench_service.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `pyimgano/train_cli.py`
- Modify: `pyimgano/recipes/builtin/classical_recipes.py`
- Modify: `pyimgano/recipes/builtin/industrial_adapt.py`
- Modify: `pyimgano/recipes/builtin/industrial_adapt_fp40.py`
- Modify: `pyimgano/recipes/builtin/industrial_adapt_highres.py`
- Modify: `pyimgano/recipes/builtin/industrial_embedding_core_fast.py`
- Modify: `pyimgano/recipes/builtin/micro_finetune_autoencoder.py`
- Test: `tests/test_workbench_export_infer_config.py`
- Test: `tests/test_integration_workbench_train_then_infer.py`

**Step 1: Write the failing test**

Add a focused assertion to `tests/test_workbench_export_infer_config.py`:

```python
def test_workbench_runner_does_not_import_cli_module(monkeypatch):
    import pyimgano.workbench.runner as runner

    imported = []

    def _spy(name, *args, **kwargs):
        imported.append(name)
        return original(name, *args, **kwargs)

    import builtins

    original = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _spy)
    try:
        assert "pyimgano.cli" not in imported
    finally:
        monkeypatch.setattr(builtins, "__import__", original)
```

If that specific shape is too brittle, replace it with a regression test that exercises `run_workbench(...)` while mocking `pyimgano.cli` import to raise.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workbench_export_infer_config.py -v`

Expected: FAIL because `pyimgano.workbench.runner` currently imports CLI helpers.

**Step 3: Write minimal implementation**

Create `pyimgano/services/workbench_service.py` for:

- model option resolution
- detector creation
- threshold calibration wiring
- infer-config export assembly

Then:

- update `pyimgano/workbench/runner.py` to depend on the service module
- update built-in recipes to depend on service/workbench APIs rather than CLI helpers
- keep `train_cli.py` as a parser/formatter adapter

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_workbench_export_infer_config.py tests/test_integration_workbench_train_then_infer.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/services/workbench_service.py pyimgano/workbench/runner.py pyimgano/train_cli.py pyimgano/recipes/builtin/classical_recipes.py pyimgano/recipes/builtin/industrial_adapt.py pyimgano/recipes/builtin/industrial_adapt_fp40.py pyimgano/recipes/builtin/industrial_adapt_highres.py pyimgano/recipes/builtin/industrial_embedding_core_fast.py pyimgano/recipes/builtin/micro_finetune_autoencoder.py tests/test_workbench_export_infer_config.py tests/test_integration_workbench_train_then_infer.py
git commit -m "refactor: remove cli dependencies from workbench flows"
```

---

### Task 6: Verification Sweep for Milestone 1

**Files:**
- Modify: none
- Test: `tests/test_detector_protocols.py`
- Test: `tests/test_inference_runtime_adapter.py`
- Test: `tests/test_model_options_service.py`
- Test: `tests/test_inference_service.py`
- Test: `tests/test_cli_model_kwargs.py`
- Test: `tests/test_infer_cli_smoke.py`
- Test: `tests/test_workbench_export_infer_config.py`
- Test: `tests/test_integration_workbench_train_then_infer.py`

**Step 1: Run focused verification**

Run:

```bash
pytest \
  tests/test_detector_protocols.py \
  tests/test_inference_runtime_adapter.py \
  tests/test_model_options_service.py \
  tests/test_inference_service.py \
  tests/test_cli_model_kwargs.py \
  tests/test_infer_cli_smoke.py \
  tests/test_workbench_export_infer_config.py \
  tests/test_integration_workbench_train_then_infer.py -v
```

Expected: PASS.

**Step 2: Run smoke verification for the existing surfaces**

Run:

```bash
pytest tests/test_cli_smoke.py tests/test_core_models_registry_smoke.py -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git add .
git commit -m "test: verify package boundary refactor milestone one"
```

Plan complete and saved to `docs/plans/2026-03-11-package-boundary-refactor.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
