# Robust CLI Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin `pyimgano.robust_cli` by moving robustness benchmark orchestration into a reusable service module.

**Architecture:** Keep `pyimgano.robust_cli` as an argparse/JSON adapter. Introduce a `robustness_service` that owns dataset loading, model option resolution, corruption/input preparation, pixel-postprocess construction, and robustness benchmark execution so the orchestration is testable without the CLI surface.

**Tech Stack:** Python, argparse, dataclasses, pytest, NumPy, existing `pyimgano.robustness.benchmark`, model registry helpers, JSON reporting utilities.

---

### Task 1: Add a Robustness Service Seam

**Files:**
- Create: `pyimgano/services/robustness_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/robust_cli.py`
- Test: `tests/test_robustness_service.py`
- Test: `tests/test_robust_cli_smoke.py`

**Step 1: Write the failing test**

Create `tests/test_robustness_service.py` with one service delegation assertion:

```python
from __future__ import annotations

from pyimgano.services.robustness_service import RobustnessRunRequest, run_robustness_request


def test_run_robustness_request_delegates_to_benchmark(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service

    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: calls.append(kwargs) or {"clean": {}, "corruptions": {}},
    )

    payload = run_robustness_request(
        RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="vision_ecod",
        )
    )

    assert "robustness" in payload
    assert payload["model"] == "vision_ecod"
```

Add one CLI delegation assertion in `tests/test_robust_cli_smoke.py`:

```python
def test_robust_cli_delegates_run_mode_to_robustness_service(monkeypatch, capsys) -> None:
    from pyimgano.robust_cli import main
    import pyimgano.services.robustness_service as robustness_service

    calls = []

    monkeypatch.setattr(
        robustness_service,
        "run_robustness_request",
        lambda request: calls.append(request) or {"dataset": request.dataset, "category": request.category, "model": request.model, "robustness": {"clean": {}, "corruptions": {}}},
    )

    rc = main(["--dataset", "mvtec", "--root", "/tmp", "--category", "bottle", "--model", "vision_ecod"])
    assert rc == 0
    assert calls[0].model == "vision_ecod"
    assert '"model": "vision_ecod"' in capsys.readouterr().out
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_robustness_service.py tests/test_robust_cli_smoke.py -k "delegates" -v
```

Expected: FAIL because `pyimgano.services.robustness_service` does not exist and `pyimgano.robust_cli` still assembles execution inline.

**Step 3: Write minimal implementation**

Create `pyimgano/services/robustness_service.py` with:

- `RobustnessRunRequest` dataclass
- `run_robustness_request(request)`
- narrow helpers for model resolution, corruption/input preparation, pixel postprocess creation, and notes/report payload assembly

Then migrate the execution branch in `pyimgano/robust_cli.py` to build a request object and delegate. Keep parser construction and final stdout/file formatting in the CLI layer.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_robustness_service.py tests/test_robust_cli_smoke.py -v
```

Expected: PASS.

### Task 2: Reuse Shared Discovery/JSON Helpers Where It Helps

**Files:**
- Modify: `pyimgano/robust_cli.py`
- Test: `tests/test_robust_cli_smoke.py`

**Step 1: Write the failing test**

Add one small seam assertion if needed to confirm `--list-models` reuses shared discovery helpers instead of reading the registry inline.

**Step 2: Run test to verify it fails**

Run the focused test only.

**Step 3: Write minimal implementation**

Prefer reusing `pyimgano.services.discovery_service.list_discovery_model_names()` if it keeps the adapter thinner without changing output shape.

**Step 4: Run tests to verify they pass**

Run the robustness CLI tests again.

### Task 3: Verification Sweep

**Files:**
- Modify: none
- Test: `tests/test_robustness_service.py`
- Test: `tests/test_robust_cli_smoke.py`
- Test: `tests/test_robustness_benchmark.py`
- Test: `tests/test_cli_pretrained_defaults_offline_safe_v12.py`

**Step 1: Run focused verification**

Run:

```bash
pytest --no-cov \
  tests/test_robustness_service.py \
  tests/test_robust_cli_smoke.py \
  tests/test_robustness_benchmark.py \
  tests/test_cli_pretrained_defaults_offline_safe_v12.py -v
```

Expected: PASS.
