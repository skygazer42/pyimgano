# Benchmark/Discovery CLI Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin the remaining heavyweight benchmark/discovery CLI adapters by moving discovery payload assembly and benchmark orchestration into reusable service modules.

**Architecture:** Keep `pyimgano.cli`, `pyimgano.pyim_cli`, and related CLIs as parser/formatter adapters. Introduce a discovery service for registry/suite/sweep/category payloads and a benchmark service for benchmark/suite execution requests so orchestration is testable outside `argparse`.

**Tech Stack:** Python, argparse, dataclasses, pytest, existing `pyimgano.discovery`, registry helpers, pipeline runners, JSON reporting utilities.

---

### Task 1: Add a Discovery Service Seam

**Files:**
- Create: `pyimgano/services/discovery_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/pyim_cli.py`
- Test: `tests/test_discovery_service.py`
- Test: `tests/test_cli_discovery.py`
- Test: `tests/test_cli_feature_discovery.py`
- Test: `tests/test_cli_list_categories.py`
- Test: `tests/test_cli_list_categories_manifest.py`
- Test: `tests/test_cli_baseline_suites_v16.py`

**Step 1: Write the failing test**

Create `tests/test_discovery_service.py` with focused payload assertions:

```python
from __future__ import annotations

from pyimgano.services.discovery_service import (
    build_model_info_payload,
    list_discovery_model_names,
)


def test_list_discovery_model_names_supports_family_and_year_filters() -> None:
    names = list_discovery_model_names(family="patchcore", year="2021")
    assert isinstance(names, list)


def test_build_model_info_payload_returns_json_ready_shape() -> None:
    payload = build_model_info_payload("vision_ecod")
    assert payload["name"] == "vision_ecod"
    assert "accepted_kwargs" in payload
    assert "constructor" in payload
```

Add one CLI delegation assertion in `tests/test_cli_discovery.py`:

```python
def test_cli_list_models_delegates_to_discovery_service(monkeypatch, capsys):
    from pyimgano.cli import main
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **kwargs: ["delegated_model"],
    )

    rc = main(["--list-models"])
    assert rc == 0
    assert "delegated_model" in capsys.readouterr().out
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_discovery_service.py tests/test_cli_discovery.py -v
```

Expected: FAIL because `pyimgano.services.discovery_service` does not exist and `pyimgano.cli` still assembles discovery payloads inline.

**Step 3: Write minimal implementation**

Create `pyimgano/services/discovery_service.py` with stable APIs:

- `list_discovery_model_names(...)`
- `list_discovery_feature_names(...)`
- `list_dataset_categories_payload(...)`
- `build_model_info_payload(name)`
- `build_feature_info_payload(name)`
- `list_baseline_suites_payload()`
- `build_suite_info_payload(name)`
- `list_sweeps_payload()`
- `build_sweep_info_payload(name)`

Reuse existing `pyimgano.discovery`, registry, and baseline helpers. Return plain Python dict/list payloads only; no printing or `argparse`.

Then migrate `pyimgano/cli.py` discovery/list/info branches to call the service. Keep CLI text rendering local. Update `pyimgano/pyim_cli.py` to reuse the same service APIs where it overlaps instead of reassembling payloads.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov \
  tests/test_discovery_service.py \
  tests/test_cli_discovery.py \
  tests/test_cli_feature_discovery.py \
  tests/test_cli_list_categories.py \
  tests/test_cli_list_categories_manifest.py \
  tests/test_cli_baseline_suites_v16.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/services/discovery_service.py pyimgano/services/__init__.py pyimgano/cli.py pyimgano/pyim_cli.py tests/test_discovery_service.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_cli_list_categories.py tests/test_cli_list_categories_manifest.py tests/test_cli_baseline_suites_v16.py
git commit -m "refactor: extract discovery service from cli adapters"
```

---

### Task 2: Add a Benchmark Service for Single-Run Benchmark Mode

**Files:**
- Create: `pyimgano/services/benchmark_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/cli.py`
- Test: `tests/test_benchmark_service.py`
- Test: `tests/test_cli_smoke.py`
- Test: `tests/test_cli_config_file_v1.py`
- Test: `tests/test_cli_manifest_pixel_requires_masks.py`
- Test: `tests/test_cli_smoke_manifest_benchmark.py`

**Step 1: Write the failing test**

Create `tests/test_benchmark_service.py` with one request-to-pipeline delegation test:

```python
from __future__ import annotations

from pyimgano.services.benchmark_service import BenchmarkRunRequest, run_benchmark_request


def test_run_benchmark_request_delegates_to_pipeline(monkeypatch):
    import pyimgano.services.benchmark_service as benchmark_service

    calls = []

    monkeypatch.setattr(
        benchmark_service,
        "_run_benchmark_pipeline",
        lambda **kwargs: calls.append(kwargs) or {"ok": True},
    )

    request = BenchmarkRunRequest(
        dataset="custom",
        root="/tmp/custom",
        manifest_path=None,
        category="custom",
        model="vision_ecod",
        input_mode="paths",
        resize=(16, 16),
    )
    payload = run_benchmark_request(request)
    assert payload["ok"] is True
    assert calls[0]["model"] == "vision_ecod"
```

Add one CLI seam test proving benchmark mode calls the service instead of building the whole execution inline.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_benchmark_service.py tests/test_cli_smoke.py -v
```

Expected: FAIL because the benchmark service seam does not exist yet.

**Step 3: Write minimal implementation**

Create `pyimgano/services/benchmark_service.py` with:

- `BenchmarkRunRequest` dataclass
- `SuiteRunRequest` dataclass (optional in this task if shared setup helps)
- `run_benchmark_request(request)`
- small helpers for parsing suite include/exclude lists and building pixel postprocess objects from already-parsed values

The service should delegate to `pyimgano.pipelines.run_benchmark.run_benchmark` for non-pixel mode and keep request validation out of `cli.py` where practical.

Update `pyimgano/cli.py` so the single-model benchmark branch builds a request object and delegates execution to the service, while keeping stdout/file formatting in the CLI layer.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov \
  tests/test_benchmark_service.py \
  tests/test_cli_smoke.py \
  tests/test_cli_config_file_v1.py \
  tests/test_cli_manifest_pixel_requires_masks.py \
  tests/test_cli_smoke_manifest_benchmark.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/services/benchmark_service.py pyimgano/services/__init__.py pyimgano/cli.py tests/test_benchmark_service.py tests/test_cli_smoke.py tests/test_cli_config_file_v1.py tests/test_cli_manifest_pixel_requires_masks.py tests/test_cli_smoke_manifest_benchmark.py
git commit -m "refactor: add benchmark service seam for benchmark cli"
```

---

### Task 3: Add a Benchmark Service for Suite Mode

**Files:**
- Modify: `pyimgano/services/benchmark_service.py`
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_baseline_suites_v16.py`

**Step 1: Write the failing test**

Add a focused suite delegation test to `tests/test_cli_baseline_suites_v16.py`:

```python
def test_benchmark_cli_suite_mode_delegates_to_benchmark_service(monkeypatch, capsys):
    from pyimgano.cli import main
    import pyimgano.services.benchmark_service as benchmark_service

    monkeypatch.setattr(
        benchmark_service,
        "run_suite_request",
        lambda request: {"suite": request.suite, "rows": []},
    )

    rc = main(["--dataset", "custom", "--root", "/tmp/x", "--suite", "industrial-v1"])
    assert rc == 0
    assert '"suite": "industrial-v1"' in capsys.readouterr().out
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_cli_baseline_suites_v16.py -k delegates_to_benchmark_service -v
```

Expected: FAIL because suite mode is still assembled inline in `pyimgano.cli`.

**Step 3: Write minimal implementation**

Extend `pyimgano/services/benchmark_service.py` with:

- `SuiteRunRequest`
- `run_suite_request(request)`
- helpers for suite include/exclude normalization and suite export validation

Migrate the `--suite` branch in `pyimgano/cli.py` to build a request object and delegate to the service. Keep table export and final stdout/file formatting in CLI.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_cli_baseline_suites_v16.py tests/test_benchmark_service.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/services/benchmark_service.py pyimgano/cli.py tests/test_cli_baseline_suites_v16.py tests/test_benchmark_service.py
git commit -m "refactor: move suite execution orchestration into benchmark service"
```

---

### Task 4: Verification Sweep for Benchmark/Discovery Thinning

**Files:**
- Modify: none
- Test: `tests/test_discovery_service.py`
- Test: `tests/test_benchmark_service.py`
- Test: `tests/test_cli_discovery.py`
- Test: `tests/test_cli_feature_discovery.py`
- Test: `tests/test_cli_list_categories.py`
- Test: `tests/test_cli_list_categories_manifest.py`
- Test: `tests/test_cli_baseline_suites_v16.py`
- Test: `tests/test_cli_smoke.py`
- Test: `tests/test_cli_config_file_v1.py`
- Test: `tests/test_cli_smoke_manifest_benchmark.py`

**Step 1: Run focused verification**

Run:

```bash
pytest --no-cov \
  tests/test_discovery_service.py \
  tests/test_benchmark_service.py \
  tests/test_cli_discovery.py \
  tests/test_cli_feature_discovery.py \
  tests/test_cli_list_categories.py \
  tests/test_cli_list_categories_manifest.py \
  tests/test_cli_baseline_suites_v16.py \
  tests/test_cli_smoke.py \
  tests/test_cli_config_file_v1.py \
  tests/test_cli_smoke_manifest_benchmark.py -v
```

Expected: PASS.

**Step 2: Run broader smoke verification**

Run:

```bash
pytest --no-cov tests/test_cli_plugins_flag_v1.py tests/test_more_models_added.py -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git add .
git commit -m "test: verify benchmark and discovery cli thinning"
```
