# Dataset Split Service Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce split-loading duplication by routing benchmark-style dataset split loading through a shared service boundary.

**Architecture:** Add a `pyimgano.services.dataset_split_service` module that normalizes benchmark and manifest split loading into one result type with optional `pixel_skip_reason`. Refactor `benchmark_service.py` and `robustness_service.py` to depend on this adapter instead of importing `pyimgano.datasets.manifest` and `pyimgano.pipelines.mvtec_visa` directly.

**Tech Stack:** Python, dataclasses, NumPy, pytest, existing benchmark split loaders, existing manifest split policy/config semantics.

---

### Task 1: Lock The Service Boundary With Failing Tests

**Files:**
- Create: `tests/test_dataset_split_service.py`
- Modify: `tests/test_architecture_boundaries.py`
- Modify: `tests/test_robustness_service.py`

**Step 1: Write the failing test**

Add tests that prove:
- `dataset_split_service` can load plain benchmark splits and manifest-backed benchmark splits through one API
- the service preserves `pixel_skip_reason` for manifest-backed loads
- `benchmark_service.py` and `robustness_service.py` no longer import split loaders directly
- `robustness_service` delegates split loading through the new service seam

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_dataset_split_service.py tests/test_robustness_service.py tests/test_architecture_boundaries.py -k "dataset_split_service or split_service_boundary or robustness_request_delegates_split_loading" -v
```

Expected: FAIL because `pyimgano.services.dataset_split_service` does not exist and the old direct imports are still present.

### Task 2: Add The Service And Refactor Call Sites

**Files:**
- Create: `pyimgano/services/dataset_split_service.py`
- Modify: `pyimgano/services/benchmark_service.py`
- Modify: `pyimgano/services/robustness_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `tests/test_architecture_boundaries.py`
- Modify: `tests/test_robustness_service.py`

**Step 1: Write minimal implementation**

- Add a `LoadedBenchmarkSplit` dataclass with `split` and `pixel_skip_reason`
- Add a `load_benchmark_style_split(...)` helper that:
  - loads plain benchmark datasets via `pyimgano.pipelines.mvtec_visa.load_benchmark_split`
  - loads manifest-backed benchmark datasets via `pyimgano.datasets.manifest.load_manifest_benchmark_split`
  - preserves existing split policy semantics for seed and `manifest_test_normal_fraction`
- Refactor `benchmark_service` pixel mode and `robustness_service` split loading to use the service
- Preserve payload shapes and existing error behavior

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_dataset_split_service.py tests/test_benchmark_service.py tests/test_robustness_service.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/services/dataset_split_service.py`
- Modify: `pyimgano/services/benchmark_service.py`
- Modify: `pyimgano/services/robustness_service.py`
- Modify: `tests/test_dataset_split_service.py`
- Modify: `tests/test_benchmark_service.py`
- Modify: `tests/test_robustness_service.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_dataset_split_service.py tests/test_benchmark_service.py tests/test_robustness_service.py tests/test_robust_cli_smoke.py tests/test_cli_smoke.py tests/test_architecture_boundaries.py tests/test_services_package.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-dataset-split-service-boundary.md pyimgano/services/dataset_split_service.py pyimgano/services/benchmark_service.py pyimgano/services/robustness_service.py pyimgano/services/__init__.py tests/test_dataset_split_service.py tests/test_benchmark_service.py tests/test_robustness_service.py tests/test_architecture_boundaries.py
```

Expected: no output
