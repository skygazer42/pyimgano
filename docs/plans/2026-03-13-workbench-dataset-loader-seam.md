# Workbench Dataset Loader Seam Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` responsibility by extracting dataset split loading into a dedicated workbench loader seam.

**Architecture:** Introduce a `pyimgano.workbench.dataset_loader` module that converts `WorkbenchConfig` plus category into one normalized `WorkbenchSplit` payload. Refactor `runner.py` to consume that payload instead of branching on `manifest`, benchmark datasets, and `numpy` inputs inline.

**Tech Stack:** Python, dataclasses, NumPy, pytest, existing benchmark split loaders, manifest split loaders, `WorkbenchConfig`.

---

### Task 1: Lock The Loader Seam With Focused Tests

**Files:**
- Create: `tests/test_workbench_dataset_loader.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.workbench.dataset_loader` exposes a normalized split dataclass and loader helper
- manifest path-mode loading preserves calibration paths, pixel skip reason, and test metadata
- numpy-mode loading normalizes arrays into list inputs and sets `input_format="rgb_u8_hwc"`
- `runner.py` no longer owns manifest/benchmark/numpy branching inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_dataset_loader.py tests/test_architecture_boundaries.py -k "workbench_dataset_loader or runner_uses_dataset_loader_boundary" -v
```

Expected: FAIL because `pyimgano.workbench.dataset_loader` does not exist and `runner.py` still imports and branches on split loaders directly.

### Task 2: Add The Loader And Refactor Runner To Use It

**Files:**
- Create: `pyimgano/workbench/dataset_loader.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `pyimgano/workbench/__init__.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write minimal implementation**

- Add a `WorkbenchSplit` dataclass with normalized fields for train/calibration/test inputs, labels, masks, input format, pixel metric status, and optional metadata
- Add `load_workbench_split(config, category, load_masks)` that encapsulates:
  - benchmark dataset path-mode loading
  - manifest path-mode loading with calibration fallback
  - numpy-mode loading through `pyimgano.datasets.load_dataset`
- Refactor `runner.py` to call the loader helper and consume the normalized payload
- Keep limit handling, training, inference, report writing, and output semantics unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_dataset_loader.py tests/test_workbench_report_dataset_summary.py tests/test_integration_workbench_train_then_infer.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/dataset_loader.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_dataset_loader.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_dataset_loader.py tests/test_workbench_report_dataset_summary.py tests/test_workbench_preflight_manifest.py tests/test_integration_workbench_train_then_infer.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-dataset-loader-seam.md pyimgano/workbench/dataset_loader.py pyimgano/workbench/runner.py pyimgano/workbench/__init__.py tests/test_workbench_dataset_loader.py tests/test_architecture_boundaries.py
```

Expected: no output
