# Workbench Inference Runtime Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` inference and evaluation coupling by moving infer/evaluate orchestration into a dedicated workbench runtime helper.

**Architecture:** Add a `pyimgano.workbench.inference_runtime` module that accepts the detector, test inputs, postprocess settings, threshold, and optional masks, then returns normalized scores, maps, and evaluation results. Move anomaly-map resizing for pixel metrics and `evaluate_detector(...)` invocation into this helper while keeping threshold calibration in `runner.py`.

**Tech Stack:** Python, dataclasses, NumPy, pytest, existing `infer(...)`, `evaluate_detector(...)`, OpenCV resizing.

---

### Task 1: Lock The Inference Runtime Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_inference_runtime.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- the helper requests anomaly maps when needed and resizes them to mask shape before pixel evaluation
- the helper skips map collection when neither saved maps nor postprocess require them
- `runner.py` no longer calls `infer(...)`, `evaluate_detector(...)`, or owns `_maybe_resize_maps_to_masks(...)` inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_inference_runtime.py tests/test_architecture_boundaries.py -k "workbench_inference_runtime or runner_uses_inference_runtime_boundary" -v
```

Expected: FAIL because the helper module does not exist and `runner.py` still owns inline inference/evaluation logic.

### Task 2: Add The Inference Runtime Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/inference_runtime.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add a small dataclass for normalized inference outputs
- add `run_workbench_inference(...)`
- move `include_maps` computation, `infer(...)`, pixel-score resizing, and `evaluate_detector(...)` into the helper
- keep payload/report generation unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_inference_runtime.py tests/test_workbench_runner_smoke.py tests/test_recipe_industrial_adapt_maps.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/inference_runtime.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_inference_runtime.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_inference_runtime.py tests/test_workbench_runner_smoke.py tests/test_recipe_industrial_adapt_maps.py tests/test_workbench_manifest_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-inference-runtime-boundary.md pyimgano/workbench/inference_runtime.py pyimgano/workbench/runner.py tests/test_workbench_inference_runtime.py tests/test_architecture_boundaries.py
```

Expected: no output
