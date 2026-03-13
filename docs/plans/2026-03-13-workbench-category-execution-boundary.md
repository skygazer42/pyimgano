# Workbench Category Execution Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` responsibility by moving per-category execution orchestration into a dedicated helper while preserving payloads and artifacts.

**Architecture:** Add a `pyimgano.workbench.category_execution` helper that owns split preparation, detector setup, training, threshold calibration, inference, report assembly, and optional category artifact persistence for one category. Keep `runner.py` focused on runtime guardrails, run-context setup, single-category vs all-category dispatch, and aggregate report writing.

**Tech Stack:** Python, pytest, numpy, existing workbench boundary helpers, `WorkbenchConfig`.

---

### Task 1: Lock The Category Execution Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_category_execution.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper orchestrates split loading, training, threshold calibration, inference, report building, and category output persistence
- `runner.py` delegates category execution to the helper instead of owning `_run_category(...)`
- the helper, not `runner.py`, is the module that depends on the training, inference, detector setup, category report, runtime split, and category output boundaries

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_category_execution.py tests/test_architecture_boundaries.py -k "category_execution or runner_uses_category_execution_boundary" -v
```

Expected: FAIL because `pyimgano.workbench.category_execution` does not exist and `runner.py` still owns inline category execution orchestration.

### Task 2: Add The Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/category_execution.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add `run_workbench_category(config, recipe_name, category, run_dir)`
- move split preparation, detector/training/inference/report/output orchestration into the helper
- keep payload structure, calibration behavior, and output persistence unchanged
- make `runner.py` delegate per-category runs to the helper

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_category_execution.py tests/test_workbench_runner_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/category_execution.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_category_execution.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_category_execution.py tests/test_workbench_runner_smoke.py tests/test_workbench_manifest_smoke.py tests/test_workbench_runner_pixel_map_requirements.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-category-execution-boundary.md pyimgano/workbench/category_execution.py pyimgano/workbench/runner.py tests/test_workbench_category_execution.py tests/test_architecture_boundaries.py
```

Expected: no output
