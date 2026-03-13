# Workbench Multi-Category Execution Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` responsibility by moving all-category orchestration into a dedicated helper while preserving aggregate payloads.

**Architecture:** Add a `pyimgano.workbench.multi_category_execution` helper that owns category discovery, per-category execution fan-out, and aggregate report construction. Keep `runner.py` focused on runtime guardrails, run-context setup, single-category vs all-category dispatch, and top-level report persistence.

**Tech Stack:** Python, pytest, existing workbench category execution helper, dataset loader seam, aggregate report helper, `WorkbenchConfig`.

---

### Task 1: Lock The Multi-Category Execution Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_multi_category_execution.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper discovers categories, runs per-category execution, and builds the aggregate report
- `runner.py` delegates all-category orchestration to the helper instead of listing categories and looping inline
- the helper, not `runner.py`, is the module that depends on `dataset_loader` category listing and aggregate report construction

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_multi_category_execution.py tests/test_architecture_boundaries.py -k "multi_category_execution or runner_uses_multi_category_execution_boundary" -v
```

Expected: FAIL because `pyimgano.workbench.multi_category_execution` does not exist and `runner.py` still owns inline all-category orchestration.

### Task 2: Add The Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/multi_category_execution.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add `run_all_workbench_categories(config, recipe_name, run_dir)`
- move category discovery, per-category loop, and aggregate payload construction into the helper
- keep aggregate payload structure unchanged
- make `runner.py` delegate the all-category branch to the helper

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_multi_category_execution.py tests/test_workbench_runner_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/multi_category_execution.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_multi_category_execution.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_multi_category_execution.py tests/test_workbench_runner_smoke.py tests/test_workbench_manifest_smoke.py tests/test_workbench_runner_pixel_map_requirements.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-multi-category-execution-boundary.md pyimgano/workbench/multi_category_execution.py pyimgano/workbench/runner.py tests/test_workbench_multi_category_execution.py tests/test_architecture_boundaries.py
```

Expected: no output
