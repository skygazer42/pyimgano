# Workbench Category Listing Seam Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove category-listing dataset special cases from `pyimgano.workbench.runner` by routing the `category="all"` flow through the existing workbench dataset loader boundary.

**Architecture:** Extend `pyimgano.workbench.dataset_loader` with a narrow helper that resolves workbench categories from `WorkbenchConfig`. Reuse `pyimgano.datasets.catalog.list_dataset_categories(...)` so manifest and benchmark datasets already share one source of truth, then refactor `runner.py` to depend only on the workbench seam.

**Tech Stack:** Python, pytest, `WorkbenchConfig`, `pyimgano.datasets.catalog`, existing workbench runner tests.

---

### Task 1: Lock The Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_dataset_loader.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.workbench.dataset_loader` exposes `list_workbench_categories(...)`
- the helper delegates manifest and non-manifest listing through the dataset catalog boundary with the right arguments
- `runner.py` no longer directly references `list_manifest_categories` or `list_dataset_categories`

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_dataset_loader.py tests/test_architecture_boundaries.py -k "list_workbench_categories or runner_uses_dataset_loader_boundary" -v
```

Expected: FAIL because the helper does not exist and `runner.py` still owns direct category-listing imports.

### Task 2: Add The Category Listing Seam

**Files:**
- Modify: `pyimgano/workbench/dataset_loader.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `pyimgano/workbench/__init__.py`

**Step 1: Write minimal implementation**

- add `list_workbench_categories(config)` to `dataset_loader.py`
- delegate to `pyimgano.datasets.catalog.list_dataset_categories(...)`
- pass `manifest_path` when the workbench dataset is manifest
- refactor `runner.py` to use the helper in the `category="all"` path
- keep report aggregation and output semantics unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_dataset_loader.py tests/test_workbench_manifest_smoke.py tests/test_workbench_runner_smoke.py tests/test_architecture_boundaries.py -v
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
pytest --no-cov tests/test_workbench_dataset_loader.py tests/test_workbench_manifest_smoke.py tests/test_workbench_runner_smoke.py tests/test_recipe_industrial_adapt_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-category-listing-seam.md pyimgano/workbench/dataset_loader.py pyimgano/workbench/runner.py pyimgano/workbench/__init__.py tests/test_workbench_dataset_loader.py tests/test_architecture_boundaries.py
```

Expected: no output
