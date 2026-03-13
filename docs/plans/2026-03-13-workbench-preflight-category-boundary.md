# Workbench Preflight Category Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make workbench preflight reuse the same category-listing seam as the runner so dataset enumeration logic has one workbench boundary.

**Architecture:** Reuse `pyimgano.workbench.dataset_loader.list_workbench_categories(...)` inside non-manifest preflight. Keep root existence checks and custom dataset structure validation in `preflight.py`, but remove the direct dependency on `pyimgano.datasets.catalog`.

**Tech Stack:** Python, pytest, `WorkbenchConfig`, workbench preflight tests, architecture boundary tests.

---

### Task 1: Lock The Preflight Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_preflight_non_manifest.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- non-manifest preflight reads categories from `list_workbench_categories(...)`
- `pyimgano.workbench.preflight` no longer directly references `list_dataset_categories`

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -k "preflight_non_manifest_uses_workbench_category_boundary or preflight_uses_workbench_dataset_loader_boundary" -v
```

Expected: FAIL because `preflight.py` still imports `list_dataset_categories` directly.

### Task 2: Refactor Preflight To Use The Workbench Boundary

**Files:**
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Write minimal implementation**

- import `list_workbench_categories` from `pyimgano.workbench.dataset_loader`
- replace the direct dataset catalog call in `_preflight_non_manifest(...)`
- keep error handling, root checks, and custom dataset validation unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_preflight_non_manifest.py tests/test_workbench_preflight_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `tests/test_workbench_preflight_non_manifest.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_preflight_non_manifest.py tests/test_workbench_preflight_manifest.py tests/test_train_cli_preflight.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-preflight-category-boundary.md pyimgano/workbench/preflight.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py
```

Expected: no output
