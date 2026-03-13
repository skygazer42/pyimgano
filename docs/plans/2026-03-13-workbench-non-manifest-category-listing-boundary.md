# Workbench Non-Manifest Category Listing Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.non_manifest_preflight` coupling by moving dataset category listing and list-failure issue emission into a dedicated helper boundary.

**Architecture:** Keep `pyimgano.workbench.non_manifest_preflight.run_non_manifest_preflight(...)` as the orchestrator that validates the source, delegates category listing, delegates requested-category selection, and assembles the final summary. Move `list_workbench_categories(...)`, `DATASET_CATEGORY_LIST_FAILED` issue emission, and the early failure summary into `pyimgano.workbench.non_manifest_category_listing`, returning the discovered categories plus an optional early summary payload.

**Tech Stack:** Python, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Non-Manifest Category Listing Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper returns the listed categories unchanged when category loading succeeds
- a dedicated helper emits `DATASET_CATEGORY_LIST_FAILED` and returns an early error summary when category loading raises
- `non_manifest_preflight.py` imports `pyimgano.workbench.non_manifest_category_listing`
- `non_manifest_preflight.py` calls `load_non_manifest_preflight_categories(...)`
- `non_manifest_preflight.py` no longer hosts the `DATASET_CATEGORY_LIST_FAILED` issue string or inline category-list try/except logic

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "non_manifest_category_listing or non_manifest_preflight_uses_category_listing_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `non_manifest_preflight.py` still hosts category-list failure handling inline.

### Task 2: Extract The Non-Manifest Category Listing Helper

**Files:**
- Create: `pyimgano/workbench/non_manifest_category_listing.py`
- Modify: `pyimgano/workbench/non_manifest_preflight.py`

**Step 1: Write minimal implementation**

- move category listing, `DATASET_CATEGORY_LIST_FAILED` issue emission, and early failure summary creation into `load_non_manifest_preflight_categories(...)`
- return the category list plus an optional summary payload
- keep `run_non_manifest_preflight(...)` result shape and issue codes/messages unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/non_manifest_preflight.py`
- Modify: `pyimgano/workbench/non_manifest_category_listing.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_non_manifest.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-non-manifest-category-listing-boundary.md pyimgano/workbench/non_manifest_preflight.py pyimgano/workbench/non_manifest_category_listing.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
