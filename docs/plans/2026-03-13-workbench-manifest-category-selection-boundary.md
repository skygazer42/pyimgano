# Workbench Manifest Category Selection Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_preflight` branching by moving manifest requested-category selection and empty-category issue emission into a dedicated helper boundary.

**Architecture:** Keep `pyimgano.workbench.manifest_preflight.run_manifest_preflight(...)` as the orchestrator that validates the source, loads records, computes split policy, runs per-category analysis, and assembles the final payload. Move `"all"` handling, category sorting, requested-category selection, and `MANIFEST_CATEGORY_EMPTY` issue emission into `pyimgano.workbench.manifest_category_selection`, returning both the selected category list and a flag describing whether the request expanded to all categories.

**Tech Stack:** Python, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Category Selection Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper sorts all available categories when requested category is `all`
- a dedicated helper returns the single requested category and emits `MANIFEST_CATEGORY_EMPTY` when missing
- `manifest_preflight.py` imports `pyimgano.workbench.manifest_category_selection`
- `manifest_preflight.py` calls `select_manifest_preflight_categories(...)`
- `manifest_preflight.py` no longer hosts the `MANIFEST_CATEGORY_EMPTY` issue string or direct `requested_category.lower() == "all"` selection logic inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "manifest_category_selection or manifest_preflight_uses_category_selection_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `manifest_preflight.py` still performs category selection inline.

### Task 2: Extract The Manifest Category Selection Helper

**Files:**
- Create: `pyimgano/workbench/manifest_category_selection.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`

**Step 1: Write minimal implementation**

- move requested-category selection, `all` expansion, category sorting, and empty-category issue emission into `select_manifest_preflight_categories(...)`
- return the selected categories plus a boolean describing whether the request expanded to all categories
- keep `run_manifest_preflight(...)` result shape unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/manifest_category_selection.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-manifest-category-selection-boundary.md pyimgano/workbench/manifest_preflight.py pyimgano/workbench/manifest_category_selection.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
