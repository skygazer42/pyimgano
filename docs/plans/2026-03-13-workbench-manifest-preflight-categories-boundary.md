# Workbench Manifest Preflight Categories Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_preflight` orchestration complexity by moving per-category record filtering and category preflight dispatch into a dedicated helper boundary.

**Architecture:** Keep `pyimgano.workbench.manifest_preflight.run_manifest_preflight(...)` as the top-level manifest preflight orchestrator that validates the source, builds the split policy, loads records, handles empty manifests, selects categories, and assembles the final report. Move the `for cat in categories` loop, category-specific record filtering, and `preflight_manifest_category(...)` delegation into `pyimgano.workbench.manifest_preflight_categories`, returning the assembled `per_category` mapping.

**Tech Stack:** Python, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Category-Batch Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_workbench_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper filters records by category and delegates each category to `preflight_manifest_category(...)`
- the helper preserves the selected category order and passes empty record lists through when a requested category has no records
- `manifest_preflight.py` imports and uses `preflight_manifest_categories(...)`
- `manifest_preflight.py` no longer hosts the `for cat in categories` loop or inline `cat_records = ...` filtering logic

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py -k "manifest_preflight_categories or manifest_preflight_uses_category_batch_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `manifest_preflight.py` still filters records and loops inline.

### Task 2: Extract The Manifest Category-Batch Helper

**Files:**
- Create: `pyimgano/workbench/manifest_preflight_categories.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`

**Step 1: Write minimal implementation**

- move per-category record filtering and `preflight_manifest_category(...)` delegation into `preflight_manifest_categories(...)`
- return the `per_category` mapping without changing report shape or issue behavior
- keep `MANIFEST_EMPTY` handling and report assembly in `manifest_preflight.py`

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/manifest_preflight_categories.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_workbench_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_workbench_dataset_loader.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-manifest-preflight-categories-boundary.md pyimgano/workbench/manifest_preflight.py pyimgano/workbench/manifest_preflight_categories.py tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py
```

Expected: no output
