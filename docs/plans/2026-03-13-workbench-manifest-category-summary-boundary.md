# Workbench Manifest Category Summary Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_category_preflight` to a thinner orchestrator by moving manifest-category record filtering and explicit split counting into a dedicated summary helper.

**Architecture:** Keep `pyimgano.workbench.manifest_category_preflight.preflight_manifest_category(...)` as the category-level assembler that coordinates summary, path inspection, and assignment analysis. Move `ManifestRecord` filtering plus explicit split and explicit test-label counting into `pyimgano.workbench.manifest_category_summary`, returning both the filtered records and the count summary so downstream helpers can reuse the normalized record set without duplicating record-shape assumptions.

**Tech Stack:** Python, pytest, manifest dataset utilities, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Category Summary Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper can filter manifest records and build the explicit split summary
- `manifest_category_preflight.py` imports `pyimgano.workbench.manifest_category_summary`
- `manifest_category_preflight.py` calls `summarize_manifest_category_records(...)`
- `manifest_category_preflight.py` no longer hosts the explicit count loop inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "manifest_category_summary or manifest_category_preflight_module_hosts_category_analysis or manifest_category_preflight_uses_summary_helper_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `manifest_category_preflight.py` still contains the explicit counting loop.

### Task 2: Extract The Manifest Category Summary Helper

**Files:**
- Create: `pyimgano/workbench/manifest_category_summary.py`
- Modify: `pyimgano/workbench/manifest_category_preflight.py`

**Step 1: Write minimal implementation**

- move `ManifestRecord` filtering and explicit split / explicit test-label counting into `summarize_manifest_category_records(...)`
- return the filtered records and summary counts in a helper-friendly structure
- keep `preflight_manifest_category(...)` output shape unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_category_preflight.py`
- Modify: `pyimgano/workbench/manifest_category_summary.py`
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
git diff --check -- docs/plans/2026-03-13-workbench-manifest-category-summary-boundary.md pyimgano/workbench/manifest_category_preflight.py pyimgano/workbench/manifest_category_summary.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
