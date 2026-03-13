# Workbench Non-Manifest Preflight Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.preflight` responsibility by moving non-manifest dataset validation into a dedicated helper while preserving issue codes and summary shape.

**Architecture:** Add a `pyimgano.workbench.non_manifest_preflight` helper that owns root existence checks, custom dataset structure validation, and category discovery for non-manifest datasets. Keep `preflight.py` as the entrypoint that routes between manifest and non-manifest helpers and owns `PreflightIssue` construction.

**Tech Stack:** Python, pytest, existing `WorkbenchConfig`, workbench dataset loader seam, custom dataset validator.

---

### Task 1: Lock The Non-Manifest Preflight Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper returns the expected non-manifest summary shape and uses the category loader seam
- `preflight.py` delegates non-manifest validation to the helper instead of owning `_preflight_non_manifest(...)`
- the new helper, not `preflight.py`, is the module that depends on `dataset_loader`

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py -k "non_manifest_preflight or preflight_uses_non_manifest_preflight_boundary" -v
```

Expected: FAIL because `pyimgano.workbench.non_manifest_preflight` does not exist and `preflight.py` still owns inline non-manifest validation.

### Task 2: Add The Helper And Refactor Preflight

**Files:**
- Create: `pyimgano/workbench/non_manifest_preflight.py`
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Write minimal implementation**

- add `run_non_manifest_preflight(config, issues, issue_builder)`
- move root checks, custom dataset structure validation, and category resolution into the helper
- keep issue codes/messages and summary shape stable
- make `preflight.py` delegate to the helper

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/non_manifest_preflight.py`
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_non_manifest.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-non-manifest-preflight-boundary.md pyimgano/workbench/non_manifest_preflight.py pyimgano/workbench/preflight.py tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py
```

Expected: no output
