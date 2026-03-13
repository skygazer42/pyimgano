# Workbench Manifest Record Preflight Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_preflight` branching by moving manifest record loading and empty-manifest early-failure handling into the existing record-preflight boundary.

**Architecture:** Keep `pyimgano.workbench.manifest_preflight.run_manifest_preflight(...)` as the top-level orchestrator that validates the source, builds the split policy, resolves record-preflight output, selects categories, dispatches per-category analysis, and assembles the final report. Extend `pyimgano.workbench.manifest_record_preflight` with a higher-level helper that wraps `load_manifest_records_best_effort(...)`, emits `MANIFEST_EMPTY` when no valid records remain, and returns the loaded records/categories plus an optional early summary.

**Tech Stack:** Python, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Record-Preflight Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_workbench_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper delegates to `load_manifest_records_best_effort(...)` and returns records/categories when valid rows exist
- the helper emits `MANIFEST_EMPTY` and returns an early failure summary when no valid records remain
- `manifest_preflight.py` imports and uses `resolve_manifest_preflight_records(...)`
- `manifest_preflight.py` no longer hosts the `MANIFEST_EMPTY` issue string or inline `if not records:` branching

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py -k "manifest_record_preflight or record_preflight_boundary" -v
```

Expected: FAIL because `resolve_manifest_preflight_records(...)` does not exist yet and `manifest_preflight.py` still hosts empty-manifest handling inline.

### Task 2: Extend The Manifest Record-Preflight Boundary

**Files:**
- Modify: `pyimgano/workbench/manifest_record_preflight.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`

**Step 1: Write minimal implementation**

- add `resolve_manifest_preflight_records(...)` to wrap `load_manifest_records_best_effort(...)`
- emit `MANIFEST_EMPTY` and return `{"manifest_path": ..., "manifest": {"ok": False}}` when no valid records remain
- keep the returned records/categories and manifest preflight report shape unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_record_preflight.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`
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
git diff --check -- docs/plans/2026-03-13-workbench-manifest-record-preflight-boundary.md pyimgano/workbench/manifest_record_preflight.py pyimgano/workbench/manifest_preflight.py tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py
```

Expected: no output
