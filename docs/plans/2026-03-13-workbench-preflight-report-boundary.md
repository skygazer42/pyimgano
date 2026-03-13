# Workbench Preflight Report Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move final `PreflightReport` assembly out of `pyimgano.workbench.preflight` into a dedicated report helper boundary while preserving the current report shape and imports.

**Architecture:** Keep `pyimgano.workbench.preflight.run_preflight(...)` as the runtime orchestrator that gathers dataset/category context, accumulates issues, runs model compatibility checks, and resolves the dataset-specific summary. Add `pyimgano.workbench.preflight_report.build_preflight_report(...)` to own `PreflightReport(...)` construction so the entrypoint no longer mixes orchestration with result-object assembly.

**Tech Stack:** Python, dataclasses, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Preflight Report Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a new `preflight_report` helper builds `PreflightReport` objects with the expected dataset/category/summary/issues fields
- `preflight.py` imports and uses `build_preflight_report(...)`
- `preflight.py` no longer constructs `PreflightReport(...)` inline
- `preflight_report.py` hosts the final `PreflightReport(...)` assembly logic

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py -k "preflight_report or preflight_uses_report_builder_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `preflight.py` still assembles the `PreflightReport` inline.

### Task 2: Add The Report Helper And Delegate To It

**Files:**
- Create: `pyimgano/workbench/preflight_report.py`
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Write minimal implementation**

- add `build_preflight_report(...) -> PreflightReport`
- move the final report assembly into that helper
- keep `run_preflight(...)` return behavior-compatible

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Create: `pyimgano/workbench/preflight_report.py`
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_package.py tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-preflight-report-boundary.md pyimgano/workbench/preflight.py pyimgano/workbench/preflight_report.py tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py
```

Expected: no output
