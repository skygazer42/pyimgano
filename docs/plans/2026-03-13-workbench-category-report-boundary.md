# Workbench Category Report Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` report-assembly responsibility by moving category-level payload construction into a dedicated workbench report builder.

**Architecture:** Add a `pyimgano.workbench.category_report` helper that accepts normalized runtime facts for one category and returns the stamped category payload. Move dataset-summary calculation, threshold provenance, pixel-metric status, and optional training/checkpoint metadata into this pure boundary while leaving top-level aggregate report assembly in `runner.py`.

**Tech Stack:** Python, dataclasses, NumPy, pytest, existing reporting schema helpers, `WorkbenchConfig`.

---

### Task 1: Lock The Category Report Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_category_report.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a category-report helper builds the expected payload fields and schema stamp
- pixel-metric status follows manifest skip reasons and missing-mask fallback rules
- `runner.py` no longer assembles category payload fields inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_category_report.py tests/test_architecture_boundaries.py -k "workbench_category_report or runner_uses_category_report_boundary" -v
```

Expected: FAIL because the helper module does not exist and `runner.py` still owns the category payload assembly.

### Task 2: Add The Category Report Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/category_report.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add a small dataclass for category report inputs
- add `build_workbench_category_report(...)`
- move dataset-summary creation, threshold provenance, optional pixel status, optional training/checkpoint attachment, and report stamping into the helper
- keep top-level aggregate report behavior unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_category_report.py tests/test_workbench_report_dataset_summary.py tests/test_workbench_schema_version.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/category_report.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_category_report.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_category_report.py tests/test_workbench_report_dataset_summary.py tests/test_workbench_schema_version.py tests/test_workbench_manifest_smoke.py tests/test_workbench_repro_provenance.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-category-report-boundary.md pyimgano/workbench/category_report.py pyimgano/workbench/runner.py tests/test_workbench_category_report.py tests/test_architecture_boundaries.py
```

Expected: no output
