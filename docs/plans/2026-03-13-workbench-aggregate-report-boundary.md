# Workbench Aggregate Report Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` aggregate-report responsibility by moving `category="all"` payload construction into a dedicated workbench helper.

**Architecture:** Add a `pyimgano.workbench.aggregate_report` helper that accepts `WorkbenchConfig`, recipe name, the resolved category list, and per-category payloads, then returns the stamped top-level aggregate payload. Move mean/std metric aggregation and top-level report field assembly into this helper while leaving category execution and run-dir persistence in `runner.py`.

**Tech Stack:** Python, NumPy, pytest, existing reporting schema helpers, `WorkbenchConfig`.

---

### Task 1: Lock The Aggregate Report Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_aggregate_report.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- the helper builds `categories`, `per_category`, `mean_metrics`, and `std_metrics`
- non-finite or missing metric values are ignored during averaging
- `runner.py` no longer owns `_safe_float`, aggregate metric loops, or inline aggregate payload assembly

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_aggregate_report.py tests/test_architecture_boundaries.py -k "workbench_aggregate_report or runner_uses_aggregate_report_boundary" -v
```

Expected: FAIL because the helper module does not exist and `runner.py` still owns aggregate payload logic inline.

### Task 2: Add The Aggregate Report Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/aggregate_report.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add `build_workbench_aggregate_report(...)`
- move mean/std calculation for `auroc` and `average_precision` into the helper
- move top-level aggregate payload assembly and report stamping into the helper
- keep `run_dir` attachment and final persistence in `runner.py`

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_aggregate_report.py tests/test_workbench_manifest_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/aggregate_report.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_aggregate_report.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_aggregate_report.py tests/test_workbench_manifest_smoke.py tests/test_workbench_runner_smoke.py tests/test_workbench_schema_version.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-aggregate-report-boundary.md pyimgano/workbench/aggregate_report.py pyimgano/workbench/runner.py tests/test_workbench_aggregate_report.py tests/test_architecture_boundaries.py
```

Expected: no output
