# Workbench Run Report Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` responsibility by moving top-level run report persistence into a dedicated helper while preserving payload shape and report artifacts.

**Architecture:** Add a `pyimgano.workbench.run_report` helper that injects `run_dir` into the final payload and writes the top-level `report.json` when run paths exist. Keep `runner.py` focused on orchestration only, and include focused regression coverage for the existing `build_infer_config_payload(...)` entrypoint.

**Tech Stack:** Python, pytest, existing report serializer, `WorkbenchRunPaths`, `WorkbenchConfig`.

---

### Task 1: Lock The Run Report Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_run_report.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper injects `run_dir` and writes the top-level `report.json`
- the helper becomes a no-op when run paths are absent
- `runner.py` delegates final report persistence to the helper instead of calling `save_run_report(...)` directly

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_run_report.py tests/test_architecture_boundaries.py -k "run_report or runner_uses_run_report_boundary" -v
```

Expected: FAIL because `pyimgano.workbench.run_report` does not exist and `runner.py` still persists reports inline.

### Task 2: Add The Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/run_report.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add `persist_workbench_run_report(payload, paths)`
- inject `run_dir` into a copied payload and write `report.json` when paths exist
- return the copied payload unchanged when paths are absent
- make `runner.py` delegate final report persistence to the helper
- restore any required imports for `build_infer_config_payload(...)`

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_run_report.py tests/test_workbench_preprocessing_config.py tests/test_workbench_feature_pipeline.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/run_report.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_run_report.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_run_report.py tests/test_workbench_preprocessing_config.py tests/test_workbench_feature_pipeline.py tests/test_workbench_runner_smoke.py tests/test_workbench_manifest_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-run-report-boundary.md pyimgano/workbench/run_report.py pyimgano/workbench/runner.py tests/test_workbench_run_report.py tests/test_architecture_boundaries.py
```

Expected: no output
