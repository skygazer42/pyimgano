# Workbench Category Output Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` I/O coupling by extracting category-level report, anomaly-map, and per-image JSONL persistence into a dedicated workbench boundary.

**Architecture:** Add a `pyimgano.workbench.category_outputs` helper module that accepts the finalized category payload plus inference artifacts and writes them into the run directory. Keep checkpoint writing and top-level run report creation in `runner.py` for now; only the category-output tail moves out.

**Tech Stack:** Python, dataclasses, NumPy, pytest, existing reporting helpers, `build_workbench_run_paths`, workbench map persistence.

---

### Task 1: Lock The Output Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_category_outputs.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a category-output helper writes `categories/<cat>/report.json`
- anomaly maps are persisted under `artifacts/maps`
- per-image JSONL records include metadata and saved anomaly-map references
- `runner.py` no longer writes anomaly maps or per-image JSONL inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_category_outputs.py tests/test_architecture_boundaries.py -k "workbench_category_outputs or runner_uses_category_output_boundary" -v
```

Expected: FAIL because the helper module does not exist and `runner.py` still owns the category-output persistence logic.

### Task 2: Add The Category Output Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/category_outputs.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add a small dataclass for finalized category-output persistence inputs
- add `save_workbench_category_outputs(...)`
- move category report writing, anomaly-map saving, and per-image JSONL generation into the helper
- keep file layout and payload structure unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_category_outputs.py tests/test_workbench_runner_smoke.py tests/test_workbench_manifest_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/category_outputs.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_category_outputs.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_category_outputs.py tests/test_workbench_runner_smoke.py tests/test_workbench_manifest_smoke.py tests/test_workbench_runner_checkpoints.py tests/test_workbench_report_dataset_summary.py tests/test_recipe_industrial_adapt_maps.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-category-output-boundary.md pyimgano/workbench/category_outputs.py pyimgano/workbench/runner.py tests/test_workbench_category_outputs.py tests/test_architecture_boundaries.py
```

Expected: no output
