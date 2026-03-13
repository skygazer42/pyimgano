# Workbench Runtime Split Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` split-shaping responsibility by moving runtime split preparation and `limit_train` / `limit_test` slicing into a dedicated workbench helper.

**Architecture:** Add a `pyimgano.workbench.runtime_split` module that accepts a loaded `WorkbenchSplit` plus `WorkbenchConfig` and returns a prepared runtime split with normalized lists/arrays and applied limits. Keep dataset loading in `dataset_loader.py`; this helper only handles the post-load runtime shaping used by the runner.

**Tech Stack:** Python, dataclasses, NumPy, pytest, existing `WorkbenchSplit` loader seam, `WorkbenchConfig`.

---

### Task 1: Lock The Runtime Split Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_runtime_split.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- the helper applies `limit_train` and `limit_test` consistently to train/calibration/test inputs
- test labels, masks, and metadata are sliced with test limits
- `runner.py` no longer owns inline `limit_train` / `limit_test` slicing

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_runtime_split.py tests/test_architecture_boundaries.py -k "workbench_runtime_split or runner_uses_runtime_split_boundary" -v
```

Expected: FAIL because the helper module does not exist and `runner.py` still owns inline split slicing.

### Task 2: Add The Runtime Split Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/runtime_split.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add a prepared runtime split dataclass
- add `prepare_workbench_runtime_split(config, split)`
- move list/array normalization and limit slicing into the helper
- refactor `runner.py` to consume the prepared split instead of slicing inline

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_runtime_split.py tests/test_workbench_runner_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/runtime_split.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_runtime_split.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_runtime_split.py tests/test_workbench_runner_smoke.py tests/test_workbench_manifest_smoke.py tests/test_workbench_report_dataset_summary.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-runtime-split-boundary.md pyimgano/workbench/runtime_split.py pyimgano/workbench/runner.py tests/test_workbench_runtime_split.py tests/test_architecture_boundaries.py
```

Expected: no output
