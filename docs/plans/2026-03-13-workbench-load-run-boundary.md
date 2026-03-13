# Workbench Load Run Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `pyimgano.workbench.load_run` into coherent run-artifact loading and checkpoint-restore boundaries while preserving the existing service and caller imports.

**Architecture:** Move config/report/category-selection/threshold/checkpoint-path helpers into `pyimgano.workbench.run_artifacts` and detector checkpoint restore logic into `pyimgano.workbench.checkpoint_restore`. Keep `pyimgano.workbench.load_run` as a thin compatibility facade that re-exports the public API so `workbench_run_service` and existing tests can keep importing the same functions.

**Tech Stack:** Python, pytest, string-based architecture boundary tests, existing serialization helpers.

---

### Task 1: Lock The Load-Run Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_run_artifacts.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated run-artifacts module hosts config/report/category-selection helpers
- a dedicated checkpoint-restore module hosts detector checkpoint restoration
- `pyimgano.workbench.load_run` acts as a compatibility facade instead of hosting JSON parsing and restore implementation inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_run_artifacts.py tests/test_architecture_boundaries.py -k "run_artifacts or load_run_module_uses_boundaries or checkpoint_restore" -v
```

Expected: FAIL because the dedicated modules do not exist yet and `load_run.py` still contains the monolithic implementation.

### Task 2: Extract Run Artifact And Checkpoint Restore Modules

**Files:**
- Create: `pyimgano/workbench/run_artifacts.py`
- Create: `pyimgano/workbench/checkpoint_restore.py`
- Modify: `pyimgano/workbench/load_run.py`

**Step 1: Write minimal implementation**

- move config/report/category/threshold/checkpoint-path helpers into `run_artifacts.py`
- move `load_checkpoint_into_detector(...)` into `checkpoint_restore.py`
- keep `pyimgano.workbench.load_run` exporting the same public API as a compatibility facade

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_run_artifacts.py tests/test_workbench_load_run_checkpoint.py tests/test_workbench_run_service.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/load_run.py`
- Modify: `pyimgano/workbench/run_artifacts.py`
- Modify: `pyimgano/workbench/checkpoint_restore.py`
- Modify: `tests/test_workbench_run_artifacts.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_run_artifacts.py tests/test_workbench_load_run_checkpoint.py tests/test_workbench_run_service.py tests/test_infer_context_service.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-load-run-boundary.md pyimgano/workbench/load_run.py pyimgano/workbench/run_artifacts.py pyimgano/workbench/checkpoint_restore.py tests/test_workbench_run_artifacts.py tests/test_architecture_boundaries.py
```

Expected: no output
