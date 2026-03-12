# Infer Workbench Run Service Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce cross-package coupling by routing infer services through a service-layer adapter for workbench run loading helpers instead of importing `pyimgano.workbench.load_run` directly.

**Architecture:** Add a small `pyimgano.services.workbench_run_service` adapter that re-exports the run-loading helpers needed by infer services. Refactor `infer_context_service.py` and `infer_setup_service.py` to depend on that service boundary, then lock the rule with architecture tests so infer orchestration no longer reaches into `pyimgano.workbench` internals directly.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano/services/infer_context_service.py`, `pyimgano/services/infer_setup_service.py`, `pyimgano/workbench/load_run.py`, `tests/test_architecture_boundaries.py`

---

### Task 1: Lock The New Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_run_service.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.services.workbench_run_service` exposes the intended helper surface
- the new adapter delegates to `pyimgano.workbench.load_run`
- `infer_context_service.py` and `infer_setup_service.py` no longer import `pyimgano.workbench.load_run` directly

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_workbench_run_service.py tests/test_architecture_boundaries.py -v`
Expected: FAIL because `workbench_run_service.py` does not yet exist and infer services still import `pyimgano.workbench.load_run` directly.

### Task 2: Add The Adapter And Move Infer Services To It

**Files:**
- Create: `pyimgano/services/workbench_run_service.py`
- Modify: `pyimgano/services/infer_context_service.py`
- Modify: `pyimgano/services/infer_setup_service.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write minimal implementation**

- Add a thin service adapter around the workbench run-loading helpers used by infer services
- Refactor infer services to import those helpers from `pyimgano.services.workbench_run_service`
- Update service import whitelist expectations for the new service dependency
- Preserve infer behavior, checkpoints, thresholds, and error messages exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_workbench_run_service.py tests/test_architecture_boundaries.py tests/test_infer_context_service.py tests/test_infer_setup_service.py -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-infer-workbench-run-service-boundary.md`
- Create: `pyimgano/services/workbench_run_service.py`
- Modify: `pyimgano/services/infer_context_service.py`
- Modify: `pyimgano/services/infer_setup_service.py`
- Modify: `tests/test_workbench_run_service.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_workbench_run_service.py tests/test_infer_context_service.py tests/test_infer_setup_service.py tests/test_infer_continue_service.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-infer-workbench-run-service-boundary.md pyimgano/services/workbench_run_service.py pyimgano/services/infer_context_service.py pyimgano/services/infer_setup_service.py tests/test_workbench_run_service.py tests/test_architecture_boundaries.py`
Expected: no output
