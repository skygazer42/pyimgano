# Infer Runtime Workbench Adaptation Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce infer/runtime coupling to the workbench package by moving workbench postprocess construction behind a service-layer adapter.

**Architecture:** Add a narrow `pyimgano.services.workbench_adaptation_service` adapter that accepts infer-facing payload dicts and delegates the actual `MapPostprocessConfig` construction and `build_postprocess()` call to `pyimgano.workbench.adaptation`. Refactor `infer_runtime_service.py` to depend on that service boundary and lock the rule with tests so infer runtime orchestration no longer imports `pyimgano.workbench.adaptation` directly.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano/services/infer_runtime_service.py`, `pyimgano/workbench/adaptation.py`, `tests/test_infer_runtime_service.py`, `tests/test_architecture_boundaries.py`

---

### Task 1: Lock The New Adaptation Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_adaptation_service.py`
- Modify: `tests/test_architecture_boundaries.py`
- Modify: `tests/test_infer_runtime_service.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.services.workbench_adaptation_service` exposes the intended helper surface
- the adapter delegates payload coercion to `pyimgano.workbench.adaptation`
- `infer_runtime_service.py` no longer imports `pyimgano.workbench.adaptation` directly
- infer runtime planning delegates infer-config postprocess payload handling through the new service boundary

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_workbench_adaptation_service.py tests/test_infer_runtime_service.py tests/test_architecture_boundaries.py -v`
Expected: FAIL because the new service does not yet exist and `infer_runtime_service.py` still imports `pyimgano.workbench.adaptation` directly.

### Task 2: Add The Adapter And Move Infer Runtime To It

**Files:**
- Create: `pyimgano/services/workbench_adaptation_service.py`
- Modify: `pyimgano/services/infer_runtime_service.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write minimal implementation**

- Add a thin service adapter that converts infer postprocess payload dicts into workbench adaptation calls
- Refactor `infer_runtime_service.py` to use that service adapter instead of importing workbench adaptation directly
- Update service whitelist expectations for the new dependency
- Preserve infer runtime outputs, pixel threshold behavior, and postprocess semantics exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_workbench_adaptation_service.py tests/test_infer_runtime_service.py tests/test_architecture_boundaries.py -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-infer-runtime-workbench-adaptation-boundary.md`
- Create: `pyimgano/services/workbench_adaptation_service.py`
- Modify: `pyimgano/services/infer_runtime_service.py`
- Modify: `tests/test_workbench_adaptation_service.py`
- Modify: `tests/test_infer_runtime_service.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_workbench_adaptation_service.py tests/test_infer_runtime_service.py tests/test_infer_continue_service.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-infer-runtime-workbench-adaptation-boundary.md pyimgano/services/workbench_adaptation_service.py pyimgano/services/infer_runtime_service.py tests/test_workbench_adaptation_service.py tests/test_infer_runtime_service.py tests/test_architecture_boundaries.py`
Expected: no output
