# Infer Wrapper Service Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce infer setup coupling by moving detector wrapper application into a dedicated service-layer boundary.

**Architecture:** Add a narrow `pyimgano.services.infer_wrapper_service` module that owns wrapper request/result contracts plus tiling and preprocessing application. Refactor callers and service-root exports to depend on that module so `infer_setup_service.py` remains focused on direct/config-backed detector loading while wrapper orchestration lives behind a separate boundary.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano/infer_cli.py`, `pyimgano/services/infer_setup_service.py`, `pyimgano/inference/*`, `tests/test_infer_setup_service.py`, `tests/test_infer_cli_infer_config.py`, `tests/test_architecture_boundaries.py`, `tests/test_services_package.py`

---

### Task 1: Lock The Wrapper Boundary With Failing Tests

**Files:**
- Create: `tests/test_infer_wrapper_service.py`
- Modify: `tests/test_infer_setup_service.py`
- Modify: `tests/test_infer_cli_infer_config.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.services.infer_wrapper_service` exposes the intended wrapper boundary
- detector wrapper behavior stays the same after extraction
- infer CLI delegates wrapper setup through the new service instead of `infer_setup_service`
- architecture/service export checks include the new boundary module

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_infer_wrapper_service.py tests/test_infer_setup_service.py tests/test_infer_cli_infer_config.py tests/test_architecture_boundaries.py tests/test_services_package.py -v`
Expected: FAIL because the new service does not yet exist and wrapper calls/exports still point at `infer_setup_service.py`.

### Task 2: Extract Wrapper Logic Into The New Service

**Files:**
- Create: `pyimgano/services/infer_wrapper_service.py`
- Modify: `pyimgano/services/infer_setup_service.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/services/__init__.py`

**Step 1: Write minimal implementation**

- Move wrapper request/result dataclasses and wrapper application logic into `infer_wrapper_service.py`
- Keep `_require_numpy_model_for_preprocessing()` private to the new module
- Update `infer_cli.py` to import and call the new service boundary
- Repoint service-root exports for wrapper APIs to the new module
- Preserve detector threshold handling, tiling payload semantics, and preprocessing behavior exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_infer_wrapper_service.py tests/test_infer_setup_service.py tests/test_infer_cli_infer_config.py tests/test_architecture_boundaries.py tests/test_services_package.py -v`
Expected: PASS

### Task 3: Regression And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-infer-wrapper-service-boundary.md`
- Create: `pyimgano/services/infer_wrapper_service.py`
- Modify: `pyimgano/services/infer_setup_service.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/services/__init__.py`
- Create: `tests/test_infer_wrapper_service.py`
- Modify: `tests/test_infer_setup_service.py`
- Modify: `tests/test_infer_cli_infer_config.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_infer_wrapper_service.py tests/test_infer_setup_service.py tests/test_infer_cli_infer_config.py tests/test_infer_runtime_service.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-infer-wrapper-service-boundary.md pyimgano/services/infer_wrapper_service.py pyimgano/services/infer_setup_service.py pyimgano/infer_cli.py pyimgano/services/__init__.py tests/test_infer_wrapper_service.py tests/test_infer_setup_service.py tests/test_infer_cli_infer_config.py tests/test_architecture_boundaries.py`
Expected: no output
