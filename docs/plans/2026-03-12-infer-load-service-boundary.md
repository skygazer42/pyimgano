# Infer Load Service Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clarify infer detector loading boundaries by moving load implementation into a dedicated `infer_load_service` and shrinking `infer_setup_service` to a compatibility layer.

**Architecture:** Add `pyimgano.services.infer_load_service` as the concrete home for direct/config-backed detector load contracts and implementation. Update `infer_cli.py` and the service-root export map to depend on that module, then reduce `infer_setup_service.py` to a thin re-export shim so the package reads as a coherent set of `infer_*_service` boundaries rather than a mixed bag.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano/infer_cli.py`, `pyimgano/services/infer_setup_service.py`, `pyimgano/services/__init__.py`, `tests/test_infer_cli_smoke.py`, `tests/test_infer_cli_infer_config.py`, `tests/test_architecture_boundaries.py`

---

### Task 1: Lock The New Load Boundary With Failing Tests

**Files:**
- Create: `tests/test_infer_load_service.py`
- Modify: `tests/test_infer_setup_service.py`
- Modify: `tests/test_infer_cli_smoke.py`
- Modify: `tests/test_infer_cli_infer_config.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.services.infer_load_service` exposes the current detector-loading boundary
- `infer_setup_service.py` is now only a compatibility re-export surface
- `infer_cli.py` delegates detector loading through `infer_load_service`
- architecture rules know about the new service and expect `infer_setup_service.py` to depend only on it

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_infer_load_service.py tests/test_infer_setup_service.py tests/test_infer_cli_smoke.py tests/test_infer_cli_infer_config.py tests/test_architecture_boundaries.py tests/test_services_package.py -v`
Expected: FAIL because `infer_load_service.py` does not exist yet and current imports/exports still point at `infer_setup_service.py`.

### Task 2: Introduce infer_load_service And Migrate Callers

**Files:**
- Create: `pyimgano/services/infer_load_service.py`
- Modify: `pyimgano/services/infer_setup_service.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/services/__init__.py`

**Step 1: Write minimal implementation**

- Move current load dataclasses and load functions into `infer_load_service.py`
- Repoint CLI imports and service-root exports to the new module
- Turn `infer_setup_service.py` into a compatibility re-export with explicit `__all__`
- Preserve detector creation, alias resolution, checkpoint restoration, and threshold application exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_infer_load_service.py tests/test_infer_setup_service.py tests/test_infer_cli_smoke.py tests/test_infer_cli_infer_config.py tests/test_architecture_boundaries.py tests/test_services_package.py -v`
Expected: PASS

### Task 3: Regression And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-infer-load-service-boundary.md`
- Create: `pyimgano/services/infer_load_service.py`
- Modify: `pyimgano/services/infer_setup_service.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/services/__init__.py`
- Create: `tests/test_infer_load_service.py`
- Modify: `tests/test_infer_setup_service.py`
- Modify: `tests/test_infer_cli_smoke.py`
- Modify: `tests/test_infer_cli_infer_config.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_infer_load_service.py tests/test_infer_setup_service.py tests/test_infer_wrapper_service.py tests/test_infer_cli_smoke.py tests/test_infer_cli_infer_config.py tests/test_infer_runtime_service.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-infer-load-service-boundary.md pyimgano/services/infer_load_service.py pyimgano/services/infer_setup_service.py pyimgano/infer_cli.py pyimgano/services/__init__.py tests/test_infer_load_service.py tests/test_infer_setup_service.py tests/test_infer_cli_smoke.py tests/test_infer_cli_infer_config.py tests/test_architecture_boundaries.py`
Expected: no output
