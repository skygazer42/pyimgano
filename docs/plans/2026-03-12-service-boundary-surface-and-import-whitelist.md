# Service Boundary Surface And Import Whitelist Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `pyimgano.services` package feel like a coherent boundary by standardizing explicit module exports and locking the current service-to-service dependency graph.

**Architecture:** Add architecture tests that require every concrete service module to declare `__all__` and that restrict internal `pyimgano.services.*` imports to an explicit whitelist. Then add the missing `__all__` declarations for `inference_service.py` and `model_options.py`, keeping the current runtime behavior and package exports unchanged.

**Tech Stack:** Python 3.10, pytest, ast, existing `pyimgano/services/*.py`, `tests/test_architecture_boundaries.py`

---

### Task 1: Lock Service Boundaries With Failing Tests

**Files:**
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- every concrete module under `pyimgano/services` declares an explicit `__all__`
- service modules import only the currently allowed `pyimgano.services.*` modules
- `pyimgano.services.inference_service` exposes only `InferenceRunResult`, `iter_inference_records`, and `run_inference`
- `pyimgano.services.model_options` exposes only `apply_onnx_session_options_shorthand`, `enforce_checkpoint_requirement`, `resolve_model_options`, `resolve_preset_kwargs`, and `resolve_requested_model`

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_architecture_boundaries.py -v`
Expected: FAIL because `pyimgano/services/inference_service.py` and `pyimgano/services/model_options.py` do not yet define `__all__`.

### Task 2: Standardize Missing Service Public Surfaces

**Files:**
- Modify: `pyimgano/services/inference_service.py`
- Modify: `pyimgano/services/model_options.py`

**Step 1: Write minimal implementation**

- Add explicit `__all__` declarations to the two missing service boundary modules
- Export only the intended stable public symbols
- Preserve service behavior, call signatures, and CLI-visible behavior exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_architecture_boundaries.py -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-service-boundary-surface-and-import-whitelist.md`
- Modify: `pyimgano/services/inference_service.py`
- Modify: `pyimgano/services/model_options.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_inference_service.py tests/test_model_options_service.py tests/test_infer_continue_service.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-service-boundary-surface-and-import-whitelist.md pyimgano/services/inference_service.py pyimgano/services/model_options.py tests/test_architecture_boundaries.py`
Expected: no output
