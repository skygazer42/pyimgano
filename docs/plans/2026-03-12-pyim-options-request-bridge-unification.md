# Pyim Options Request Bridge Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move `PyimListOptions -> PyimListRequest` construction out of `pyim_app` so the app layer stops knowing request field details and only orchestrates collaborators.

**Architecture:** Extend `PyimListOptions` with a small `to_request()` bridge method that returns the neutral `PyimListRequest` contract. Refactor `pyimgano.pyim_app` to consume that method instead of importing `pyim_contracts` directly, and lock the boundary with tests so the app module remains a thin coordinator while the options module owns request translation semantics.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_cli_options`, `pyimgano.pyim_contracts`, `pyimgano.pyim_app`

---

### Task 1: Lock The Bridge Boundary With Failing Tests

**Files:**
- Modify: `tests/test_pyim_cli_options.py`
- Modify: `tests/test_pyim_app.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `PyimListOptions.to_request()` builds a neutral `PyimListRequest` with the normalized option values
- `pyim_app.run_pyim_command()` gets the request from `list_options.to_request()` instead of directly constructing it with `pyim_contracts`
- `pyimgano/pyim_app.py` no longer directly imports `pyimgano.pyim_contracts`

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_cli_options.py tests/test_pyim_app.py tests/test_architecture_boundaries.py::test_pyim_app_uses_audit_helpers_boundary -v`
Expected: FAIL because `PyimListOptions` does not yet expose `to_request()` and `pyim_app.py` still imports `pyim_contracts`.

### Task 2: Move The Bridge

**Files:**
- Modify: `pyimgano/pyim_cli_options.py`
- Modify: `pyimgano/pyim_app.py`

**Step 1: Write minimal implementation**

- Add `PyimListOptions.to_request()` that returns `pyimgano.pyim_contracts.PyimListRequest`
- Remove direct `pyimgano.pyim_contracts` usage from `pyim_app`
- Delegate service collection through `list_options.to_request()`
- Preserve request contents, CLI output, and exit codes exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_cli_options.py tests/test_pyim_app.py tests/test_architecture_boundaries.py::test_pyim_app_uses_audit_helpers_boundary -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-options-request-bridge-unification.md`
- Modify: `pyimgano/pyim_cli_options.py`
- Modify: `pyimgano/pyim_app.py`
- Modify: `tests/test_pyim_cli_options.py`
- Modify: `tests/test_pyim_app.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_pyim_audit_service.py tests/test_pyim_audit_rendering.py tests/test_pyim_app.py tests/test_pyim_payload_collectors.py tests/test_pyim_section_views.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-options-request-bridge-unification.md pyimgano/pyim_cli_options.py pyimgano/pyim_app.py tests/test_pyim_cli_options.py tests/test_pyim_app.py tests/test_architecture_boundaries.py`
Expected: no output
