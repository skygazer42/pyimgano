# Pyim Dependency Whitelist Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the current `pyim` architecture into an explicitly enforced module dependency whitelist and remove the last direct `pyim_cli -> pyim_list_spec` dependency.

**Architecture:** Add a data-driven architecture test that enumerates the allowed internal `pyimgano.*` imports for the main `pyim` modules and fails on any dependency creep. To make the CLI adapter comply with a stricter boundary, re-export `PYIM_LIST_KIND_CHOICES` from `pyimgano.pyim_cli_options` and refactor `pyimgano.pyim_cli` to source parser choices from the options module instead of importing `pyim_list_spec` directly.

**Tech Stack:** Python 3.10, pytest, ast, existing `pyimgano.pyim_cli`, `pyimgano.pyim_cli_options`, `tests/test_architecture_boundaries.py`

---

### Task 1: Lock The New Whitelist Boundary With Failing Tests

**Files:**
- Modify: `tests/test_pyim_cli_options.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.pyim_cli_options` re-exports the shared `PYIM_LIST_KIND_CHOICES` used by the CLI
- the main `pyim` modules import only the allowed internal `pyimgano.*` modules according to an explicit whitelist
- `pyimgano/pyim_cli.py` is not allowed to import `pyimgano.pyim_list_spec` directly

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py -v`
Expected: FAIL because `pyim_cli_options.py` does not yet expose `PYIM_LIST_KIND_CHOICES` and `pyim_cli.py` still directly imports `pyim_list_spec`.

### Task 2: Make The CLI Follow The Whitelist

**Files:**
- Modify: `pyimgano/pyim_cli_options.py`
- Modify: `pyimgano/pyim_cli.py`

**Step 1: Write minimal implementation**

- Re-export `PYIM_LIST_KIND_CHOICES` from `pyimgano.pyim_cli_options`
- Update `pyimgano.pyim_cli` to use `pyim_cli_options.PYIM_LIST_KIND_CHOICES`
- Remove the direct `pyimgano.pyim_list_spec` import from the CLI adapter
- Preserve parser behavior, list choices, help text, and exit codes exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-dependency-whitelist-hardening.md`
- Modify: `pyimgano/pyim_cli_options.py`
- Modify: `pyimgano/pyim_cli.py`
- Modify: `tests/test_pyim_cli_options.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_pyim_audit_service.py tests/test_pyim_audit_rendering.py tests/test_pyim_app.py tests/test_pyim_payload_collectors.py tests/test_pyim_section_views.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-dependency-whitelist-hardening.md pyimgano/pyim_cli_options.py pyimgano/pyim_cli.py tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py`
Expected: no output
