# Pyim Public Surface Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `pyim` slice feel like a coherent boundary by standardizing explicit module exports and locking the intended public surface with architecture tests.

**Architecture:** Treat the main `pyim` modules as boundary modules with explicit `__all__` declarations, so public orchestration symbols stay clear while helper functions remain private. Add a data-driven architecture test that parses each module's `__all__` and fails if a boundary module stops declaring exports or begins exposing an unintended symbol set.

**Tech Stack:** Python 3.10, pytest, ast, existing `pyimgano.pyim_*` modules, `pyimgano.services.pyim_*`, `tests/test_architecture_boundaries.py`

---

### Task 1: Lock The Intended Public Surface With Failing Tests

**Files:**
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add a data-driven test that proves:
- the main `pyim` boundary modules define `__all__`
- `pyimgano.pyim_cli_options` exposes only `PYIM_LIST_KIND_CHOICES`, `PyimListOptions`, and `resolve_pyim_list_options`
- `pyimgano.pyim_cli_rendering` exposes only `emit_pyim_list_payload`
- existing public boundary modules continue exporting their current intended symbol sets

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_architecture_boundaries.py -v`
Expected: FAIL because `pyim_cli_options.py` and `pyim_cli_rendering.py` do not yet declare `__all__`.

### Task 2: Standardize The Boundary Modules

**Files:**
- Modify: `pyimgano/pyim_cli_options.py`
- Modify: `pyimgano/pyim_cli_rendering.py`

**Step 1: Write minimal implementation**

- Add explicit `__all__` declarations to the missing `pyim` boundary modules
- Export only the stable boundary symbols that callers are meant to use
- Preserve existing helper names, rendering behavior, parser behavior, and JSON/text output exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_architecture_boundaries.py -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-public-surface-unification.md`
- Modify: `pyimgano/pyim_cli_options.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_pyim_audit_service.py tests/test_pyim_audit_rendering.py tests/test_pyim_app.py tests/test_pyim_payload_collectors.py tests/test_pyim_section_views.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-public-surface-unification.md pyimgano/pyim_cli_options.py pyimgano/pyim_cli_rendering.py tests/test_architecture_boundaries.py`
Expected: no output
