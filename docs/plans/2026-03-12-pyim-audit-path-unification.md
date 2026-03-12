# Pyim Audit Path Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `pyim` metadata audit path follow the same split as the list path by moving audit collection and audit output into dedicated service/rendering modules.

**Architecture:** Add `pyimgano.services.pyim_audit_service` to collect the raw metadata audit payload and add `pyimgano.pyim_audit_rendering` to emit JSON/text audit output and compute the corresponding exit code. Refactor `pyimgano.pyim_app` to delegate the audit branch through those helpers so the app module remains a thin coordinator, then lock the boundary with tests.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_app`, `pyimgano.cli_output`, `pyimgano.models.registry`

---

### Task 1: Lock The Audit Boundary With Failing Tests

**Files:**
- Create: `tests/test_pyim_audit_service.py`
- Create: `tests/test_pyim_audit_rendering.py`
- Modify: `tests/test_pyim_app.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated audit service collects and returns the raw metadata audit payload from the registry
- a dedicated audit rendering helper emits JSON output and preserves the nonzero exit code when issues exist
- the same rendering helper emits the expected text summary header and status line
- `pyim_app.run_pyim_command()` delegates the audit branch to the audit service and audit rendering helpers instead of directly importing registry or `cli_output`
- `pyimgano/pyim_app.py` no longer directly imports `pyimgano.cli_output` or `pyimgano.models.registry`

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_audit_service.py tests/test_pyim_audit_rendering.py tests/test_pyim_app.py tests/test_architecture_boundaries.py::test_pyim_app_uses_audit_helpers_boundary -v`
Expected: FAIL because the audit helper modules do not exist yet and `pyim_app.py` still owns the audit branch details.

### Task 2: Extract The Audit Service And Rendering Modules

**Files:**
- Create: `pyimgano/services/pyim_audit_service.py`
- Create: `pyimgano/pyim_audit_rendering.py`
- Modify: `pyimgano/pyim_app.py`

**Step 1: Write minimal implementation**

- Add a service helper that collects the raw audit payload from `pyimgano.models.registry.audit_model_metadata`
- Add a rendering helper that emits JSON/text output and returns the same exit code semantics as today
- Refactor `pyim_app` so the audit branch delegates to those modules
- Preserve current CLI-visible output and exit codes exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_audit_service.py tests/test_pyim_audit_rendering.py tests/test_pyim_app.py tests/test_architecture_boundaries.py::test_pyim_app_uses_audit_helpers_boundary -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-audit-path-unification.md`
- Modify: `pyimgano/services/pyim_audit_service.py`
- Modify: `pyimgano/pyim_audit_rendering.py`
- Modify: `pyimgano/pyim_app.py`
- Modify: `tests/test_pyim_audit_service.py`
- Modify: `tests/test_pyim_audit_rendering.py`
- Modify: `tests/test_pyim_app.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_pyim_audit_service.py tests/test_pyim_audit_rendering.py tests/test_pyim_app.py tests/test_pyim_payload_collectors.py tests/test_pyim_section_views.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-audit-path-unification.md pyimgano/services/pyim_audit_service.py pyimgano/pyim_audit_rendering.py pyimgano/pyim_app.py tests/test_pyim_audit_service.py tests/test_pyim_audit_rendering.py tests/test_pyim_app.py tests/test_architecture_boundaries.py`
Expected: no output
