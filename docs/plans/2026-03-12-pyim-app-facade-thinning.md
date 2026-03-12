# Pyim App Facade Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin `pyim_cli.py` into a parser-only adapter by moving command orchestration and audit/list branching into a neutral `pyim_app` facade.

**Architecture:** Add a root-level `pyimgano.pyim_app` module with a small `PyimCommand` dataclass and `run_pyim_command()` entrypoint that coordinates option normalization, request construction, service collection, rendering, and metadata audit output. Refactor `pyimgano.pyim_cli` to keep only parser setup, help behavior, command object construction, and parser-owned error reporting, then lock that import boundary with tests.

**Tech Stack:** Python 3.10, pytest, dataclasses, argparse, existing `pyimgano.pyim_cli_options`, `pyimgano.pyim_cli_rendering`, `pyimgano.services.pyim_service`

---

### Task 1: Lock The New App Boundary With Failing Tests

**Files:**
- Create: `tests/test_pyim_app.py`
- Modify: `tests/test_pyim_cli.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `run_pyim_command()` delegates list flow through the shared option helper, neutral request contract, payload collection service, and rendering helper
- `run_pyim_command()` handles metadata audit JSON output and preserves the nonzero exit behavior when issues exist
- `pyim_cli.main()` delegates parsed command arguments to the app facade instead of directly talking to rendering, contracts, services, or audit helpers
- `pyimgano/pyim_cli.py` no longer directly imports `cli_output`, `pyim_contracts`, `pyim_cli_options`, `pyim_cli_rendering`, `pyimgano.services.pyim_service`, or `pyimgano.models.registry`

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_app.py tests/test_pyim_cli.py tests/test_architecture_boundaries.py::test_pyim_cli_uses_app_facade_boundary -v`
Expected: FAIL because `pyim_app.py` does not exist yet and `pyim_cli.py` still owns orchestration details.

### Task 2: Extract The App Facade

**Files:**
- Create: `pyimgano/pyim_app.py`
- Modify: `pyimgano/pyim_cli.py`

**Step 1: Write minimal implementation**

- Add a frozen `PyimCommand` dataclass carrying parsed command inputs
- Add `run_pyim_command()` with separate audit and list flow helpers
- Refactor `pyim_cli.main()` to build a `PyimCommand` and delegate execution to `pyim_app`
- Preserve current help behavior, parser error behavior, text output, JSON output, and exit codes exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_app.py tests/test_pyim_cli.py tests/test_architecture_boundaries.py::test_pyim_cli_uses_app_facade_boundary -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-app-facade-thinning.md`
- Modify: `pyimgano/pyim_app.py`
- Modify: `pyimgano/pyim_cli.py`
- Modify: `tests/test_pyim_app.py`
- Modify: `tests/test_pyim_cli.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_pyim_app.py tests/test_pyim_payload_collectors.py tests/test_pyim_section_views.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-app-facade-thinning.md pyimgano/pyim_app.py pyimgano/pyim_cli.py tests/test_pyim_app.py tests/test_pyim_cli.py tests/test_architecture_boundaries.py`
Expected: no output
