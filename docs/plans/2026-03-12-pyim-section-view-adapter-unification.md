# Pyim Section View Adapter Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce a neutral `pyim` section-view adapter so the CLI rendering layer no longer owns list-kind section lookup, payload coercion, or `all`-mode section selection semantics.

**Architecture:** Add a small `pyimgano.pyim_section_views` module that combines shared list-kind spec metadata with a `PyimListPayload` to expose JSON payload resolution, single text section views, and ordered `all`-mode text section iteration. Refactor `pyimgano.pyim_cli_rendering` to consume those neutral views, keeping the actual formatting functions local while removing remaining `list_kind`-specific business knowledge from the CLI adapter.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_contracts`, `pyimgano.pyim_list_spec`, `pyimgano.pyim_cli_rendering`

---

### Task 1: Lock The New Boundary With Failing Tests

**Files:**
- Create: `tests/test_pyim_section_views.py`
- Modify: `tests/test_pyim_cli_rendering.py`

**Step 1: Write the failing test**

Add tests that prove:
- a neutral section-view helper can resolve a single text section for `model-presets`, carrying the shared title, render kind, and section value
- the same helper can iterate `all` text sections in shared order while still omitting empty optional `recipes` and `datasets`
- `emit_pyim_list_payload(..., list_kind="all", json_output=False)` still renders the same section order after the rendering module is simplified around the new adapter boundary

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_section_views.py tests/test_pyim_cli_rendering.py -v`
Expected: FAIL because the neutral section-view module does not exist yet and the rendering layer still owns section resolution.

### Task 2: Extract The Section View Adapter

**Files:**
- Create: `pyimgano/pyim_section_views.py`
- Modify: `pyimgano/pyim_cli_rendering.py`

**Step 1: Write minimal implementation**

- Add a small frozen text section view dataclass that exposes `list_kind`, `title`, `render_kind`, and `value`
- Add helper functions to coerce payload mappings into `PyimListPayload`
- Add helper functions to resolve JSON payload for a list kind, resolve a single text section view, and iterate `all` text section views
- Refactor `pyim_cli_rendering` to consume those helpers so it no longer imports `pyim_list_spec` or `pyim_contracts` directly
- Preserve current JSON and text output shape exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_section_views.py tests/test_pyim_cli_rendering.py -v`
Expected: PASS

### Task 3: Guard The Architecture And Regressions

**Files:**
- Modify: `tests/test_architecture_boundaries.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Modify: `pyimgano/pyim_section_views.py`
- Modify: `tests/test_pyim_section_views.py`
- Modify: `tests/test_pyim_cli_rendering.py`

**Step 1: Write the failing architecture test**

Add a boundary test that asserts `pyimgano/pyim_cli_rendering.py` does not directly import `pyimgano.pyim_list_spec` or `pyimgano.pyim_contracts`.

**Step 2: Run tests to verify it fails before the refactor is complete**

Run: `pytest --no-cov tests/test_architecture_boundaries.py::test_pyim_cli_rendering_uses_section_view_adapter_boundary -v`
Expected: FAIL until the rendering module depends only on the neutral section-view adapter.

**Step 3: Run regression suite**

Run: `pytest --no-cov tests/test_pyim_section_views.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py tests/test_services_package.py -v`
Expected: PASS

**Step 4: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-section-view-adapter-unification.md pyimgano/pyim_section_views.py pyimgano/pyim_cli_rendering.py tests/test_pyim_section_views.py tests/test_pyim_cli_rendering.py tests/test_architecture_boundaries.py`
Expected: no output
