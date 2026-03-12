# Pyim Rendering Spec Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining duplicated presentation metadata in `pyim_cli_rendering` by moving section titles, text renderer kinds, and `all`-mode text order into the shared `pyim` list spec.

**Architecture:** Extend `pyimgano.pyim_list_spec` so each list kind declares its text title and text renderer kind, and add a shared ordered tuple for `all`-mode text rendering. Refactor `pyimgano.pyim_cli_rendering` to dispatch through that metadata, keeping formatting logic local but eliminating section-specific wiring from the rendering module.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_list_spec`, `pyimgano.pyim_cli_rendering`

---

### Task 1: Lock Shared Rendering Metadata With Failing Tests

**Files:**
- Modify: `tests/test_pyim_list_spec.py`
- Modify: `tests/test_pyim_cli_rendering.py`

**Step 1: Write the failing test**

Add tests that prove:
- the shared list spec exposes text title and text renderer metadata for representative kinds such as `models`, `metadata-contract`, `preprocessing`, and `datasets`
- the shared spec exports the ordered text sections used by `pyim --list`
- `emit_pyim_list_payload(..., list_kind="all", json_output=False)` renders sections in that shared order and still skips empty optional recipe/dataset sections

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_cli_rendering.py -v`
Expected: FAIL because the list spec does not yet expose rendering metadata and the rendering module still owns the all-mode order.

### Task 2: Refactor Rendering To Consume The Shared Spec

**Files:**
- Modify: `pyimgano/pyim_list_spec.py`
- Modify: `pyimgano/pyim_cli_rendering.py`

**Step 1: Write minimal implementation**

- Add text title and text renderer metadata to the list-kind spec
- Export the shared `all`-mode text order
- Replace list-kind keyed rendering dispatch with render-kind keyed dispatch plus spec lookup
- Keep current textual output unchanged

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_cli_rendering.py -v`
Expected: PASS

### Task 3: Regression And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-rendering-spec-unification.md`
- Modify: `pyimgano/pyim_list_spec.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Modify: `tests/test_pyim_list_spec.py`
- Modify: `tests/test_pyim_cli_rendering.py`

**Step 1: Run pyim regression suite**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-rendering-spec-unification.md pyimgano/pyim_list_spec.py pyimgano/pyim_cli_rendering.py tests/test_pyim_list_spec.py tests/test_pyim_cli_rendering.py`
Expected: no output
