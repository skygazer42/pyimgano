# Pyim List Kind Spec Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Centralize `pyim --list` section semantics so parser choices, option validation, and payload serialization stop duplicating string-based list-kind logic across modules.

**Architecture:** Add a neutral `pyimgano.pyim_list_spec` module that defines the supported list kinds and their behavioral metadata: payload field, inclusion flags, and filter support. Refactor `pyimgano.pyim_cli`, `pyimgano.pyim_cli_options`, and `pyimgano.pyim_cli_rendering` to consume that shared spec, and move per-section JSON access into `PyimListPayload` so serialization rules live with the payload contract.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_contracts`, `pyimgano.pyim_cli`, `pyimgano.pyim_cli_options`, `pyimgano.pyim_cli_rendering`

---

### Task 1: Lock The Shared List-Kind Behavior With Failing Tests

**Files:**
- Create: `tests/test_pyim_list_spec.py`
- Modify: `tests/test_pyim_contracts.py`

**Step 1: Write the failing test**

Add tests that prove:
- a shared `PYIM_LIST_KIND_CHOICES` catalog exists and includes the supported `pyim --list` values in the CLI order
- the shared spec exposes inclusion/filter flags for representative kinds such as `all`, `models`, `recipes`, and `preprocessing`
- `PyimListPayload` can serialize a section by list kind while preserving the external JSON shape for typed sections

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_contracts.py -v`
Expected: FAIL because the shared spec module and per-kind payload serializer do not exist yet.

### Task 2: Refactor Pyim Modules To Consume The Shared Spec

**Files:**
- Create: `pyimgano/pyim_list_spec.py`
- Modify: `pyimgano/pyim_contracts.py`
- Modify: `pyimgano/pyim_cli.py`
- Modify: `pyimgano/pyim_cli_options.py`
- Modify: `pyimgano/pyim_cli_rendering.py`

**Step 1: Write minimal implementation**

- Add a neutral list-kind spec catalog and lookup helper
- Export shared parser choices from that module
- Move section JSON serialization into `PyimListPayload`
- Update parser choices, option validation, and rendering to consume the shared spec
- Keep current external text and JSON output unchanged

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_cli.py tests/test_pyim_cli_options.py tests/test_pyim_cli_rendering.py -v`
Expected: PASS

### Task 3: Regression And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-list-kind-spec-unification.md`
- Modify: `pyimgano/pyim_list_spec.py`
- Modify: `pyimgano/pyim_contracts.py`
- Modify: `pyimgano/pyim_cli.py`
- Modify: `pyimgano/pyim_cli_options.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Modify: `tests/test_pyim_list_spec.py`
- Modify: `tests/test_pyim_contracts.py`

**Step 1: Run pyim regression suite**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-list-kind-spec-unification.md pyimgano/pyim_list_spec.py pyimgano/pyim_contracts.py pyimgano/pyim_cli.py pyimgano/pyim_cli_options.py pyimgano/pyim_cli_rendering.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py`
Expected: no output
