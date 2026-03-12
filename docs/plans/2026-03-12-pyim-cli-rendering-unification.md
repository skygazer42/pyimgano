# Pyim CLI Rendering Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin `pyimgano.pyim_cli` further by moving list-mode rendering into a dedicated adapter while preserving current JSON payload shapes, text output, and exit codes.

**Architecture:** Add a `pyimgano.pyim_cli_rendering` module that owns `pyim --list` presentation rules: which payload slice is emitted for each `list_kind`, how JSON payloads differ from text sections, and how the combined `all` view is rendered. Keep `pyimgano.pyim_cli` responsible only for argparse, audit handling, option normalization, service invocation, and final delegation to the rendering adapter.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano.cli_output`, existing `pyimgano.services.pyim_service`.

---

### Task 1: Add A Dedicated Pyim Rendering Adapter

**Files:**
- Create: `pyimgano/pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli_rendering.py`

**Step 1: Write the failing test**

Add tests that prove:
- `emit_pyim_list_payload(...)` delegates JSON output through `pyimgano.cli_output`
- `model-presets` JSON uses `model_preset_infos`, not `model_presets`
- `all` JSON preserves the existing payload shape and does not expose `model_preset_infos`
- text rendering for `all` still prints the expected section headers and optional recipe/dataset blocks

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py -v`
Expected: FAIL because `pyimgano.pyim_cli_rendering` does not exist yet.

**Step 3: Write minimal implementation**

- Create `pyimgano/pyim_cli_rendering.py`
- Move the section rendering helpers and list-kind dispatch into that module
- Keep the module CLI-facing only; do not move it into services

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py -v`
Expected: PASS

### Task 2: Refactor `pyim_cli` To Delegate Rendering

**Files:**
- Modify: `pyimgano/pyim_cli.py`
- Test: `tests/test_pyim_cli.py`

**Step 1: Write the failing test**

Extend `tests/test_pyim_cli.py` so it proves:
- `pyimgano.pyim_cli.main(... --list models ...)` delegates final rendering to `pyimgano.pyim_cli_rendering`
- the CLI forwards the normalized `list_kind`, service payload, and `json_output` flag to the rendering adapter

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py tests/test_pyim_cli.py -v`
Expected: FAIL because `pyim_cli` still renders list output inline.

**Step 3: Write minimal implementation**

- Import `pyimgano.pyim_cli_rendering` into `pyimgano.pyim_cli`
- Replace the inline list-kind rendering branches with a single adapter call
- Preserve audit behavior and request construction

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py tests/test_pyim_cli.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Create: `pyimgano/pyim_cli_rendering.py`
- Modify: `pyimgano/pyim_cli.py`
- Test: `tests/test_pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`
- Test: `tests/test_pyim_cli_model_presets.py`
- Test: `tests/test_pyim_service.py`
- Test: `tests/test_pyim_cli_options.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py tests/test_pyim_cli.py tests/test_pyim_cli_options.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_model_presets.py tests/test_pyim_service.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-cli-rendering-unification.md pyimgano/pyim_cli.py pyimgano/pyim_cli_rendering.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py`
Expected: no output
