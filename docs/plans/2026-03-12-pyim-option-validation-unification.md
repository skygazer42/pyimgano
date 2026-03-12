# Pyim Option Validation Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.pyim_cli` coupling by moving list-mode option validation and normalization into a dedicated helper while preserving existing command behavior and output.

**Architecture:** Add a `pyimgano.pyim_cli_options` module that owns `pyim --list` option semantics: which filters are legal for each list mode, how filter values are validated, and how list-mode inclusion flags are derived. Keep `pyimgano.pyim_cli` responsible only for argparse, top-level command branching, service invocation, and text/JSON rendering.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing discovery resolver helpers, existing `pyimgano.services.pyim_service`.

---

### Task 1: Add A Dedicated Pyim Options Helper

**Files:**
- Create: `pyimgano/pyim_cli_options.py`
- Test: `tests/test_pyim_cli_options.py`

**Step 1: Write the failing test**

Add tests that prove:
- `resolve_pyim_list_options(...)` normalizes `list_kind`, `family`, `algorithm_type`, `year`, and `deployable_only`
- invalid combinations raise stable errors:
  - `--family` is rejected outside `all`, `models`, and `model-presets`
  - `--type` is rejected outside `models`
  - `--year` is rejected outside `models`
  - `--deployable-only` is rejected outside `all` and `preprocessing`
- family/type/year values are validated through the existing discovery resolver functions
- section inclusion flags are derived from `list_kind` instead of being re-encoded in the CLI

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_pyim_cli_options.py -v`
Expected: FAIL because `pyimgano.pyim_cli_options` does not exist yet.

**Step 3: Write minimal implementation**

- Create `pyimgano/pyim_cli_options.py`
- Add a frozen dataclass for normalized options
- Add `resolve_pyim_list_options(...)`
- Keep the helper free of printing and service imports

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_cli_options.py -v`
Expected: PASS

### Task 2: Refactor `pyim_cli` To Use The Shared Options Helper

**Files:**
- Modify: `pyimgano/pyim_cli.py`
- Test: `tests/test_pyim_cli.py`

**Step 1: Write the failing test**

Extend `tests/test_pyim_cli.py` so it proves:
- `pyimgano.pyim_cli.main(... --list models ...)` delegates option normalization to `pyimgano.pyim_cli_options`
- the normalized values from the helper are what reach `pyimgano.services.pyim_service.PyimListRequest`

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_cli.py tests/test_pyim_cli_options.py -v`
Expected: FAIL because `pyim_cli` still performs inline validation and request shaping.

**Step 3: Write minimal implementation**

- Import `pyimgano.pyim_cli_options` into `pyimgano.pyim_cli`
- Replace the inline list-mode validation block with one helper call
- Build service requests from the normalized options object
- Preserve JSON/text payload shapes and exit codes

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_cli.py tests/test_pyim_cli_options.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Modify: `pyimgano/pyim_cli.py`
- Create: `pyimgano/pyim_cli_options.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_options.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`
- Test: `tests/test_pyim_cli_model_presets.py`
- Test: `tests/test_pyim_service.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_pyim_cli_options.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_model_presets.py tests/test_pyim_service.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-option-validation-unification.md pyimgano/pyim_cli.py pyimgano/pyim_cli_options.py tests/test_pyim_cli.py tests/test_pyim_cli_options.py`
Expected: no output
