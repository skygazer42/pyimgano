# Pyim Service Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyim_cli` coupling by moving list payload assembly behind a dedicated service boundary while preserving existing CLI output and validation behavior.

**Architecture:** Add a `pyimgano.services.pyim_service` module that owns the data collection for `pyim --list` sections. Keep CLI concerns in `pyimgano.pyim_cli`: argparse, user-facing validation, JSON/text rendering, and exit codes. Move the multi-source section aggregation logic into a request-driven service so the CLI stops importing a broad mix of registry, discovery, preprocessing, preset, recipe, and dataset modules directly.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing discovery/preprocessing/preset/recipe/dataset APIs.

---

### Task 1: Add A Dedicated Pyim Listing Service

**Files:**
- Create: `pyimgano/services/pyim_service.py`
- Test: `tests/test_pyim_service.py`

**Step 1: Write the failing test**

Add `tests/test_pyim_service.py` with tests that prove:
- a request object can collect the core `pyim --list` sections (`models`, `families`, `types`, `years`, `metadata_contract`, `preprocessing`, `features`, `model_presets`, `defects_presets`)
- optional recipe and dataset sections are included only when requested
- model preset filters still respect `tags` and `family`

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_pyim_service.py -v`
Expected: FAIL because `pyimgano.services.pyim_service` does not exist yet.

**Step 3: Write minimal implementation**

- Create `pyimgano/services/pyim_service.py`
- Add a `PyimListRequest` dataclass
- Add `collect_pyim_listing_payload(request) -> dict[str, Any]`
- Keep the service responsible only for payload collection, not terminal rendering

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_service.py -v`
Expected: PASS

### Task 2: Refactor `pyim_cli` To Delegate Payload Assembly

**Files:**
- Modify: `pyimgano/pyim_cli.py`
- Modify: `pyimgano/services/__init__.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`

**Step 1: Write the failing test**

Extend CLI tests so they prove:
- `pyimgano.pyim_cli.main(... --list models ...)` delegates section assembly to `pyimgano.services.pyim_service`
- `pyimgano.pyim_cli.main(... --list --json ...)` delegates with recipe and dataset inclusion enabled

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py -v`
Expected: FAIL because `pyim_cli` still assembles payloads inline.

**Step 3: Write minimal implementation**

- Import the new service in `pyimgano.pyim_cli`
- Replace inline multi-source section collection with one service call
- Preserve current parser validation, text rendering, and JSON payload shapes
- Update `pyimgano.services.__init__` exports only if needed for compatibility

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_model_presets.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Test: `tests/test_pyim_service.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`
- Test: `tests/test_pyim_cli_model_presets.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_model_presets.py tests/test_cli_discovery_options.py tests/test_cli_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-service-extraction.md pyimgano/services/pyim_service.py pyimgano/services/__init__.py pyimgano/pyim_cli.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py`
Expected: no output
