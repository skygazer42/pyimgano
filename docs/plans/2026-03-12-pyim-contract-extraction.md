# Pyim Contract Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining module-level coupling between `pyimgano` CLI/rendering code and `pyimgano.services.pyim_service` by extracting `PyimListRequest` and `PyimListPayload` into a neutral contracts module.

**Architecture:** Add a `pyimgano.pyim_contracts` module that owns the request/payload dataclasses and the `all` JSON payload helper. Update `pyimgano.services.pyim_service` to consume and return those contracts, while keeping compatibility re-exports so existing imports do not break. Update `pyimgano.pyim_cli` and `pyimgano.pyim_cli_rendering` to depend on the neutral contracts module instead of importing service-layer types.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.services.pyim_service`, existing `pyimgano.pyim_cli`, existing `pyimgano.pyim_cli_rendering`.

---

### Task 1: Add A Neutral Pyim Contracts Module

**Files:**
- Create: `pyimgano/pyim_contracts.py`
- Test: `tests/test_pyim_contracts.py`
- Modify: `tests/test_pyim_service.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.pyim_contracts` exports `PyimListRequest` and `PyimListPayload`
- `pyimgano.services.pyim_service` re-exports the same contract objects for compatibility
- service tests can import contracts from the neutral module and still observe the same behavior

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_service.py -v`
Expected: FAIL because `pyimgano.pyim_contracts` does not exist yet.

**Step 3: Write minimal implementation**

- Create `pyimgano/pyim_contracts.py`
- Move `PyimListRequest` and `PyimListPayload` definitions there
- Import and re-export those contracts from `pyimgano.services.pyim_service`

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_service.py -v`
Expected: PASS

### Task 2: Refactor CLI And Rendering To Use Neutral Contracts

**Files:**
- Modify: `pyimgano/pyim_cli.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli_model_presets.py`

**Step 1: Write the failing test**

Extend tests so they prove:
- `pyimgano.pyim_cli` builds requests through `pyimgano.pyim_contracts.PyimListRequest`
- `pyimgano.pyim_cli_rendering` imports and consumes `PyimListPayload` from the neutral contracts module
- rendering/CLI delegation tests use the neutral contract type directly

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py -v`
Expected: FAIL because `pyim_cli` and `pyim_cli_rendering` still import contract types from `pyimgano.services.pyim_service`.

**Step 3: Write minimal implementation**

- Update `pyimgano.pyim_cli` to import `pyimgano.pyim_contracts`
- Update `pyimgano.pyim_cli_rendering` to import `PyimListPayload` from `pyimgano.pyim_contracts`
- Preserve service invocation, JSON payload shapes, and text rendering

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Create: `pyimgano/pyim_contracts.py`
- Modify: `pyimgano/services/pyim_service.py`
- Modify: `pyimgano/pyim_cli.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Test: `tests/test_pyim_contracts.py`
- Test: `tests/test_pyim_service.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli_model_presets.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`
- Test: `tests/test_services_package.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-contract-extraction.md pyimgano/pyim_contracts.py pyimgano/services/pyim_service.py pyimgano/pyim_cli.py pyimgano/pyim_cli_rendering.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py`
Expected: no output
