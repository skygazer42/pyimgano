# Pyim Request Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce duplicated `pyim` section semantics by letting `PyimListRequest` normalize section inclusion from `list_kind`, and by thinning `pyim_service` into clearer collectors.

**Architecture:** Extend `pyimgano.pyim_contracts.PyimListRequest` with an optional `list_kind` field that derives inclusion flags from `pyimgano.pyim_list_spec` when present, while preserving legacy explicit-flag behavior for existing callers. Refactor `pyimgano.pyim_cli` to pass `list_kind` instead of re-computing derived booleans, and split `pyimgano.services.pyim_service` into small section collectors so orchestration is clearer and less coupled to one monolithic function body.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_contracts`, `pyimgano.pyim_cli`, `pyimgano.services.pyim_service`

---

### Task 1: Lock Request Normalization With Failing Tests

**Files:**
- Modify: `tests/test_pyim_contracts.py`
- Modify: `tests/test_pyim_cli.py`
- Modify: `tests/test_pyim_cli_recipes_and_datasets.py`
- Modify: `tests/test_pyim_service.py`

**Step 1: Write the failing test**

Add tests that prove:
- `PyimListRequest(list_kind="datasets")` derives `include_core_sections=False`, `include_recipes=False`, `include_datasets=True`
- `PyimListRequest()` keeps the legacy core-only defaults
- `pyimgano.pyim_cli` builds neutral requests with `list_kind` and without passing the derived include flags into the contract constructor
- service payload collection works when callers use `list_kind` instead of manual include booleans

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_service.py -v`
Expected: FAIL because the request contract does not yet normalize `list_kind` and the CLI still passes derived booleans.

### Task 2: Implement Request Normalization And Service Collectors

**Files:**
- Modify: `pyimgano/pyim_contracts.py`
- Modify: `pyimgano/pyim_cli.py`
- Modify: `pyimgano/services/pyim_service.py`

**Step 1: Write minimal implementation**

- Add optional `list_kind` to `PyimListRequest`
- Normalize inclusion flags in `PyimListRequest.__post_init__`
- Refactor `pyimgano.pyim_cli._build_list_request()` to pass `list_kind` and raw filters only
- Split `pyimgano.services.pyim_service` into focused collectors for core sections, recipes, and datasets

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_service.py -v`
Expected: PASS

### Task 3: Regression And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-request-normalization.md`
- Modify: `pyimgano/pyim_contracts.py`
- Modify: `pyimgano/pyim_cli.py`
- Modify: `pyimgano/services/pyim_service.py`
- Modify: `tests/test_pyim_contracts.py`
- Modify: `tests/test_pyim_cli.py`
- Modify: `tests/test_pyim_cli_recipes_and_datasets.py`
- Modify: `tests/test_pyim_service.py`

**Step 1: Run pyim regression suite**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-request-normalization.md pyimgano/pyim_contracts.py pyimgano/pyim_cli.py pyimgano/services/pyim_service.py tests/test_pyim_contracts.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_service.py`
Expected: no output
