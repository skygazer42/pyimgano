# Pyim Section Item Typing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce the remaining stringly-typed structure inside `PyimListPayload` by turning the stable `pyim --list` section items into explicit contract types.

**Architecture:** Extend `pyimgano.pyim_contracts` with typed section items for the stable discovery-oriented payload slices: families/types, years, metadata contract fields, preprocessing schemes, and dataset summaries. Update `PyimListPayload` to hold those typed objects and preserve the existing JSON payload shapes through explicit `to_payload()` methods. Keep dynamic shapes such as recipe info and model preset info as-is for now.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_contracts`, existing `pyimgano.services.pyim_service`, existing `pyimgano.pyim_cli_rendering`.

---

### Task 1: Add Typed Section Item Contracts

**Files:**
- Modify: `pyimgano/pyim_contracts.py`
- Test: `tests/test_pyim_contracts.py`
- Test: `tests/test_pyim_service.py`

**Step 1: Write the failing test**

Add tests that prove:
- stable section items are represented by explicit contracts
- `PyimListPayload.to_all_json_payload()` preserves current JSON shapes, including:
  - family/type summaries still expose `tags` and `sample_models`
  - year summaries preserve `year=None` for `unknown`
  - `pre-2001` preserves `year_start=None` and `year_end`
  - preprocessing items omit optional `entrypoint/config_key/payload` keys when absent

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_service.py -v`
Expected: FAIL because the section items are still plain dicts.

**Step 3: Write minimal implementation**

- Add typed item dataclasses to `pyimgano.pyim_contracts`
- Add `from_mapping(...)` and `to_payload()` helpers as needed
- Update `PyimListPayload` to use those typed item lists

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_service.py -v`
Expected: PASS

### Task 2: Refactor Service And Rendering To Consume Typed Items

**Files:**
- Modify: `pyimgano/services/pyim_service.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli.py`

**Step 1: Write the failing test**

Extend rendering-oriented tests so they prove:
- typed section items reach `pyim_cli_rendering`
- text rendering reads typed attributes instead of raw dict keys
- JSON output still matches the previous external shape

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py tests/test_pyim_cli.py -v`
Expected: FAIL because the service still emits dict items and rendering still expects mapping-like access.

**Step 3: Write minimal implementation**

- Convert service outputs into typed item objects
- Update rendering helpers to use typed attributes where section items are now explicit contracts
- Keep payload coercion only as a compatibility shim

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py tests/test_pyim_cli.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Modify: `pyimgano/pyim_contracts.py`
- Modify: `pyimgano/services/pyim_service.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Test: `tests/test_pyim_contracts.py`
- Test: `tests/test_pyim_service.py`
- Test: `tests/test_pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_model_presets.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-section-item-typing.md pyimgano/pyim_contracts.py pyimgano/services/pyim_service.py pyimgano/pyim_cli_rendering.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli.py`
Expected: no output
