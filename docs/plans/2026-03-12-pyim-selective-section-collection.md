# Pyim Selective Section Collection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano.services.pyim_service` collect only the payload sections required for the current `pyim` request instead of eagerly materializing every core section for single-section listings.

**Architecture:** Extend `pyimgano.pyim_list_spec` with the payload field names needed by each `list_kind`, and add a `PyimListRequest` helper that resolves the requested payload fields while preserving legacy and explicit-override behavior. Refactor `pyimgano.services.pyim_service` into a field-collector map that iterates over requested fields rather than hardcoding broad core/recipes/datasets branches.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_list_spec`, `pyimgano.pyim_contracts`, `pyimgano.services.pyim_service`

---

### Task 1: Lock Requested-Field Semantics With Failing Tests

**Files:**
- Modify: `tests/test_pyim_list_spec.py`
- Modify: `tests/test_pyim_contracts.py`
- Modify: `tests/test_pyim_service.py`

**Step 1: Write the failing test**

Add tests that prove:
- `model-presets` declares both `model_presets` and `model_preset_infos` as requested payload fields
- `PyimListRequest(list_kind="families")` resolves only the `families` field
- `PyimListRequest(list_kind="datasets", include_core_sections=True)` still includes the core fields plus `datasets`
- `collect_pyim_listing_payload(PyimListRequest(list_kind="families"))` populates only `families`
- `collect_pyim_listing_payload(PyimListRequest(list_kind="model-presets"))` populates only preset-related fields

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py -v`
Expected: FAIL because the spec does not yet expose requested payload fields and the service still collects all core sections for specific core listings.

### Task 2: Implement Requested-Field Metadata And Selective Collection

**Files:**
- Modify: `pyimgano/pyim_list_spec.py`
- Modify: `pyimgano/pyim_contracts.py`
- Modify: `pyimgano/services/pyim_service.py`

**Step 1: Write minimal implementation**

- Add requested payload field metadata to the shared list-kind spec
- Add a `PyimListRequest.requested_payload_fields()` helper that resolves the effective field list
- Refactor `pyimgano.services.pyim_service` to use a payload-field collector map and populate only requested fields
- Preserve legacy behavior when no `list_kind` is provided

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py -v`
Expected: PASS

### Task 3: Regression And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-selective-section-collection.md`
- Modify: `pyimgano/pyim_list_spec.py`
- Modify: `pyimgano/pyim_contracts.py`
- Modify: `pyimgano/services/pyim_service.py`
- Modify: `tests/test_pyim_list_spec.py`
- Modify: `tests/test_pyim_contracts.py`
- Modify: `tests/test_pyim_service.py`

**Step 1: Run pyim regression suite**

Run: `pytest --no-cov tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-selective-section-collection.md pyimgano/pyim_list_spec.py pyimgano/pyim_contracts.py pyimgano/services/pyim_service.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py`
Expected: no output
