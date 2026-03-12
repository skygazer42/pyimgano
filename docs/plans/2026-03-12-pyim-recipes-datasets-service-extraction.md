# Pyim Recipes And Datasets Service Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining direct recipe and dataset discovery imports from `pyim_cli` by extending `pyim_service` to serve those list modes too.

**Architecture:** Extend `PyimListRequest` with an option to skip core discovery sections when only recipes or datasets are requested. Update `collect_pyim_listing_payload(...)` so it can cheaply produce recipe-only or dataset-only payloads, then refactor `pyim_cli` to delegate the `--list recipes` and `--list datasets` branches through the service layer instead of importing those registries directly.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.services.pyim_service`, recipe registry, dataset converters.

---

### Task 1: Extend `pyim_service` For Recipe/Dataset-Only Requests

**Files:**
- Modify: `pyimgano/services/pyim_service.py`
- Test: `tests/test_pyim_service.py`

**Step 1: Write the failing test**

Extend `tests/test_pyim_service.py` with tests that prove:
- a request can skip core sections while still including recipe payloads
- a request can skip core sections while still including dataset payloads

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_pyim_service.py -v`
Expected: FAIL because `PyimListRequest` does not support core-section skipping yet.

**Step 3: Write minimal implementation**

- Add an `include_core_sections` flag to `PyimListRequest`
- Update `collect_pyim_listing_payload(...)` to avoid collecting core discovery sections when that flag is false
- Preserve the returned payload shape

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_service.py -v`
Expected: PASS

### Task 2: Refactor `pyim_cli` Recipe And Dataset Branches

**Files:**
- Modify: `pyimgano/pyim_cli.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`

**Step 1: Write the failing test**

Extend `tests/test_pyim_cli_recipes_and_datasets.py` so it proves:
- `pyim --list recipes` delegates payload collection to `pyim_service`
- `pyim --list datasets` delegates payload collection to `pyim_service`

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_service.py tests/test_pyim_cli_recipes_and_datasets.py -v`
Expected: FAIL because `pyim_cli` still imports recipes and datasets directly.

**Step 3: Write minimal implementation**

- Replace the `recipes` early-return branch with a `pyim_service.collect_pyim_listing_payload(...)` call
- Replace the `datasets` early-return branch the same way
- Remove now-unused direct imports/helpers from `pyim_cli`

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_service.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli.py tests/test_pyim_cli_model_presets.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Test: `tests/test_pyim_service.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`
- Test: `tests/test_pyim_cli_model_presets.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_model_presets.py tests/test_services_package.py tests/test_service_import_style.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-recipes-datasets-service-extraction.md pyimgano/services/pyim_service.py pyimgano/pyim_cli.py tests/test_pyim_service.py tests/test_pyim_cli_recipes_and_datasets.py`
Expected: no output
