# Pyim Service Output Typing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano.services.pyim_service` construct stable `pyim --list` sections as explicit contract objects at the service boundary instead of passing raw mappings into `PyimListPayload`.

**Architecture:** Keep `pyimgano.pyim_contracts` as the neutral cross-layer contract module and treat `pyimgano.services.pyim_service` as the translation boundary from discovery/registry/dataset sources into those contracts. Preserve `PyimListPayload.__post_init__` as a compatibility shim for callers that still hand in mappings, but stop relying on that shim from the main service path.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_contracts`, existing `pyimgano.services.pyim_service`

---

### Task 1: Lock The Service Boundary With A Failing Test

**Files:**
- Modify: `tests/test_pyim_service.py`

**Step 1: Write the failing test**

Add a test that monkeypatches `pyimgano.services.pyim_service.PyimListPayload` with a capture stub and proves the service passes:
- `PyimModelFacetSummary` items for `families` and `types`
- `PyimYearSummary` items for `years`
- `PyimMetadataContractField` items for `metadata_contract`
- `PyimPreprocessingSchemeSummary` items for `preprocessing`
- `PyimDatasetSummary` items for `datasets`

Keep `recipes` and `model_preset_infos` as dict payloads.

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_pyim_service.py::test_collect_pyim_listing_payload_builds_typed_sections_before_payload_coercion -v`
Expected: FAIL because the service still passes raw dicts into `PyimListPayload`.

### Task 2: Move Section Translation Into The Service Layer

**Files:**
- Modify: `pyimgano/services/pyim_service.py`

**Step 1: Write minimal implementation**

- Import the stable contract types from `pyimgano.pyim_contracts`
- Add small helper functions that translate raw discovery mappings into typed contract lists
- Build datasets as `PyimDatasetSummary` instances directly
- Keep recipes and model preset infos unchanged

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_service.py tests/test_pyim_contracts.py tests/test_pyim_cli_rendering.py -v`
Expected: PASS

### Task 3: Regression And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-service-output-typing.md`
- Modify: `pyimgano/services/pyim_service.py`
- Modify: `tests/test_pyim_service.py`

**Step 1: Run pyim regression suite**

Run: `pytest --no-cov tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-service-output-typing.md pyimgano/services/pyim_service.py tests/test_pyim_service.py`
Expected: no output
