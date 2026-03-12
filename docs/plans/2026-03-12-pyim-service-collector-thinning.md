# Pyim Service Collector Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyim_service` coupling by extracting payload field collection details into a dedicated helper module while preserving the existing `collect_pyim_listing_payload()` API and output shape.

**Architecture:** Introduce a `pyimgano.services.pyim_payload_collectors` module that owns field-by-field collection functions, typed conversion helpers, and empty payload initialization. Refactor `pyimgano.services.pyim_service` into a thin orchestration facade that only iterates requested fields and constructs `PyimListPayload`, and add architecture tests so domain-specific imports do not creep back into the service facade.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.pyim_contracts`, `pyimgano.pyim_list_spec`, `pyimgano.services`

---

### Task 1: Lock The Collector Boundary With Failing Tests

**Files:**
- Create: `tests/test_pyim_payload_collectors.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated collector helper can build an empty payload kwargs dict covering every shared payload field
- the helper can collect typed dataset summaries for the `datasets` field
- the helper rejects unsupported payload field names
- `pyimgano/services/pyim_service.py` does not directly import discovery, registry, recipe, dataset, or preset modules once the refactor is complete

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_payload_collectors.py tests/test_architecture_boundaries.py::test_pyim_service_uses_payload_collector_boundary -v`
Expected: FAIL because the new collector module does not exist yet and `pyim_service.py` still directly imports data-source modules inside collector functions.

### Task 2: Extract The Collector Module

**Files:**
- Create: `pyimgano/services/pyim_payload_collectors.py`
- Modify: `pyimgano/services/pyim_service.py`

**Step 1: Write minimal implementation**

- Move tag normalization, typed conversion helpers, and field-specific collectors into `pyim_payload_collectors.py`
- Export a small API for empty payload kwargs construction and per-field collection
- Refactor `collect_pyim_listing_payload()` to delegate through that helper module
- Preserve current typed payload construction and output behavior exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_payload_collectors.py tests/test_pyim_service.py tests/test_architecture_boundaries.py::test_pyim_service_uses_payload_collector_boundary -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-pyim-service-collector-thinning.md`
- Modify: `pyimgano/services/pyim_payload_collectors.py`
- Modify: `pyimgano/services/pyim_service.py`
- Modify: `tests/test_pyim_payload_collectors.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_pyim_payload_collectors.py tests/test_pyim_section_views.py tests/test_pyim_list_spec.py tests/test_pyim_contracts.py tests/test_pyim_service.py tests/test_pyim_cli.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py tests/test_architecture_boundaries.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-service-collector-thinning.md pyimgano/services/pyim_payload_collectors.py pyimgano/services/pyim_service.py tests/test_pyim_payload_collectors.py tests/test_architecture_boundaries.py`
Expected: no output
