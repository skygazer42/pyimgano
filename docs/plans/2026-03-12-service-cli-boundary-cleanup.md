# Service-CLI Boundary Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the current reverse dependencies from `pyimgano.services` into CLI-oriented helper modules and add architecture guard tests for those boundaries.

**Architecture:** Introduce neutral modules for preset catalog access and model-kwarg support, then convert `pyimgano.cli_presets` and `pyimgano.cli_common` into compatibility adapters over those neutral modules. Update service-layer imports to depend only on the neutral modules, and add tests that fail if service modules import CLI helper modules again.

**Tech Stack:** Python 3.10, pytest, `ast`, dataclasses, existing `pyimgano` registry and preset helpers.

---

### Task 1: Extract a Neutral Preset Catalog Module

**Files:**
- Create: `pyimgano/presets/catalog.py`
- Modify: `pyimgano/cli_presets.py`
- Modify: `pyimgano/services/model_options.py`
- Modify: `pyimgano/services/infer_options_service.py`
- Modify: `pyimgano/services/discovery_service.py`
- Modify: `pyimgano/services/doctor_service.py`
- Test: `tests/test_preset_catalog.py`
- Test: `tests/test_pyim_cli_model_presets.py`

**Step 1: Write the failing test**

Add `tests/test_preset_catalog.py` with tests that:
- resolve a model preset through `pyimgano.presets.catalog`
- resolve a defects preset through `pyimgano.presets.catalog`
- prove `pyimgano.cli_presets.resolve_model_preset(...)` delegates to the catalog module

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_preset_catalog.py -v`
Expected: FAIL because `pyimgano.presets.catalog` does not exist and `cli_presets.py` is not a wrapper yet.

**Step 3: Write minimal implementation**

- Move the preset dataclasses and lookup helpers from `pyimgano/cli_presets.py` into `pyimgano/presets/catalog.py`
- Keep `pyimgano/cli_presets.py` as a thin compatibility wrapper that re-exports or delegates to the catalog module
- Update service-layer modules to import from `pyimgano.presets.catalog`, not from `pyimgano.cli_presets`

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_preset_catalog.py tests/test_pyim_cli_model_presets.py -v`
Expected: PASS

### Task 2: Extract Neutral Model-Kwarg Support

**Files:**
- Create: `pyimgano/models/model_kwargs.py`
- Modify: `pyimgano/cli_common.py`
- Modify: `pyimgano/services/model_options.py`
- Test: `tests/test_model_kwargs_module.py`
- Test: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Add `tests/test_model_kwargs_module.py` with tests that:
- merge `checkpoint_path` through `pyimgano.models.model_kwargs.merge_checkpoint_path(...)`
- validate kwargs through `pyimgano.models.model_kwargs.validate_user_model_kwargs(...)`
- prove `pyimgano.cli_common.build_model_kwargs(...)` delegates to the neutral module

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_model_kwargs_module.py tests/test_cli_model_kwargs.py -v`
Expected: FAIL because `pyimgano.models.model_kwargs` does not exist and `cli_common.py` still owns the implementation.

**Step 3: Write minimal implementation**

- Move `_get_model_signature_info`, `merge_checkpoint_path`, `validate_user_model_kwargs`, and `build_model_kwargs` into `pyimgano/models/model_kwargs.py`
- Keep `parse_model_kwargs(...)` in `cli_common.py` because its error messages are explicitly CLI-shaped
- Convert the moved functions in `cli_common.py` into thin wrappers over the new neutral module
- Update `pyimgano.services.model_options` to import from `pyimgano.models.model_kwargs`

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_model_kwargs_module.py tests/test_cli_model_kwargs.py tests/test_model_options_service.py -v`
Expected: PASS

### Task 3: Add Architecture Guard Tests

**Files:**
- Create: `tests/test_architecture_boundaries.py`
- Test: `tests/test_discovery_service.py`
- Test: `tests/test_doctor_service.py`
- Test: `tests/test_infer_options_service.py`

**Step 1: Write the failing test**

Create `tests/test_architecture_boundaries.py` with an AST-based test that scans `pyimgano/services/*.py` and fails if any service module imports:
- `pyimgano.cli_common`
- `pyimgano.cli_presets`

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_architecture_boundaries.py -v`
Expected: FAIL until Tasks 1 and 2 are complete.

**Step 3: Write minimal implementation**

- Finish any remaining service imports that still point at CLI helper modules
- Keep adapter-layer imports in CLI modules intact

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_architecture_boundaries.py tests/test_discovery_service.py tests/test_doctor_service.py tests/test_infer_options_service.py tests/test_model_options_service.py -v`
Expected: PASS

### Task 4: Run Focused Regression Coverage

**Files:**
- Test: `tests/test_cli_common.py`
- Test: `tests/test_cli_presets_preprocessing.py`
- Test: `tests/test_cli_presets_defects_fp40.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Test: `tests/test_infer_cli_preprocessing_preset.py`
- Test: `tests/test_infer_cli_onnx_session_options_v1.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_preset_catalog.py tests/test_model_kwargs_module.py tests/test_architecture_boundaries.py tests/test_cli_common.py tests/test_cli_model_kwargs.py tests/test_cli_presets_preprocessing.py tests/test_cli_presets_defects_fp40.py tests/test_pyim_cli_model_presets.py tests/test_discovery_service.py tests/test_doctor_service.py tests/test_infer_options_service.py tests/test_model_options_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_infer_cli_preprocessing_preset.py tests/test_infer_cli_onnx_session_options_v1.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- pyimgano/cli_common.py pyimgano/cli_presets.py pyimgano/models/model_kwargs.py pyimgano/presets/catalog.py pyimgano/services/model_options.py pyimgano/services/infer_options_service.py pyimgano/services/discovery_service.py pyimgano/services/doctor_service.py tests/test_preset_catalog.py tests/test_model_kwargs_module.py tests/test_architecture_boundaries.py`
Expected: no output
