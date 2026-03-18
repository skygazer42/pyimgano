# Inference Rejection Config Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make confidence-based rejection policy exportable and reproducible through workbench config, `infer_config.json`, and `pyimgano-infer`.

**Architecture:** Introduce a dedicated top-level `prediction` config section in workbench and infer-config payloads. Parse and validate `prediction.reject_confidence_below` / `prediction.reject_label`, export them from workbench artifacts, carry them through infer-context loading, and apply them as CLI defaults only when the user did not override them explicitly.

**Tech Stack:** Python 3.9+, dataclasses, argparse, existing workbench config parser, existing infer context / CLI services, pytest.

---

## Stream A: Config Surface

### Task 1: Add failing config parsing and export tests

**Files:**
- Modify: `tests/test_workbench_config.py`
- Modify: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_workbench_export_infer_config.py`

**Verify:**
- `pytest --no-cov tests/test_workbench_config.py tests/test_workbench_config_parser.py tests/test_workbench_export_infer_config.py -q`

### Task 2: Implement `prediction` config model and parser

**Files:**
- Modify: `pyimgano/workbench/config_types.py`
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/workbench/config_parser.py`
- Modify: `pyimgano/workbench/config_section_parsers.py`
- Modify: `pyimgano/workbench/config_model_output_section_parser.py`
- Create: `pyimgano/workbench/config_prediction_section_parser.py`
- Modify: `pyimgano/services/workbench_service.py`

**Verify:**
- `pytest --no-cov tests/test_workbench_config.py tests/test_workbench_config_parser.py tests/test_workbench_export_infer_config.py -q`

## Stream B: Infer Runtime Defaults

### Task 3: Add failing infer-context and CLI tests

**Files:**
- Modify: `tests/test_infer_context_service.py`
- Modify: `tests/test_infer_cli_infer_config.py`
- Modify: `tests/test_infer_cli_from_run.py`

**Verify:**
- `pytest --no-cov tests/test_infer_context_service.py tests/test_infer_cli_infer_config.py tests/test_infer_cli_from_run.py -q`

### Task 4: Load and apply prediction defaults

**Files:**
- Modify: `pyimgano/services/infer_context_service.py`
- Modify: `pyimgano/infer_cli.py`

**Verify:**
- `pytest --no-cov tests/test_infer_context_service.py tests/test_infer_cli_infer_config.py tests/test_infer_cli_from_run.py -q`

## Stream C: Validation and Docs

### Task 5: Add failing infer-config validation tests

**Files:**
- Modify: `tests/test_validate_infer_config_cli.py`

**Verify:**
- `pytest --no-cov tests/test_validate_infer_config_cli.py -q`

### Task 6: Validate `prediction` payloads and document the surface

**Files:**
- Modify: `pyimgano/inference/validate_infer_config.py`
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `docs/INDUSTRIAL_INFERENCE.md`
- Modify: `docs/WORKBENCH.md`

**Verify:**
- `pytest --no-cov tests/test_validate_infer_config_cli.py -q`

## Final Verification

### Task 7: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_workbench_config.py tests/test_workbench_config_parser.py tests/test_workbench_export_infer_config.py tests/test_infer_context_service.py tests/test_infer_cli_infer_config.py tests/test_infer_cli_from_run.py tests/test_validate_infer_config_cli.py -q`
- `python -m py_compile pyimgano/infer_cli.py pyimgano/inference/validate_infer_config.py pyimgano/services/infer_context_service.py pyimgano/services/workbench_service.py pyimgano/workbench/config_parser.py pyimgano/workbench/config_prediction_section_parser.py pyimgano/workbench/config_section_parsers.py pyimgano/workbench/config_types.py`
- `git diff --check`
