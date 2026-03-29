# Infer Pipeline Engineering Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the internal engineering quality of the `pyimgano-infer` path without changing any existing public API or CLI behavior.

**Architecture:** Keep `pyimgano/infer_cli.py` as the stable public entrypoint, but thin it into orchestration-only code by moving parsing, discovery dispatch, ONNX-sweep helpers, and profile payload assembly behind narrow internal modules. Strengthen `infer_context_service`, `infer_load_service`, `infer_runtime_service`, `inference_service`, `infer_continue_service`, `infer_output_service`, and `infer_artifact_service` so each stage carries more explicit and consistent contracts.

**Tech Stack:** Python 3.9+, argparse, dataclasses, pytest, JSONL outputs, existing `pyimgano.services`, `pyimgano.workbench`, and inference/reporting helpers.

---

## Stream A: Thin `infer_cli.py` Without Changing Behavior

### Task 1: Add failing tests for CLI parsing helpers
- Files:
  - Create: `tests/test_infer_cli_inputs.py`
  - Test: `pyimgano/infer_cli.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_inputs.py -q`

### Task 2: Create `infer_cli_inputs` and move parsing/input collection helpers
- Files:
  - Create: `pyimgano/infer_cli_inputs.py`
  - Modify: `pyimgano/infer_cli.py`
  - Test: `tests/test_infer_cli_inputs.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_inputs.py tests/test_infer_cli_smoke.py -q`

### Task 3: Add failing tests for discovery command dispatch
- Files:
  - Create: `tests/test_infer_cli_discovery.py`
  - Test: `pyimgano/infer_cli.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_discovery.py -q`

### Task 4: Create `infer_cli_discovery` and move listing/info branches
- Files:
  - Create: `pyimgano/infer_cli_discovery.py`
  - Modify: `pyimgano/infer_cli.py`
  - Test: `tests/test_infer_cli_discovery.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_discovery.py tests/test_infer_cli_smoke.py -q`

### Task 5: Add failing tests for ONNX sweep helper extraction
- Files:
  - Create: `tests/test_infer_cli_onnx.py`
  - Test: `pyimgano/infer_cli.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_onnx.py -q`

### Task 6: Create `infer_cli_onnx` and move sweep/session-option helpers
- Files:
  - Create: `pyimgano/infer_cli_onnx.py`
  - Modify: `pyimgano/infer_cli.py`
  - Test: `tests/test_infer_cli_onnx.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_onnx.py tests/test_infer_cli_onnx_session_options_v1.py -q`

### Task 7: Add failing tests for profile payload helper extraction
- Files:
  - Create: `tests/test_infer_cli_profile.py`
  - Test: `pyimgano/infer_cli.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_profile.py -q`

### Task 8: Create `infer_cli_profile` and thin profile assembly in `infer_cli.py`
- Files:
  - Create: `pyimgano/infer_cli_profile.py`
  - Modify: `pyimgano/infer_cli.py`
  - Test: `tests/test_infer_cli_profile.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_profile.py tests/test_infer_cli_smoke.py -q`

## Stream B: Harden Context, Load, and Runtime Contracts

### Task 9: Add failing tests for context postprocess summary normalization
- Files:
  - Modify: `tests/test_infer_context_service.py`
  - Test: `pyimgano/services/infer_context_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_context_service.py -q`

### Task 10: Extract summary normalization helpers inside `infer_context_service`
- Files:
  - Modify: `pyimgano/services/infer_context_service.py`
  - Test: `tests/test_infer_context_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_context_service.py tests/test_infer_cli_infer_config.py -q`

### Task 11: Add failing tests for context warning and payload copy semantics
- Files:
  - Modify: `tests/test_infer_context_service.py`
  - Modify: `tests/test_infer_cli_from_run.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_context_service.py tests/test_infer_cli_from_run.py -q`

### Task 12: Normalize warning/payload copying in `infer_context_service`
- Files:
  - Modify: `pyimgano/services/infer_context_service.py`
  - Test: `tests/test_infer_context_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_context_service.py tests/test_infer_cli_from_run.py tests/test_infer_cli_infer_config.py -q`

### Task 13: Add failing tests for shared load option assembly
- Files:
  - Modify: `tests/test_infer_load_service.py`
  - Modify: `tests/test_infer_cli_infer_config.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_load_service.py tests/test_infer_cli_infer_config.py -q`

### Task 14: Extract shared option assembly helpers in `infer_load_service`
- Files:
  - Modify: `pyimgano/services/infer_load_service.py`
  - Test: `tests/test_infer_load_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_load_service.py tests/test_infer_cli_from_run.py tests/test_infer_cli_infer_config.py -q`

### Task 15: Add failing tests for runtime summary/provenance shaping
- Files:
  - Modify: `tests/test_infer_runtime_service.py`
  - Modify: `tests/test_infer_cli_from_run.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_runtime_service.py tests/test_infer_cli_from_run.py -q`

### Task 16: Normalize runtime summary/provenance helpers in `infer_runtime_service`
- Files:
  - Modify: `pyimgano/services/infer_runtime_service.py`
  - Test: `tests/test_infer_runtime_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_runtime_service.py tests/test_infer_cli_infer_config.py tests/test_infer_cli_from_run.py -q`

## Stream C: Unify Execution, Error, and Artifact Metadata

### Task 17: Add failing tests for inference decision summary normalization
- Files:
  - Modify: `tests/test_inference_service.py`
  - Modify: `tests/test_inference_api.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_inference_service.py tests/test_inference_api.py -q`

### Task 18: Refactor decision summary helpers in `inference_service`
- Files:
  - Modify: `pyimgano/services/inference_service.py`
  - Test: `tests/test_inference_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_inference_service.py tests/test_inference_api.py tests/test_inference_api_tuple_outputs.py -q`

### Task 19: Add failing tests for continue-on-error stage/error bookkeeping
- Files:
  - Modify: `tests/test_infer_continue_service.py`
  - Modify: `tests/test_infer_cli_production_guardrails_v1.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_continue_service.py tests/test_infer_cli_production_guardrails_v1.py -q`

### Task 20: Normalize chunk/fallback error bookkeeping in `infer_continue_service`
- Files:
  - Modify: `pyimgano/services/infer_continue_service.py`
  - Test: `tests/test_infer_continue_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_continue_service.py tests/test_infer_cli_production_guardrails_v1.py -q`

### Task 21: Add failing tests for artifact option parity across builder paths
- Files:
  - Modify: `tests/test_infer_artifact_service.py`
  - Modify: `tests/test_infer_cli_smoke.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py -q`

### Task 22: Extract shared artifact-option normalization in `infer_artifact_service`
- Files:
  - Modify: `pyimgano/services/infer_artifact_service.py`
  - Test: `tests/test_infer_artifact_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_artifact_service.py tests/test_infer_cli_defects_regions_jsonl.py -q`

### Task 23: Add failing tests for infer output/error record consistency
- Files:
  - Modify: `tests/test_infer_output_service.py`
  - Modify: `tests/test_infer_continue_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_output_service.py tests/test_infer_continue_service.py -q`

### Task 24: Normalize output/error record helpers in `infer_output_service`
- Files:
  - Modify: `pyimgano/services/infer_output_service.py`
  - Test: `tests/test_infer_output_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_output_service.py tests/test_infer_continue_service.py tests/test_infer_cli_smoke.py -q`

## Stream D: Lock Down Boundaries and Public Contracts

### Task 25: Add failing architecture tests for new infer CLI helper modules
- Files:
  - Modify: `tests/test_architecture_boundaries.py`
  - Test: `pyimgano/infer_cli_inputs.py`
  - Test: `pyimgano/infer_cli_discovery.py`
  - Test: `pyimgano/infer_cli_onnx.py`
  - Test: `pyimgano/infer_cli_profile.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`

### Task 26: Add explicit `__all__` and import boundaries for infer CLI helpers
- Files:
  - Modify: `pyimgano/infer_cli_inputs.py`
  - Modify: `pyimgano/infer_cli_discovery.py`
  - Modify: `pyimgano/infer_cli_onnx.py`
  - Modify: `pyimgano/infer_cli_profile.py`
  - Test: `tests/test_architecture_boundaries.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_architecture_boundaries.py tests/test_infer_cli_inputs.py tests/test_infer_cli_discovery.py tests/test_infer_cli_onnx.py tests/test_infer_cli_profile.py -q`

### Task 27: Add failing tests for selected infer dataclass contract stability
- Files:
  - Create: `tests/test_infer_service_contracts.py`
  - Test: `pyimgano/services/infer_context_service.py`
  - Test: `pyimgano/services/infer_load_service.py`
  - Test: `pyimgano/services/infer_runtime_service.py`
  - Test: `pyimgano/services/infer_output_service.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_service_contracts.py -q`

### Task 28: Normalize dataclass copy/serialization expectations in infer services
- Files:
  - Modify: `pyimgano/services/infer_context_service.py`
  - Modify: `pyimgano/services/infer_load_service.py`
  - Modify: `pyimgano/services/infer_runtime_service.py`
  - Modify: `pyimgano/services/infer_output_service.py`
  - Test: `tests/test_infer_service_contracts.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_service_contracts.py tests/test_infer_context_service.py tests/test_infer_load_service.py tests/test_infer_runtime_service.py tests/test_infer_output_service.py -q`

### Task 29: Add failing tests for shared postprocess-summary parity across inference paths
- Files:
  - Modify: `tests/test_infer_cli_smoke.py`
  - Modify: `tests/test_infer_cli_from_run.py`
  - Modify: `tests/test_infer_cli_infer_config.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_smoke.py tests/test_infer_cli_from_run.py tests/test_infer_cli_infer_config.py -q`

### Task 30: Align direct/config/run-backed postprocess summary wiring
- Files:
  - Modify: `pyimgano/infer_cli.py`
  - Modify: `pyimgano/services/infer_context_service.py`
  - Modify: `pyimgano/services/infer_runtime_service.py`
  - Test: `tests/test_infer_cli_smoke.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_smoke.py tests/test_infer_cli_from_run.py tests/test_infer_cli_infer_config.py tests/test_infer_runtime_service.py -q`

### Task 31: Add failing tests for profile JSON payload shaping
- Files:
  - Modify: `tests/test_infer_cli_profile.py`
  - Modify: `tests/test_infer_cli_smoke.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_profile.py tests/test_infer_cli_smoke.py -q`

### Task 32: Align profile payload shape and error counts in CLI helpers
- Files:
  - Modify: `pyimgano/infer_cli_profile.py`
  - Modify: `pyimgano/infer_cli.py`
  - Test: `tests/test_infer_cli_profile.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_profile.py tests/test_infer_cli_smoke.py tests/test_infer_cli_production_guardrails_v1.py -q`

## Stream E: Docs and Regression Safety

### Task 33: Add failing tests for inference docs command references
- Files:
  - Create: `tests/test_infer_docs_contract.py`
  - Test: `docs/CLI_REFERENCE.md`
  - Test: `docs/INDUSTRIAL_INFERENCE.md`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_docs_contract.py -q`

### Task 34: Document the direct / infer-config / from-run inference paths
- Files:
  - Modify: `docs/CLI_REFERENCE.md`
  - Test: `tests/test_infer_docs_contract.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_docs_contract.py -q`

### Task 35: Document runtime summaries, postprocess provenance, and defect-threshold context
- Files:
  - Modify: `docs/INDUSTRIAL_INFERENCE.md`
  - Modify: `README.md`
  - Test: `tests/test_infer_docs_contract.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_docs_contract.py -q`

### Task 36: Update help-text-sensitive docs for CLI profiling and artifacts
- Files:
  - Modify: `docs/CLI_REFERENCE.md`
  - Modify: `README.md`
  - Test: `tests/test_infer_docs_contract.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_docs_contract.py -q`

### Task 37: Add thin end-to-end regression for direct inference path
- Files:
  - Modify: `tests/test_infer_cli_smoke.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_smoke.py -q`

### Task 38: Add thin end-to-end regression for `--infer-config`
- Files:
  - Modify: `tests/test_infer_cli_infer_config.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_infer_config.py -q`

### Task 39: Add thin end-to-end regression for `--from-run`
- Files:
  - Modify: `tests/test_infer_cli_from_run.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_from_run.py -q`

### Task 40: Run the consolidated inference-boundary verification suite
- Files:
  - Verify only:
    - `tests/test_infer_cli_inputs.py`
    - `tests/test_infer_cli_discovery.py`
    - `tests/test_infer_cli_onnx.py`
    - `tests/test_infer_cli_profile.py`
    - `tests/test_infer_service_contracts.py`
    - `tests/test_infer_docs_contract.py`
    - `tests/test_infer_context_service.py`
    - `tests/test_infer_load_service.py`
    - `tests/test_infer_runtime_service.py`
    - `tests/test_inference_service.py`
    - `tests/test_infer_continue_service.py`
    - `tests/test_infer_output_service.py`
    - `tests/test_infer_artifact_service.py`
    - `tests/test_infer_cli_smoke.py`
    - `tests/test_infer_cli_from_run.py`
    - `tests/test_infer_cli_infer_config.py`
    - `tests/test_architecture_boundaries.py`
- Verify:
  - `python3 -m pytest --no-cov tests/test_infer_cli_inputs.py tests/test_infer_cli_discovery.py tests/test_infer_cli_onnx.py tests/test_infer_cli_profile.py tests/test_infer_service_contracts.py tests/test_infer_docs_contract.py tests/test_infer_context_service.py tests/test_infer_load_service.py tests/test_infer_runtime_service.py tests/test_inference_service.py tests/test_infer_continue_service.py tests/test_infer_output_service.py tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py tests/test_infer_cli_from_run.py tests/test_infer_cli_infer_config.py tests/test_architecture_boundaries.py -q`
