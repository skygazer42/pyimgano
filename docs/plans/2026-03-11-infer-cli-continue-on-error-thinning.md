# Infer CLI Continue-On-Error Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract `infer_cli` continue-on-error batch/fallback orchestration into a dedicated service while keeping artifact materialization and output writing in the CLI.

**Architecture:** Add a focused service that owns best-effort inference control flow: chunking, batch execution, fallback to per-input inference, error stage classification, and `max-errors` early stop. The CLI will keep detector setup, output target lifecycle, and `_process_ok_result`, but delegate continue-on-error orchestration through explicit request/result dataclasses and callbacks.

**Tech Stack:** Python 3.10, pytest, dataclasses, callback-based service boundaries

---

### Task 1: Add failing orchestration tests

**Files:**
- Create: `tests/test_infer_continue_service.py`
- Modify: `tests/test_infer_cli_production_guardrails_v1.py`

**Step 1: Write the failing tests**

```python
def test_run_continue_on_error_inference_falls_back_to_per_input_when_batch_fails():
    ...

def test_run_continue_on_error_inference_records_artifact_stage_errors():
    ...

def test_infer_cli_continue_on_error_delegates_to_service():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_infer_continue_service.py tests/test_infer_cli_production_guardrails_v1.py -q`
Expected: FAIL because `pyimgano.services.infer_continue_service` does not exist yet.

### Task 2: Implement continue-on-error service and wire CLI

**Files:**
- Create: `pyimgano/services/infer_continue_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`

**Step 1: Write minimal implementation**

```python
@dataclass(frozen=True)
class ContinueOnErrorInferRequest: ...

def run_continue_on_error_inference(...): ...
```

**Step 2: Refactor CLI to delegate**

```python
continue_result = infer_continue_service.run_continue_on_error_inference(...)
errors = continue_result.errors
processed = continue_result.processed
infer_timing.seconds += continue_result.timing_seconds
```

**Step 3: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_infer_continue_service.py tests/test_infer_cli_production_guardrails_v1.py -q`
Expected: PASS

### Task 3: Run focused infer regression coverage

**Files:**
- Test: `tests/test_infer_cli_smoke.py`
- Test: `tests/test_infer_cli_infer_config.py`
- Test: `tests/test_infer_cli_from_run.py`
- Test: `tests/test_integration_workbench_train_then_infer.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_infer_continue_service.py tests/test_infer_output_service.py tests/test_infer_runtime_service.py tests/test_infer_artifact_service.py tests/test_infer_setup_service.py tests/test_infer_context_service.py tests/test_infer_cli_smoke.py tests/test_infer_cli_defects_regions_jsonl.py tests/test_infer_cli_maps_vs_defects_flags.py tests/test_infer_cli_infer_config.py tests/test_infer_cli_from_run.py tests/test_infer_cli_from_run_errors.py tests/test_infer_cli_production_guardrails_v1.py tests/test_infer_cli_onnx_session_options_v1.py tests/test_integration_workbench_train_then_infer.py -q`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- pyimgano/infer_cli.py pyimgano/services/__init__.py pyimgano/services/infer_continue_service.py tests/test_infer_continue_service.py tests/test_infer_cli_production_guardrails_v1.py docs/plans/2026-03-11-infer-cli-continue-on-error-thinning.md`
Expected: no output
