# Infer CLI Artifact Options Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining CLI-to-artifact field mapping from `infer_cli` so the CLI no longer constructs artifact-domain option payloads directly.

**Architecture:** Keep `infer_cli` responsible for outer orchestration, runtime values, and file lifecycle, but move the long `DefectsArtifactConfigBuildRequest(...)` argument mapping into `infer_artifact_service` behind a dedicated options/request builder. The artifact service should own both the artifact request structure and the domain-specific option normalization that feeds it.

**Tech Stack:** Python 3.10, pytest, dataclasses

---

### Task 1: Add failing tests for artifact option assembly

**Files:**
- Modify: `tests/test_infer_artifact_service.py`
- Modify: `tests/test_infer_cli_smoke.py`

**Step 1: Write the failing tests**

```python
def test_build_infer_result_artifact_build_request_from_options_populates_defects_fields():
    ...

def test_build_infer_result_artifact_build_request_from_options_skips_defects_when_disabled():
    ...

def test_infer_cli_smoke_delegates_artifact_option_mapping_to_service():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py -q`
Expected: FAIL because the new options dataclass/builder do not exist yet.

### Task 2: Implement artifact option builder and refactor CLI

**Files:**
- Modify: `pyimgano/services/infer_artifact_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`

**Step 1: Add dedicated options dataclasses**

```python
@dataclass(frozen=True)
class InferArtifactOptions:
    include_anomaly_map_values: bool = False
    maps_dir: str | None = None
    overlays_dir: str | None = None
    defects_enabled: bool = False
    ...

@dataclass(frozen=True)
class InferResultArtifactAssemblyRequest:
    index: int
    input_path: str
    result: Any
    include_status: bool = False
    options: InferArtifactOptions = ...
```

**Step 2: Add builder function**

```python
def build_infer_result_artifact_request_from_options(
    request: InferResultArtifactAssemblyRequest,
) -> InferResultArtifactBuildRequest:
    ...
```

The builder should internally assemble `DefectsArtifactConfigBuildRequest` so callers do not need to know that dataclass shape.

**Step 3: Refactor CLI**

Replace the long nested `DefectsArtifactConfigBuildRequest(...)` block in `pyimgano/infer_cli.py` with:

```python
artifact_request = infer_artifact_service.build_infer_result_artifact_request_from_options(...)
```

**Step 4: Run test to verify it passes**

Run: `pytest --no-cov tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py -q`
Expected: PASS

### Task 3: Run focused infer regression coverage

**Files:**
- Test: `tests/test_infer_cli_production_guardrails_v1.py`
- Test: `tests/test_infer_cli_infer_config.py`
- Test: `tests/test_integration_workbench_train_then_infer.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_infer_artifact_service.py tests/test_infer_output_service.py tests/test_infer_continue_service.py tests/test_infer_runtime_service.py tests/test_infer_setup_service.py tests/test_infer_context_service.py tests/test_infer_cli_smoke.py tests/test_infer_cli_defects_regions_jsonl.py tests/test_infer_cli_maps_vs_defects_flags.py tests/test_infer_cli_infer_config.py tests/test_infer_cli_from_run.py tests/test_infer_cli_from_run_errors.py tests/test_infer_cli_production_guardrails_v1.py tests/test_infer_cli_onnx_session_options_v1.py tests/test_integration_workbench_train_then_infer.py -q`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- pyimgano/infer_cli.py pyimgano/services/__init__.py pyimgano/services/infer_artifact_service.py tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py docs/plans/2026-03-11-infer-cli-artifact-options-thinning.md`
Expected: no output
