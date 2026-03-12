# Infer CLI Artifact Config Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move `DefectsArtifactConfig` assembly out of `infer_cli` and into the artifact service layer.

**Architecture:** Extend `infer_artifact_service` with an explicit builder request for defects artifact config. Keep `infer_cli` responsible for runtime values and outer sequencing, but make artifact-domain field mapping live with the artifact-domain dataclasses that consume it.

**Tech Stack:** Python 3.10, pytest, dataclasses

---

### Task 1: Add failing tests for the builder seam

**Files:**
- Modify: `tests/test_infer_artifact_service.py`
- Modify: `tests/test_infer_cli_smoke.py`

**Step 1: Write the failing tests**

```python
def test_build_defects_artifact_config_populates_fields():
    ...

def test_build_defects_artifact_config_requires_resolved_pixel_threshold():
    ...

def test_infer_cli_smoke_delegates_defects_config_building_to_service():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py -q`
Expected: FAIL because the new builder request/function do not exist yet.

### Task 2: Implement builder and refactor CLI

**Files:**
- Modify: `pyimgano/services/infer_artifact_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`

**Step 1: Write minimal implementation**

```python
@dataclass(frozen=True)
class DefectsArtifactConfigBuildRequest: ...

def build_defects_artifact_config(...): ...
```

**Step 2: Refactor CLI to delegate**

```python
defects_config = infer_artifact_service.build_defects_artifact_config(...)
```

**Step 3: Run tests to verify they pass**

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

Run: `git diff --check -- pyimgano/infer_cli.py pyimgano/services/__init__.py pyimgano/services/infer_artifact_service.py tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py docs/plans/2026-03-11-infer-cli-artifact-config-thinning.md`
Expected: no output
