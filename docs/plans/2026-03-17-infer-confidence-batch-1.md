# Inference Confidence Export Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Surface detector label-confidence semantics in `pyimgano-infer` outputs without changing the default inference path.

**Architecture:** Extend `InferenceResult` and the inference service with an opt-in confidence path that reuses detector-native confidence helpers when present, and otherwise stays silent. Wire the behavior into `pyimgano-infer` through an explicit CLI flag so JSONL output gains additive metadata without surprising existing consumers.

**Tech Stack:** Python 3.9+, argparse, NumPy, existing inference service/artifact service, pytest.

---

## Stream A: Confidence-Aware Inference Output

### Task 1: Add failing inference API and CLI tests

**Files:**
- Modify: `tests/test_inference_api.py`
- Modify: `tests/test_infer_cli_smoke.py`

**Verify:**
- `pytest --no-cov tests/test_inference_api.py tests/test_infer_cli_smoke.py -q`

### Task 2: Implement opt-in confidence collection in inference service

**Files:**
- Modify: `pyimgano/inference/api.py`
- Modify: `pyimgano/services/inference_service.py`

**Verify:**
- `pytest --no-cov tests/test_inference_api.py -q`

### Task 3: Wire CLI flag and JSONL serialization

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/inference/api.py`
- Modify: `pyimgano/services/infer_artifact_service.py`

**Verify:**
- `pytest --no-cov tests/test_infer_cli_smoke.py -q`

### Task 4: Document confidence emission semantics

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `docs/INDUSTRIAL_INFERENCE.md`

**Verify:**
- `python -m py_compile pyimgano/infer_cli.py pyimgano/inference/api.py pyimgano/services/inference_service.py`

## Final Verification

### Task 5: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_inference_api.py tests/test_infer_cli_smoke.py tests/test_workbench_export_infer_config.py -q`
- `python -m py_compile pyimgano/infer_cli.py pyimgano/inference/api.py pyimgano/services/inference_service.py`
- `git diff --check`
