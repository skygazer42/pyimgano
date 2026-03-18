# Inference Rejection Export Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add explicit low-confidence rejection output to `pyimgano-infer` records.

**Architecture:** Build on the new confidence-export path instead of inventing a second prediction API. Add opt-in rejection controls in the inference layer, rewrite low-confidence labels to a configurable reject label, and emit an explicit `rejected` boolean in JSONL records for downstream systems.

**Tech Stack:** Python 3.9+, argparse, NumPy, existing `pyimgano.inference`, existing infer services, pytest.

---

## Stream A: Rejection-Aware Inference Output

### Task 1: Add failing API, CLI, and continue-on-error tests

**Files:**
- Modify: `tests/test_inference_api.py`
- Modify: `tests/test_infer_cli_smoke.py`
- Modify: `tests/test_infer_continue_service.py`

**Verify:**
- `pytest --no-cov tests/test_inference_api.py tests/test_infer_cli_smoke.py tests/test_infer_continue_service.py -q`

### Task 2: Implement rejection semantics in inference APIs/services

**Files:**
- Modify: `pyimgano/inference/api.py`
- Modify: `pyimgano/services/inference_service.py`
- Modify: `pyimgano/services/infer_continue_service.py`

**Verify:**
- `pytest --no-cov tests/test_inference_api.py tests/test_infer_continue_service.py -q`

### Task 3: Wire CLI flags and JSONL serialization

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/inference/api.py`

**Verify:**
- `pytest --no-cov tests/test_infer_cli_smoke.py -q`

### Task 4: Document rejection behavior

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `docs/INDUSTRIAL_INFERENCE.md`

**Verify:**
- `python -m py_compile pyimgano/infer_cli.py pyimgano/inference/api.py pyimgano/services/inference_service.py`

## Final Verification

### Task 5: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_inference_api.py tests/test_infer_cli_smoke.py tests/test_infer_continue_service.py tests/test_infer_cli_from_run.py tests/test_infer_cli_infer_config.py -q`
- `python -m py_compile pyimgano/infer_cli.py pyimgano/inference/api.py pyimgano/services/inference_service.py pyimgano/services/infer_continue_service.py`
- `git diff --check`
