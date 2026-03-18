# Model Card Asset Validation Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make model cards useful for real deploy handoff by validating the referenced weight asset path and optional SHA256 digest, not just the JSON schema.

**Architecture:** Extend the existing lightweight `pyimgano.weights.model_card` validator instead of inventing a second artifact type. Keep schema validation best-effort, add optional asset checks behind explicit flags, and expose the result cleanly through `pyimgano-weights validate-model-card`.

**Tech Stack:** Python 3.9+, JSON, pathlib, existing file hash helper, argparse, pytest.

---

## Stream A: Validator

### Task 1: Add failing validator tests

**Files:**
- Modify: `tests/test_model_card_validation.py`

**Verify:**
- `pytest --no-cov tests/test_model_card_validation.py -q`

### Task 2: Implement model-card asset and hash checks

**Files:**
- Modify: `pyimgano/weights/model_card.py`

**Verify:**
- `pytest --no-cov tests/test_model_card_validation.py -q`

## Stream B: CLI

### Task 3: Add failing CLI tests

**Files:**
- Modify: `tests/test_weights_cli.py`

**Verify:**
- `pytest --no-cov tests/test_weights_cli.py -q`

### Task 4: Wire CLI flags for asset checking

**Files:**
- Modify: `pyimgano/weights_cli.py`

**Verify:**
- `pytest --no-cov tests/test_weights_cli.py tests/test_model_card_validation.py -q`

## Stream C: Docs

### Task 5: Document asset-aware model card validation

**Files:**
- Modify: `docs/MODEL_CARDS.md`
- Modify: `docs/WEIGHTS.md`
- Modify: `docs/CLI_REFERENCE.md`

**Verify:**
- `python -m py_compile pyimgano/weights/model_card.py pyimgano/weights_cli.py`

## Final Verification

### Task 6: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_model_card_validation.py tests/test_weights_cli.py tests/test_weights_manifest_v1.py -q`
- `python -m py_compile pyimgano/weights/model_card.py pyimgano/weights_cli.py`
- `git diff --check`
