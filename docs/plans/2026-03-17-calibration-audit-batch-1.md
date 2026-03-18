# Calibration Audit Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make threshold-audit artifacts stricter and more trustworthy by validating `calibration_card.json` content, surfacing that validation in run quality checks, and documenting the review workflow.

**Architecture:** Keep the current artifact shape (`artifacts/calibration_card.json`) but strengthen the schema helper instead of inventing a new file type. Reuse the existing `run_quality` surface so `pyimgano-runs quality` can report whether calibration audit artifacts are merely present or actually valid.

**Tech Stack:** Python 3.9+, JSON artifacts, existing `pyimgano.reporting`, existing run-quality/reporting helpers, pytest.

---

## Stream A: Calibration Card Schema

### Task 1: Add failing schema tests

**Files:**
- Modify: `tests/test_calibration_card.py`

**Verify:**
- `pytest --no-cov tests/test_calibration_card.py -q`

### Task 2: Harden calibration card builder/validator

**Files:**
- Modify: `pyimgano/reporting/calibration_card.py`

**Verify:**
- `pytest --no-cov tests/test_calibration_card.py -q`

## Stream B: Run Quality Auditability

### Task 3: Add failing run-quality tests for invalid calibration cards

**Files:**
- Modify: `tests/test_run_quality.py`

**Verify:**
- `pytest --no-cov tests/test_run_quality.py -q`

### Task 4: Surface calibration-card validation in run quality

**Files:**
- Modify: `pyimgano/reporting/run_quality.py`

**Verify:**
- `pytest --no-cov tests/test_run_quality.py tests/test_calibration_card.py -q`

## Stream C: Documentation

### Task 5: Add calibration audit guide and links

**Files:**
- Create: `docs/CALIBRATION_AUDIT.md`
- Modify: `docs/WORKBENCH.md`
- Modify: `docs/INDUSTRIAL_FASTPATH.md`
- Modify: `docs/CLI_REFERENCE.md`

**Verify:**
- `python -c "from pathlib import Path; Path('docs/CALIBRATION_AUDIT.md').read_text(encoding='utf-8')"`

## Final Verification

### Task 6: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_calibration_card.py tests/test_run_quality.py tests/test_runs_cli.py tests/test_workbench_export_infer_config.py -q`
- `python -m py_compile pyimgano/reporting/calibration_card.py pyimgano/reporting/run_quality.py`
- `git diff --check`
