# Benchmark Authority Batch 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn suite publication readiness from passive metadata into an actively gateable helper and CLI workflow.

**Architecture:** Build on the existing `leaderboard_metadata.json` contract emitted by `suite_export.py`. Add a small reporting helper for publication quality and expose it through `pyimgano-runs` so CI and release workflows can fail fast without re-implementing policy in shell scripts.

**Tech Stack:** Python 3.9+, JSON, argparse, existing `pyimgano.reporting`, `pyimgano.runs_cli`, pytest.

---

## Stream A: Publication Quality Helper

### Task 1: Add failing tests for publication quality evaluation

**Files:**
- Create: `tests/test_publication_quality.py`

**Verify:**
- `pytest --no-cov tests/test_publication_quality.py -q`

### Task 2: Add helper to evaluate leaderboard publication readiness

**Files:**
- Create: `pyimgano/reporting/publication_quality.py`
- Modify: `tests/test_publication_quality.py`

**Verify:**
- `pytest --no-cov tests/test_publication_quality.py -q`

## Stream B: CLI Audit Surface

### Task 3: Add failing tests for `pyimgano-runs publication`

**Files:**
- Modify: `tests/test_runs_cli.py`

**Verify:**
- `pytest --no-cov tests/test_runs_cli.py -q`

### Task 4: Implement `pyimgano-runs publication`

**Files:**
- Modify: `pyimgano/runs_cli.py`
- Modify: `tests/test_runs_cli.py`

**Verify:**
- `pytest --no-cov tests/test_runs_cli.py tests/test_publication_quality.py -q`

### Task 5: Document publication audit workflow

**Files:**
- Modify: `docs/BENCHMARK_PUBLICATION.md`
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `README.md`

**Verify:**
- `python -m py_compile pyimgano/reporting/publication_quality.py pyimgano/runs_cli.py`

## Final Verification

### Task 6: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_publication_quality.py tests/test_runs_cli.py tests/test_suite_export_metadata.py tests/test_run_quality.py -q`
- `python -m py_compile pyimgano/reporting/publication_quality.py pyimgano/runs_cli.py pyimgano/reporting/suite_export.py`
- `git diff --check`
