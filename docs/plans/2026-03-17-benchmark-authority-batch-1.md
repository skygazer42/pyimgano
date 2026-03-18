# Benchmark Authority Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strengthen `pyimgano` benchmark publication trust by making suite exports self-auditing and making run comparison able to gate on environment compatibility.

**Architecture:** Build on the existing `suite_export`, `run_index`, `runs_cli`, and `run_quality` contracts rather than introducing a new publication framework. Keep additions JSON-friendly, offline-safe, and directly reusable by CI and release automation.

**Tech Stack:** Python 3.9+, JSON, argparse, existing `pyimgano.reporting`, `pyimgano.runs_cli`, pytest.

---

## Stream A: Suite Publication Metadata

### Task 1: Add failing tests for publication metadata enrichment

**Files:**
- Modify: `tests/test_suite_export_metadata.py`

**Steps:**
1. Add a test asserting `leaderboard_metadata.json` includes:
   - `artifact_quality`
   - `exported_files`
   - `publication_ready`
2. Run:
   - `pytest --no-cov tests/test_suite_export_metadata.py -q`
3. Expected: FAIL because the metadata payload does not yet include publication-readiness fields.

### Task 2: Add publication metadata helper to suite export

**Files:**
- Modify: `pyimgano/reporting/suite_export.py`
- Modify: `tests/test_suite_export_metadata.py`

**Steps:**
1. Implement a small helper that summarizes whether a suite export is publication-ready.
2. Include `artifact_quality` and `publication_ready` in `leaderboard_metadata.json`.
3. Include the exported artifact path map so the metadata is self-describing.
4. Run:
   - `pytest --no-cov tests/test_suite_export_metadata.py -q`
5. Expected: PASS.

### Task 3: Document suite publication metadata contract

**Files:**
- Modify: `docs/BENCHMARK_PUBLICATION.md`
- Modify: `docs/CLI_REFERENCE.md`

**Steps:**
1. Document the new publication metadata fields and their intended use in CI / release review.
2. Run:
   - `python -m py_compile pyimgano/reporting/suite_export.py`

## Stream B: Environment Comparability Gating

### Task 4: Add failing tests for environment comparison summary

**Files:**
- Modify: `tests/test_run_index.py`

**Steps:**
1. Add a test asserting `compare_run_summaries(...)` includes `environment_comparison` when a baseline is provided.
2. Cover matched / mismatched / missing environment fingerprints.
3. Run:
   - `pytest --no-cov tests/test_run_index.py -q`
4. Expected: FAIL because compare payload does not yet include environment compatibility metadata.

### Task 5: Add failing tests for CLI environment gate

**Files:**
- Modify: `tests/test_runs_cli.py`

**Steps:**
1. Add a test for `pyimgano-runs compare --baseline ... --require-same-environment --json`.
2. Run:
   - `pytest --no-cov tests/test_runs_cli.py -q`
3. Expected: FAIL because the CLI does not yet expose the gate.

### Task 6: Implement environment comparison summary and gate

**Files:**
- Modify: `pyimgano/reporting/run_index.py`
- Modify: `pyimgano/runs_cli.py`
- Modify: `tests/test_run_index.py`
- Modify: `tests/test_runs_cli.py`

**Steps:**
1. Add `environment_comparison` to compare payloads:
   - baseline fingerprint
   - per-run environment status
   - summary counts
2. Add `--require-same-environment` to `pyimgano-runs compare`.
3. Make JSON and text modes both enforce the gate.
4. Run:
   - `pytest --no-cov tests/test_run_index.py tests/test_runs_cli.py -q`
5. Expected: PASS.

### Task 7: Document environment gating workflow

**Files:**
- Modify: `docs/RUN_COMPARISON.md`
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `README.md`

**Steps:**
1. Add examples for `--require-same-environment`.
2. Explain when to use split / target / environment gates together.
3. Run:
   - `python -m py_compile pyimgano/reporting/run_index.py pyimgano/runs_cli.py`

## Final Verification

### Task 8: Run focused verification bundle

**Run:**
- `pytest --no-cov tests/test_suite_export_metadata.py tests/test_run_index.py tests/test_runs_cli.py tests/test_run_quality.py -q`
- `python -m py_compile pyimgano/reporting/suite_export.py pyimgano/reporting/run_index.py pyimgano/runs_cli.py`
- `git diff --check`

Expected: PASS.
