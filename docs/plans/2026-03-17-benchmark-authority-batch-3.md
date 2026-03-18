# Benchmark Authority Batch 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repository-level audit that exercises the suite publication contract end-to-end and gates it in CI.

**Architecture:** Reuse `export_suite_tables(...)` and `evaluate_publication_quality(...)` with a synthetic official suite payload inside a small repo tool. Keep the audit deterministic, offline-safe, and lightweight so it can run in the `quality` CI job.

**Tech Stack:** Python 3.9+, tempfile, existing `pyimgano.reporting`, pytest, GitHub Actions.

---

## Stream A: Publication Contract Audit Tool

### Task 1: Add failing test for publication contract audit tool

**Files:**
- Create: `tests/test_tools_audit_publication_contract.py`

**Verify:**
- `pytest --no-cov tests/test_tools_audit_publication_contract.py -q`

### Task 2: Implement audit tool

**Files:**
- Create: `tools/audit_publication_contract.py`
- Modify: `tests/test_tools_audit_publication_contract.py`

**Verify:**
- `pytest --no-cov tests/test_tools_audit_publication_contract.py -q`

### Task 3: Wire audit tool into CI and docs

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `docs/BENCHMARK_PUBLICATION.md`

**Verify:**
- `python tools/audit_publication_contract.py`

## Final Verification

### Task 4: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_tools_audit_publication_contract.py tests/test_publication_quality.py tests/test_suite_export_metadata.py -q`
- `python -m py_compile tools/audit_publication_contract.py pyimgano/reporting/publication_quality.py pyimgano/reporting/suite_export.py`
- `git diff --check`
