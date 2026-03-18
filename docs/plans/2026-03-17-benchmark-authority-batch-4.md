# Benchmark Authority Batch 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a canonical discovery surface for official benchmark configs in `pyimgano-benchmark`.

**Architecture:** Reuse `pyimgano.reporting.benchmark_config` as the single source of truth for parsing and describing benchmark configs, then extend it with directory-based official config discovery. Wire the new helpers into `pyimgano.cli` so text and JSON discovery modes match the existing suite and sweep UX.

**Tech Stack:** Python 3.9+, argparse, pathlib, hashlib, existing `pyimgano.reporting`, pytest.

---

## Stream A: Official Benchmark Config Discovery

### Task 1: Add failing discovery tests

**Files:**
- Modify: `tests/test_cli_baseline_suites_v16.py`

**Verify:**
- `pytest --no-cov tests/test_cli_baseline_suites_v16.py -q`

### Task 2: Implement official config discovery helpers

**Files:**
- Modify: `pyimgano/reporting/benchmark_config.py`
- Modify: `tests/test_benchmark_config.py`

**Verify:**
- `pytest --no-cov tests/test_benchmark_config.py tests/test_benchmark_configs_official.py -q`

### Task 3: Wire CLI discovery flags and rendering

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_baseline_suites_v16.py`

**Verify:**
- `pytest --no-cov tests/test_cli_baseline_suites_v16.py -q`

### Task 4: Document the new official config discovery surface

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `docs/BENCHMARK_PUBLICATION.md`
- Modify: `benchmarks/configs/README.md`

**Verify:**
- `python tools/audit_benchmark_configs.py`

## Final Verification

### Task 5: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_cli_baseline_suites_v16.py tests/test_benchmark_config.py tests/test_benchmark_configs_official.py -q`
- `python tools/audit_benchmark_configs.py`
- `python -m py_compile pyimgano/cli.py pyimgano/reporting/benchmark_config.py`
- `git diff --check`
