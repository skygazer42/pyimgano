# Workbench Preflight Summary Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dataset-type dispatch from `pyimgano.workbench.preflight` by moving manifest vs non-manifest summary routing into a dedicated helper boundary.

**Architecture:** Keep `pyimgano.workbench.preflight.run_preflight(...)` as the public entrypoint that creates the issue list, runs model compatibility checks, and returns the final `PreflightReport`. Add `pyimgano.workbench.preflight_summary.resolve_workbench_preflight_summary(...)` to own dataset-name routing between `run_manifest_preflight(...)` and `run_non_manifest_preflight(...)`, so the entrypoint no longer hosts manifest/non-manifest branching inline.

**Tech Stack:** Python, pytest, dataclasses, string-based architecture boundary tests.

---

### Task 1: Lock The Preflight Summary Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a new preflight summary helper dispatches manifest configs to `run_manifest_preflight(...)`
- the same helper dispatches non-manifest configs to `run_non_manifest_preflight(...)`
- `preflight.py` imports and uses `resolve_workbench_preflight_summary(...)`
- `preflight.py` no longer hosts inline `if ds == "manifest"` dataset routing

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py -k "preflight_summary or preflight_uses_summary_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `preflight.py` still routes inline.

### Task 2: Add The Summary Helper And Delegate To It

**Files:**
- Create: `pyimgano/workbench/preflight_summary.py`
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Write minimal implementation**

- add `resolve_workbench_preflight_summary(config, issues, issue_builder) -> dict[str, Any]`
- move manifest vs non-manifest dataset dispatch into that helper
- keep `run_preflight(...)`, `PreflightIssue`, and `PreflightReport` behavior-compatible

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Create: `pyimgano/workbench/preflight_summary.py`
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-preflight-summary-boundary.md pyimgano/workbench/preflight.py pyimgano/workbench/preflight_summary.py tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py
```

Expected: no output
