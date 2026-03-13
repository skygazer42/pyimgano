# Workbench Preflight Issue Factory Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move `PreflightIssue` construction out of `pyimgano.workbench.preflight` into a dedicated helper boundary while preserving existing issue payload behavior.

**Architecture:** Keep `pyimgano.workbench.preflight` as the public orchestrator that runs model compatibility checks, resolves dataset-specific summary output, and returns the final `PreflightReport`. Add `pyimgano.workbench.preflight_issue_factory.build_preflight_issue(...)` to own `PreflightIssue` construction and string coercion, so the orchestrator no longer defines `_issue(...)` inline.

**Tech Stack:** Python, dataclasses, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Preflight Issue Factory Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a new `preflight_issue_factory` module builds `PreflightIssue` instances with the expected string coercion behavior
- `preflight.py` imports and uses `build_preflight_issue(...)`
- `preflight.py` no longer defines `_issue(...)` inline
- `preflight_issue_factory.py` hosts the `PreflightIssue(...)` construction logic

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py -k "preflight_issue_factory or preflight_uses_issue_factory_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `preflight.py` still defines `_issue(...)`.

### Task 2: Add The Issue Factory Helper And Delegate To It

**Files:**
- Create: `pyimgano/workbench/preflight_issue_factory.py`
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Write minimal implementation**

- add `build_preflight_issue(...) -> PreflightIssue`
- move the existing string coercion logic there
- keep `run_preflight(...)` and all current issue codes / payloads behavior-compatible

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Create: `pyimgano/workbench/preflight_issue_factory.py`
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_workbench_preflight_preprocessing.py tests/test_workbench_package.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-preflight-issue-factory-boundary.md pyimgano/workbench/preflight.py pyimgano/workbench/preflight_issue_factory.py tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py
```

Expected: no output
