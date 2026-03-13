# Workbench Preflight Types Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move `PreflightIssue`, `PreflightReport`, and `IssueSeverity` out of `pyimgano.workbench.preflight` into a dedicated types boundary while preserving existing imports and runtime behavior.

**Architecture:** Keep `pyimgano.workbench.preflight` as the public compatibility facade that exposes `run_preflight(...)` plus the existing type names. Add `pyimgano.workbench.preflight_types` to host the dataclasses and literal alias, so the orchestrator no longer mixes type definitions with runtime flow control.

**Tech Stack:** Python, dataclasses, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Preflight Types Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_non_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a new `pyimgano.workbench.preflight_types` module exports `IssueSeverity`, `PreflightIssue`, and `PreflightReport`
- `preflight.py` imports those names from the new module
- `preflight.py` no longer defines `@dataclass`-decorated `PreflightIssue` and `PreflightReport` inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py -k "preflight_types or preflight_uses_types_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `preflight.py` still defines the dataclasses inline.

### Task 2: Add The Types Helper And Re-Export Through The Facade

**Files:**
- Create: `pyimgano/workbench/preflight_types.py`
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Write minimal implementation**

- move `IssueSeverity`, `PreflightIssue`, and `PreflightReport` to `preflight_types.py`
- import those names into `preflight.py`
- keep `run_preflight(...)` and existing imports behavior-compatible

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Create: `pyimgano/workbench/preflight_types.py`
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
git diff --check -- docs/plans/2026-03-13-workbench-preflight-types-boundary.md pyimgano/workbench/preflight.py pyimgano/workbench/preflight_types.py tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py
```

Expected: no output
