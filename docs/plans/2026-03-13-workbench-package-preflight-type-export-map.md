# Workbench Package Preflight Type Export Map Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the lazy `pyimgano.workbench` facade resolve `PreflightIssue` and `PreflightReport` directly from `preflight_types` instead of routing those type exports through the `preflight` runtime facade.

**Architecture:** Keep `pyimgano.workbench.preflight` as the runtime entrypoint that exposes `run_preflight(...)` and compatibility imports for preflight types. Update `pyimgano.workbench.__init__` so the package-level export map points `PreflightIssue` and `PreflightReport` at `pyimgano.workbench.preflight_types`, reducing incidental module loading when callers only need the dataclasses.

**Tech Stack:** Python, pytest, lazy import facade tests.

---

### Task 1: Lock The Package Export Source Alignment With Failing Tests

**Files:**
- Modify: `tests/test_workbench_package.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `_WORKBENCH_EXPORT_GROUPS` maps `PreflightIssue` and `PreflightReport` to `pyimgano.workbench.preflight_types`
- resolving `workbench.PreflightReport` loads `pyimgano.workbench.preflight_types` but does not load `pyimgano.workbench.preflight`

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_package.py tests/test_architecture_boundaries.py -k "preflight_type_export or grouped_export_spec" -v
```

Expected: FAIL because the package still maps those exports through `pyimgano.workbench.preflight`.

### Task 2: Update The Lazy Export Map

**Files:**
- Modify: `pyimgano/workbench/__init__.py`

**Step 1: Write minimal implementation**

- repoint `PreflightIssue` and `PreflightReport` exports to `pyimgano.workbench.preflight_types`
- keep `run_preflight` sourced from `pyimgano.workbench.preflight`
- preserve all existing package-level export names

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_package.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/__init__.py`
- Modify: `tests/test_workbench_package.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_package.py tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-package-preflight-type-export-map.md pyimgano/workbench/__init__.py tests/test_workbench_package.py tests/test_architecture_boundaries.py
```

Expected: no output
