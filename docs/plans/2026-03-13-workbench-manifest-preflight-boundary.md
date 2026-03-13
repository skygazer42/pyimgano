# Workbench Manifest Preflight Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.preflight` complexity by moving manifest-specific validation into a dedicated module.

**Architecture:** Extract manifest-only preflight logic into `pyimgano.workbench.manifest_preflight`, keeping `preflight.py` as the high-level coordinator for model compatibility checks and dataset-type routing. Preserve the current JSON-friendly report payload and issue codes exactly.

**Tech Stack:** Python, dataclasses, pathlib, NumPy, pytest, existing manifest dataset helpers, `WorkbenchConfig`.

---

### Task 1: Lock The Manifest Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.workbench.manifest_preflight` exposes a callable manifest preflight helper
- the helper can validate a minimal manifest-backed config and returns the expected summary shape
- `pyimgano.workbench.preflight` delegates manifest handling through the new module instead of owning the manifest helpers inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py -k "manifest_preflight or preflight_uses_manifest_preflight_boundary" -v
```

Expected: FAIL because the new module does not exist and `preflight.py` still owns the manifest helpers directly.

### Task 2: Extract The Module And Delegate To It

**Files:**
- Create: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write minimal implementation**

- Move manifest-only preflight helpers into `manifest_preflight.py`
- Keep the helper generic by receiving an issue-builder callback from `preflight.py` rather than creating a circular dependency on `PreflightIssue`
- Update `preflight.py` to call the new module for manifest configs while preserving all existing issue codes, messages, and summary fields

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_train_cli_preflight.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `tests/test_workbench_manifest_preflight.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_train_cli_preflight.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-manifest-preflight-boundary.md pyimgano/workbench/manifest_preflight.py pyimgano/workbench/preflight.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py
```

Expected: no output
