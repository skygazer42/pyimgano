# Workbench Manifest Source Validation Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_preflight` coupling by moving manifest input-mode/path/root validation into a dedicated source-validation helper.

**Architecture:** Keep `pyimgano.workbench.manifest_preflight.run_manifest_preflight(...)` as the top-level manifest preflight orchestrator that derives split policy, loads records, selects categories, and assembles the final summary payload. Move input-mode validation, manifest path existence/readability checks, and root-fallback warning logic into `pyimgano.workbench.manifest_source_validation`, returning a structured source description plus an optional early summary for invalid manifest sources.

**Tech Stack:** Python, pytest, pathlib, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Source Validation Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper can resolve a valid manifest source and warn when the root fallback is missing
- a dedicated helper returns the expected early failure summary when `dataset.manifest_path` is missing
- `manifest_preflight.py` imports `pyimgano.workbench.manifest_source_validation`
- `manifest_preflight.py` calls `resolve_manifest_preflight_source(...)`
- `manifest_preflight.py` no longer hosts the manifest path/root/input-mode issue codes inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "manifest_source_validation or manifest_preflight_uses_source_validation_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `manifest_preflight.py` still performs source validation inline.

### Task 2: Extract The Manifest Source Validation Helper

**Files:**
- Create: `pyimgano/workbench/manifest_source_validation.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`

**Step 1: Write minimal implementation**

- move input-mode validation, manifest-path validation, and root-fallback warning logic into `resolve_manifest_preflight_source(...)`
- return the resolved manifest path, root fallback, and optional early summary payload
- keep `run_manifest_preflight(...)` result shape and issue codes/messages unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/manifest_source_validation.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-manifest-source-validation-boundary.md pyimgano/workbench/manifest_preflight.py pyimgano/workbench/manifest_source_validation.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
