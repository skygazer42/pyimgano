# Workbench Preflight Model Compatibility Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.preflight` coupling by extracting model/config compatibility checks into a dedicated helper boundary while preserving the current preflight report shape and issue codes.

**Architecture:** Keep `pyimgano.workbench.preflight.run_preflight(...)` as the public orchestrator that assembles the report and routes between manifest and non-manifest dataset preflight flows. Move the model-registry probing and pixel-map / preprocessing compatibility checks into `pyimgano.workbench.preflight_model_compat`, passing an issue builder callback so the helper stays decoupled from `PreflightIssue` construction details.

**Tech Stack:** Python, pytest, workbench config dataclasses, model compatibility helpers, string-based architecture boundary tests.

---

### Task 1: Lock The Preflight Model Compatibility Boundary With Failing Tests

**Files:**
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add a boundary test that proves:
- `preflight.py` imports a dedicated `pyimgano.workbench.preflight_model_compat` helper module
- `preflight.py` calls `run_workbench_model_compat_preflight(...)`
- `preflight.py` no longer defines `_preflight_model_compat(...)` inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_architecture_boundaries.py -k "preflight_uses_model_compat_preflight_boundary" -v
```

Expected: FAIL because the dedicated helper module does not exist yet and `preflight.py` still defines `_preflight_model_compat(...)`.

### Task 2: Extract The Preflight Model Compatibility Helper

**Files:**
- Create: `pyimgano/workbench/preflight_model_compat.py`
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Write minimal implementation**

- move the current model/config compatibility validation logic into `run_workbench_model_compat_preflight(...)`
- keep issue creation callback-driven so the helper does not depend on `PreflightIssue`
- keep `run_preflight(...)`, `PreflightIssue`, `PreflightReport`, issue codes, and error messages behavior-compatible

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_preflight_preprocessing.py tests/test_workbench_preflight_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `pyimgano/workbench/preflight_model_compat.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_preflight_preprocessing.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-preflight-model-compat-boundary.md pyimgano/workbench/preflight.py pyimgano/workbench/preflight_model_compat.py tests/test_architecture_boundaries.py
```

Expected: no output
