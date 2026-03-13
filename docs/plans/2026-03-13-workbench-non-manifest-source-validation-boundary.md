# Workbench Non-Manifest Source Validation Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.non_manifest_preflight` coupling by moving root existence and custom dataset structure validation into a dedicated source-validation helper.

**Architecture:** Keep `pyimgano.workbench.non_manifest_preflight.run_non_manifest_preflight(...)` as the orchestrator that validates the non-manifest source, lists categories, checks requested categories, and assembles the final summary. Move dataset-root existence checks and custom dataset structure validation into `pyimgano.workbench.non_manifest_source_validation`, returning the resolved dataset/root info plus an optional early summary for missing roots.

**Tech Stack:** Python, pytest, pathlib, string-based architecture boundary tests.

---

### Task 1: Lock The Non-Manifest Source Validation Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper returns the expected early summary when the dataset root is missing
- a dedicated helper emits `CUSTOM_DATASET_INVALID_STRUCTURE` for an invalid custom dataset layout while allowing preflight to continue
- `non_manifest_preflight.py` imports `pyimgano.workbench.non_manifest_source_validation`
- `non_manifest_preflight.py` calls `resolve_non_manifest_preflight_source(...)`
- `non_manifest_preflight.py` no longer hosts the root-missing and custom-structure issue logic inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "non_manifest_source_validation or non_manifest_preflight_uses_source_validation_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `non_manifest_preflight.py` still performs source validation inline.

### Task 2: Extract The Non-Manifest Source Validation Helper

**Files:**
- Create: `pyimgano/workbench/non_manifest_source_validation.py`
- Modify: `pyimgano/workbench/non_manifest_preflight.py`

**Step 1: Write minimal implementation**

- move root existence checks and custom dataset structure validation into `resolve_non_manifest_preflight_source(...)`
- return the resolved dataset/root plus an optional early summary payload
- keep `run_non_manifest_preflight(...)` output shape and issue codes/messages unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/non_manifest_preflight.py`
- Modify: `pyimgano/workbench/non_manifest_source_validation.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_non_manifest.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-non-manifest-source-validation-boundary.md pyimgano/workbench/non_manifest_preflight.py pyimgano/workbench/non_manifest_source_validation.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
