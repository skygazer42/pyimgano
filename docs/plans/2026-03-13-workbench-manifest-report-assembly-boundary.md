# Workbench Manifest Report Assembly Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_preflight` orchestration noise by moving final manifest preflight payload assembly into a dedicated helper boundary.

**Architecture:** Keep `pyimgano.workbench.manifest_preflight.run_manifest_preflight(...)` as the orchestrator that validates sources, loads records, selects categories, and computes per-category summaries. Move the final output assembly for single-category flattening versus all-category nested payloads into `pyimgano.workbench.manifest_preflight_report`, so the entrypoint no longer mixes dispatching with response-shape branching.

**Tech Stack:** Python, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Report Assembly Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper can build the single-category flattened payload
- a dedicated helper can build the all-category nested payload
- `manifest_preflight.py` imports `pyimgano.workbench.manifest_preflight_report`
- `manifest_preflight.py` calls `build_manifest_preflight_report(...)`
- `manifest_preflight.py` no longer hosts the final `out` payload assembly inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "manifest_preflight_report or manifest_preflight_uses_report_assembly_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `manifest_preflight.py` still assembles the payload inline.

### Task 2: Extract The Manifest Report Assembly Helper

**Files:**
- Create: `pyimgano/workbench/manifest_preflight_report.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`

**Step 1: Write minimal implementation**

- move final payload assembly into `build_manifest_preflight_report(...)`
- preserve current single-category flattening and all-category `per_category` behavior
- keep all existing field names and payload values unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/manifest_preflight_report.py`
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
git diff --check -- docs/plans/2026-03-13-workbench-manifest-report-assembly-boundary.md pyimgano/workbench/manifest_preflight.py pyimgano/workbench/manifest_preflight_report.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
