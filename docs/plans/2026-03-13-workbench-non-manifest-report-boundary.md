# Workbench Non-Manifest Report Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the success-summary payload assembly out of `pyimgano.workbench.non_manifest_preflight` into a dedicated report boundary so non-manifest preflight matches the manifest preflight layering style.

**Architecture:** Keep `pyimgano.workbench.non_manifest_preflight.run_non_manifest_preflight(...)` as the orchestrator that validates the source, loads categories, validates the requested category, and delegates final success-summary creation. Add `pyimgano.workbench.non_manifest_preflight_report.build_non_manifest_preflight_report(...)` to build the JSON-friendly payload, so the orchestrator no longer owns inline report assembly.

**Tech Stack:** Python, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Non-Manifest Report Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a non-manifest report helper builds the expected success payload from `root` and `categories`
- `non_manifest_preflight.py` imports and uses `build_non_manifest_preflight_report(...)`
- `non_manifest_preflight.py` no longer hosts the inline `{"dataset_root": str(root), "categories": categories, "ok": True}` payload

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "non_manifest_preflight_report or non_manifest_preflight_uses_report_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `non_manifest_preflight.py` still assembles the payload inline.

### Task 2: Add The Report Helper And Delegate To It

**Files:**
- Create: `pyimgano/workbench/non_manifest_preflight_report.py`
- Modify: `pyimgano/workbench/non_manifest_preflight.py`

**Step 1: Write minimal implementation**

- add `build_non_manifest_preflight_report(root: str, categories: list[str]) -> dict[str, Any]`
- move the success payload assembly into that helper
- keep the final summary shape unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_non_manifest_preflight.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Create: `pyimgano/workbench/non_manifest_preflight_report.py`
- Modify: `pyimgano/workbench/non_manifest_preflight.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_non_manifest_preflight.py tests/test_workbench_preflight_non_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-non-manifest-report-boundary.md pyimgano/workbench/non_manifest_preflight.py pyimgano/workbench/non_manifest_preflight_report.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
