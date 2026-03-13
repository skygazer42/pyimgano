# Workbench Manifest Report Policy Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the last inline manifest report payload mapping from `pyimgano.workbench.manifest_preflight` by moving split-policy serialization into the report assembly boundary.

**Architecture:** Keep `pyimgano.workbench.manifest_preflight.run_manifest_preflight(...)` as the orchestrator that validates inputs, resolves records, selects categories, runs per-category analysis, and delegates final report creation. Extend `pyimgano.workbench.manifest_preflight_report.build_manifest_preflight_report(...)` to accept the manifest split-policy object directly and serialize its fields into the report payload, so the orchestrator no longer builds the `split_policy` dict inline.

**Tech Stack:** Python, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Report-Policy Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- the report helper serializes a policy object into the expected `split_policy` payload
- the helper still flattens single-category output and preserves `per_category` output for `all`
- `manifest_preflight.py` imports and uses `build_manifest_preflight_report(...)`
- `manifest_preflight.py` no longer hosts the inline `split_policy={...}` mapping with `policy.mode`, `policy.scope`, `policy.seed`, and `policy.test_normal_fraction`

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "manifest_preflight_report or manifest_preflight_uses_report_assembly_boundary" -v
```

Expected: FAIL because the report helper still expects a pre-built `split_policy` dict and `manifest_preflight.py` still serializes the policy inline.

### Task 2: Extend The Manifest Report Boundary

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight_report.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`

**Step 1: Write minimal implementation**

- change `build_manifest_preflight_report(...)` to accept the policy object
- serialize the policy fields inside the report helper
- keep the final report shape unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight_report.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_workbench_dataset_loader.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-manifest-report-policy-boundary.md pyimgano/workbench/manifest_preflight_report.py pyimgano/workbench/manifest_preflight.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
