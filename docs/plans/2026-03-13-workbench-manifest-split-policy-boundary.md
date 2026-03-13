# Workbench Manifest Split Policy Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce duplication and tighten manifest-related boundaries by moving manifest split-policy construction into a shared helper used by both manifest preflight and manifest dataset loading.

**Architecture:** Keep `pyimgano.workbench.manifest_preflight.run_manifest_preflight(...)` as the orchestrator that validates the source, builds a manifest split policy, loads records, selects categories, and assembles the final summary. Keep `pyimgano.workbench.dataset_loader.load_workbench_split(...)` as the dataset split loader. Move `ManifestSplitPolicy(...)` construction and seed fallback logic into `pyimgano.workbench.manifest_split_policy`, and have both callers import `build_manifest_split_policy(...)`.

**Tech Stack:** Python, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Split Policy Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_workbench_manifest_preflight.py`
- Modify: `tests/test_workbench_dataset_loader.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper builds a `ManifestSplitPolicy` from config, preferring `dataset.split_policy.seed`
- the same helper falls back to `config.seed` and then `0` when the split-policy seed is absent
- `manifest_preflight.py` imports and uses `build_manifest_split_policy(...)`
- `dataset_loader.py` imports and uses `build_manifest_split_policy(...)`
- neither caller hosts inline `ManifestSplitPolicy(...)` construction or the old `_manifest_split_policy_from_config(...)` helper

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_workbench_dataset_loader.py tests/test_architecture_boundaries.py -k "manifest_split_policy or split_policy_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and the callers still construct manifest split policies inline.

### Task 2: Extract The Shared Manifest Split Policy Helper

**Files:**
- Create: `pyimgano/workbench/manifest_split_policy.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/dataset_loader.py`

**Step 1: Write minimal implementation**

- move `ManifestSplitPolicy(...)` construction and seed fallback logic into `build_manifest_split_policy(...)`
- import and use the helper from both `manifest_preflight.py` and `dataset_loader.py`
- keep public behavior, output shapes, and manifest split semantics unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_workbench_dataset_loader.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/dataset_loader.py`
- Modify: `pyimgano/workbench/manifest_split_policy.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_workbench_manifest_preflight.py`
- Modify: `tests/test_workbench_dataset_loader.py`
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
git diff --check -- docs/plans/2026-03-13-workbench-manifest-split-policy-boundary.md pyimgano/workbench/manifest_preflight.py pyimgano/workbench/dataset_loader.py pyimgano/workbench/manifest_split_policy.py tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_workbench_dataset_loader.py tests/test_architecture_boundaries.py
```

Expected: no output
