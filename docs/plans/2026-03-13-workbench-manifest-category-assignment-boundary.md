# Workbench Manifest Category Assignment Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_category_preflight` complexity by moving group conflict checks, split assignment, mask coverage, and pixel-metric decisions into a dedicated helper boundary.

**Architecture:** Keep `pyimgano.workbench.manifest_category_preflight.preflight_manifest_category(...)` as the category-level orchestrator that assembles explicit split counts and delegates to narrow helpers. Move group indexing, conflict issue emission, train/val/test assignment, mask coverage aggregation, and pixel-metric enablement logic into `pyimgano.workbench.manifest_category_assignment`, passing the precomputed `mask_exists_by_index` map and issue builder callback so the helper stays decoupled from path resolution and issue types.

**Tech Stack:** Python, pytest, manifest dataset utilities, NumPy RNG split selection, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Category Assignment Boundary With Failing Tests

**Files:**
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add boundary tests that prove:
- `manifest_category_preflight.py` imports a dedicated `pyimgano.workbench.manifest_category_assignment` helper module
- `manifest_category_preflight.py` calls `analyze_manifest_category_assignment(...)`
- `manifest_category_preflight.py` no longer hosts the group conflict issue strings or assignment internals inline
- `manifest_category_assignment.py` hosts the group conflict, split assignment, mask coverage, and pixel metric logic

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_architecture_boundaries.py -k "manifest_category_preflight_uses_assignment_helper_boundary or manifest_category_assignment_module_hosts_group_assignment_logic" -v
```

Expected: FAIL because the dedicated helper module does not exist yet and `manifest_category_preflight.py` still defines the assignment logic inline.

### Task 2: Extract The Manifest Category Assignment Helper

**Files:**
- Create: `pyimgano/workbench/manifest_category_assignment.py`
- Modify: `pyimgano/workbench/manifest_category_preflight.py`

**Step 1: Write minimal implementation**

- move group indexing, conflict checks, split assignment, mask coverage, and pixel metric decisions into `analyze_manifest_category_assignment(...)`
- keep the same issue codes, severities, messages, and summary payload shape
- keep `preflight_manifest_category(...)` behavior-compatible for callers

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_category_preflight.py`
- Modify: `pyimgano/workbench/manifest_category_assignment.py`
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
git diff --check -- docs/plans/2026-03-13-workbench-manifest-category-assignment-boundary.md pyimgano/workbench/manifest_category_preflight.py pyimgano/workbench/manifest_category_assignment.py tests/test_architecture_boundaries.py
```

Expected: no output
