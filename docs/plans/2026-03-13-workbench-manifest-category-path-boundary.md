# Workbench Manifest Category Path Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_category_preflight` coupling by moving manifest image/mask path validation and duplicate image detection into a dedicated helper boundary.

**Architecture:** Keep `pyimgano.workbench.manifest_category_preflight.preflight_manifest_category(...)` as the category-level orchestrator for split counts, group assignment, and pixel-metric assembly. Move manifest path resolution, missing image/mask issue emission, and duplicate image detection into `pyimgano.workbench.manifest_category_paths`, so category analysis no longer depends directly on `resolve_manifest_path_best_effort(...)`.

**Tech Stack:** Python, pytest, manifest dataset utilities, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Category Path Boundary With Failing Tests

**Files:**
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add boundary tests that prove:
- `manifest_category_preflight.py` imports a dedicated `pyimgano.workbench.manifest_category_paths` helper module
- `manifest_category_preflight.py` calls `inspect_manifest_category_paths(...)`
- `manifest_category_preflight.py` no longer imports or calls `resolve_manifest_path_best_effort(...)` directly
- `manifest_category_paths.py` hosts the path-resolution and duplicate-detection issue logic

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_architecture_boundaries.py -k "manifest_category_preflight_uses_path_helper_boundary or manifest_category_paths_module_hosts_path_resolution_checks" -v
```

Expected: FAIL because the dedicated helper module does not exist yet and `manifest_category_preflight.py` still performs path checks inline.

### Task 2: Extract The Manifest Category Path Helper

**Files:**
- Create: `pyimgano/workbench/manifest_category_paths.py`
- Modify: `pyimgano/workbench/manifest_category_preflight.py`

**Step 1: Write minimal implementation**

- move missing image/mask checks and duplicate image detection into `inspect_manifest_category_paths(...)`
- keep the same issue codes, severities, messages, and context payloads
- keep `preflight_manifest_category(...)` behavior-compatible for summary assembly

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_preflight_manifest.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_category_preflight.py`
- Modify: `pyimgano/workbench/manifest_category_paths.py`
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
git diff --check -- docs/plans/2026-03-13-workbench-manifest-category-path-boundary.md pyimgano/workbench/manifest_category_preflight.py pyimgano/workbench/manifest_category_paths.py tests/test_architecture_boundaries.py
```

Expected: no output
