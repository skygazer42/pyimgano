# Workbench Manifest Preflight Components Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.manifest_preflight` coupling by separating manifest record loading/path resolution from per-category preflight analysis while preserving the current report shape and issue messages.

**Architecture:** Keep `pyimgano.workbench.manifest_preflight.run_manifest_preflight(...)` as the top-level orchestrator. Move manifest JSONL parsing and path resolution into `pyimgano.workbench.manifest_record_preflight`, and move per-category split/mask/pixel-metric analysis into `pyimgano.workbench.manifest_category_preflight`, so callers keep the same public entrypoint but internal responsibilities become narrower and easier to test.

**Tech Stack:** Python, pytest, existing manifest dataset utilities, string-based architecture boundary tests.

---

### Task 1: Lock The Manifest Preflight Helper Boundaries With Failing Tests

**Files:**
- Create: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated module hosts manifest record loading and best-effort path resolution helpers
- a dedicated module hosts per-category manifest preflight analysis
- `manifest_preflight.py` acts as an orchestrator instead of defining the helper implementations inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py -k "manifest_preflight_components or manifest_preflight_uses_component_boundaries" -v
```

Expected: FAIL because the helper modules do not exist yet and `manifest_preflight.py` still hosts the helper implementations inline.

### Task 2: Extract Manifest Preflight Helper Modules

**Files:**
- Create: `pyimgano/workbench/manifest_record_preflight.py`
- Create: `pyimgano/workbench/manifest_category_preflight.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`

**Step 1: Write minimal implementation**

- move manifest JSON parsing / record loading / path resolution into `manifest_record_preflight.py`
- move per-category manifest summary building into `manifest_category_preflight.py`
- keep `run_manifest_preflight(...)` and `_manifest_split_policy_from_config(...)` behavior-compatible in `manifest_preflight.py`

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Modify: `pyimgano/workbench/manifest_record_preflight.py`
- Modify: `pyimgano/workbench/manifest_category_preflight.py`
- Modify: `tests/test_workbench_manifest_preflight_components.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_manifest_preflight_components.py tests/test_workbench_manifest_preflight.py tests/test_workbench_preflight_non_manifest.py tests/test_workbench_config.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-manifest-preflight-components.md pyimgano/workbench/manifest_preflight.py pyimgano/workbench/manifest_record_preflight.py pyimgano/workbench/manifest_category_preflight.py tests/test_workbench_manifest_preflight_components.py tests/test_architecture_boundaries.py
```

Expected: no output
