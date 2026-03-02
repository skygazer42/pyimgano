# PyImgAno Next 100 Tasks (v14) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enforce stricter industrial constraints:
- Avoid sprinkling new `try/except` blocks into model code (prefer deterministic, spec-friendly inputs).
- Forbid TensorRT (`tensorrt` / `trt`) imports in `pyimgano/` (import-time safety).

**Why (industrial):**
- Broad `try/except` can mask real errors and creates inconsistent behavior across environments.
- TensorRT imports frequently crash in CI / airgapped environments due to missing shared libraries.

**Constraints:** no new deps, keep contracts stable, one final commit.

---

## Phase 0 — Tests / Guardrails

### Task 1001: Add regression test: reject estimator instances for spec-friendly Feature Bagging
**Files:**
- Create: `tests/test_feature_bagging_spec_instance_rejected_v14.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_feature_bagging_spec_instance_rejected_v14.py -v`

### Task 1002: Add audit: forbid TensorRT imports anywhere under `pyimgano/`
**Files:**
- Create: `tools/audit_no_tensorrt_imports.py`

**Verify:**
- `python tools/audit_no_tensorrt_imports.py`

---

## Phase 1 — Implementation

### Task 1011: Remove instance support from `base_estimator_spec` (no deepcopy / try/except)
**Design:**
- `base_estimator_spec` must be JSON-friendly: string or `{"name": "...", "kwargs": {...}}`.
- Passing an estimator instance is rejected with a clear `TypeError`.

**Files:**
- Modify: `pyimgano/models/feature_bagging.py`

---

## Phase 2 — Docs + Changelog

### Task 1021: Update changelog
**Files:**
- Modify: `CHANGELOG.md`

---

## Phase 3 — Verification + One Final Commit

### Task 1091: Run full unit test suite
- `pytest -q -o addopts=''`

### Task 1092: Run audits
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_no_tensorrt_imports.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 1093: One final commit
- `git status --porcelain`
- `git add -A`
- `git commit -m "chore: industrial v14 (no TensorRT imports; reduce try/except in spec models)"`

