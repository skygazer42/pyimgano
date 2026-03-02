# PyImgAno Next 100 Tasks (v13) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend “spec-friendly ensembles” by adding **Feature Bagging** variants that accept JSON-friendly base-estimator specs.

**Why (industrial):**
- Feature Bagging is a cheap stability booster for classical detectors on embeddings/features.
- Industrial deployments often require JSON/YAML configs (string/dict specs), not Python objects.

**Constraints:** no new deps, preserve `BaseDetector` semantics (higher score ⇒ more anomalous), one final commit.

---

## Phase 0 — Tests

### Task 901: Add failing tests for spec-friendly feature bagging
**Files:**
- Create: `tests/test_feature_bagging_spec_v13.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_feature_bagging_spec_v13.py -v`

---

## Phase 1 — Implementation

### Task 911: Add `base_estimator_spec` support to the Feature Bagging backend
**Design:**
- Extend `CoreFeatureBagging` to accept `base_estimator_spec` (string/dict/instance).
- Resolve string/dict specs via `pyimgano.models.ensemble_spec.resolve_model_spec`.
- Ensure each estimator gets its own base-estimator instance (clone per estimator).
- Keep legacy `base_estimator="lof"` behavior unchanged.

**Files:**
- Modify: `pyimgano/models/feature_bagging.py`

### Task 912: Register spec-friendly models
**Models:**
- `core_feature_bagging_spec`
- `vision_feature_bagging_spec`

**Files:**
- Modify: `pyimgano/models/feature_bagging.py`

---

## Phase 2 — Docs + Changelog

### Task 921: Document Feature Bagging spec ensembles
**Files:**
- Modify: `docs/ARCHITECTURE_CLASSICAL_PIPELINES.md`

### Task 922: Update changelog
**Files:**
- Modify: `CHANGELOG.md`

---

## Phase 3 — Verification + One Final Commit

### Task 991: Run full unit test suite
- `pytest -q -o addopts=''`

### Task 992: Run audits
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 993: One final commit
- `git status --porcelain`
- `git add -A`
- `git commit -m "feat: industrial v13 (spec-friendly feature bagging ensembles)"`

