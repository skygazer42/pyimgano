# PyImgAno Next 100 Tasks (v11) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make torchvision-based embedding extractors more industrial-friendly by accepting **channels-first numpy arrays** (`(C,H,W)`) in addition to the existing `(H,W,C)` convention.

**Why (industrial):**
- Many upstream CV pipelines produce `numpy` images in CHW (PyTorch-style) for speed and consistency.
- Supporting CHW at the feature extractor boundary reduces glue code and prevents subtle color/shape bugs.

**Constraints:** no new deps, offline-safe defaults unchanged, one final commit.

---

## Phase 0 — Tests

### Task 701: Add failing test for CHW numpy inputs
**Files:**
- Create: `tests/test_feature_torchvision_backbone_numpy_chw_inputs.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_feature_torchvision_backbone_numpy_chw_inputs.py -v`

---

## Phase 1 — Implementation

### Task 711: Support CHW numpy arrays in `_as_pil_rgb`
**Design:**
- If a numpy image is `(3,H,W)` or `(1,H,W)`, transpose to `(H,W,C)` and reuse existing path.
- Preserve existing behavior for grayscale `(H,W)` and HWC `(H,W,C)`.

**Files:**
- Modify: `pyimgano/features/torchvision_backbone.py`

---

## Phase 2 — Verification + Commit

### Task 791: Run full unit test suite
- `pytest -q -o addopts=''`

### Task 792: Run audits
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 793: Update changelog
- Modify: `CHANGELOG.md`

### Task 794: One final commit
- `git add -A`
- `git commit -m \"feat: industrial v11 (numpy CHW support for torchvision embeddings)\"`

