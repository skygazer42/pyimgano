# PyImgAno Next 100 Tasks (v9) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand “industrial default routes” by (1) **core-ifying spec-friendly ensembles** (LSCP/SUOD) so they can be used directly on feature matrices, and (2) making the primary **torchvision embedding extractor** accept common in-memory image types (PIL) — while preserving offline safety and `BaseDetector` semantics.

**Architecture (v9 focus):**
- Keep `BaseDetector` semantics as the “physics law”: **higher score ⇒ more anomalous**, thresholding derived from `contamination`.
- Keep `core_*` detectors **feature-matrix first** (`np.ndarray` + torch tensors convertible to numpy) via `CoreFeatureDetector`.
- Treat “spec-friendly config” as a first-class industrial need: JSON-friendly base-detector specs should work for both `vision_*` and `core_*` ensembles.
- Keep deep/embedding defaults **offline-safe** (no implicit weight downloads; `pretrained=False` by default).

**Tech Stack:** Python, NumPy, scikit-learn, Torch/Torchvision, Pillow (already in repo dependencies).

---

## Constraints / Non‑Negotiables

- **No new required dependencies.**
- **No implicit weight downloads** (torchvision/openclip/diffusers/torch.hub). Any pretrained weights remain explicit opt-in.
- **Single final commit policy for this batch:** do not commit until all verification passes.

---

## Phase 0 — Guardrails (Tests First)

### Task 501: Add failing tests for `core_lscp_spec` (feature-matrix + specs)
**Files:**
- Create: `tests/test_core_lscp_spec.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_core_lscp_spec.py -v`
**Expected:** FAIL (model not registered yet).

### Task 502: Add failing tests for `core_suod_spec` (feature-matrix + specs)
**Files:**
- Create: `tests/test_core_suod_spec.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_core_suod_spec.py -v`
**Expected:** FAIL (model not registered yet).

### Task 503: Add failing test: `torchvision_backbone` accepts PIL image inputs
**Files:**
- Create: `tests/test_feature_torchvision_backbone_pil_inputs.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_feature_torchvision_backbone_pil_inputs.py -v`
**Expected:** FAIL (TypeError for PIL inputs).

---

## Phase 1 — Implementation (Core-ify + Input-Type Support)

### Task 511: Register `core_lscp` and `core_lscp_spec`
**Design:**
- Add `CoreLSCPModel(CoreFeatureDetector)` that wraps `CoreLSCP(detector_list=...)`.
- Add `CoreLSCPSpecModel(CoreFeatureDetector)` that accepts `detector_specs` and resolves them via `pyimgano.models.ensemble_spec.resolve_model_specs`.

**Files:**
- Modify: `pyimgano/models/lscp.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_core_lscp_spec.py -v`

### Task 512: Register `core_suod_spec`
**Design:**
- Add `CoreSUODSpecModel(CoreFeatureDetector)` that accepts `base_estimator_specs` and resolves them via `resolve_model_specs`, then uses `CoreSUOD(base_estimators=...)`.

**Files:**
- Modify: `pyimgano/models/suod.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_core_suod_spec.py -v`

### Task 513: Support PIL inputs in `TorchvisionBackboneExtractor`
**Design:**
- Extend `_as_pil_rgb` to accept `PIL.Image.Image` and convert to RGB.
- Keep existing behavior for `str|Path|np.ndarray`.

**Files:**
- Modify: `pyimgano/features/torchvision_backbone.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_feature_torchvision_backbone_pil_inputs.py -v`

---

## Phase 2 — Docs (Discovery UX)

### Task 521: Document core spec ensembles as stable industrial blocks
**Files:**
- Modify: `docs/ARCHITECTURE_CLASSICAL_PIPELINES.md`

**Verify:**
- `python -c "import pathlib; pathlib.Path('docs/ARCHITECTURE_CLASSICAL_PIPELINES.md').read_text(encoding='utf-8')"`

---

## Phase 3 — Final Verification + One Final Commit

### Task 591: Run full unit test suite
Run:
- `pytest -q -o addopts=''`

### Task 592: Run audits
Run:
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 593: Update changelog
**Files:**
- Modify: `CHANGELOG.md`

### Task 594: One final commit (single commit policy)
Run:
- `git status --porcelain`
- `git add -A`
- `git commit -m "feat: industrial v9 (core spec ensembles + PIL-friendly torchvision embeddings)"`

