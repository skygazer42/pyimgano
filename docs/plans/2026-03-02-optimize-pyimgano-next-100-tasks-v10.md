# PyImgAno Next 100 Tasks (v10) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tighten the “deep embedding + classical core” industrial route by improving **in-memory input support** for torchvision embeddings and making **LSCP ensembles** usable out-of-the-box in both `core_*` and `vision_*` forms.

**Architecture (v10 focus):**
- Keep `BaseDetector` semantics stable: **higher score ⇒ more anomalous**.
- Keep deep/embedding defaults **offline-safe** (`pretrained=False` by default; no implicit downloads).
- Improve ergonomics without changing dependency surface:
  - Allow `TorchvisionBackboneExtractor` to accept `torch.Tensor` images directly.
  - Make `vision_lscp` / `vision_lscp_spec` default to a cheap, diverse detector set (mirrors `core_lscp`).

**Tech Stack:** Python, NumPy, Torch/Torchvision, Pillow (already required).

---

## Constraints / Non‑Negotiables

- **No new required dependencies.**
- **No implicit weight downloads** (torchvision/openclip/diffusers).
- **Single final commit policy for this batch.**

---

## Phase 0 — Tests (Guardrails)

### Task 601: Add failing test for `TorchvisionBackboneExtractor` torch.Tensor inputs
**Files:**
- Create: `tests/test_feature_torchvision_backbone_torch_tensor_inputs.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_feature_torchvision_backbone_torch_tensor_inputs.py -v`

### Task 602: Add failing tests for `vision_lscp` defaults (no explicit detector list)
**Files:**
- Create: `tests/test_vision_lscp_defaults.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_vision_lscp_defaults.py -v`

---

## Phase 1 — Implementation

### Task 611: Support `torch.Tensor` images in `torchvision_backbone`
**Design:**
- Extend `_as_pil_rgb` to accept `torch.Tensor` in (C,H,W) and (H,W,C) forms.
- Keep behavior for paths / numpy arrays / PIL unchanged.

**Files:**
- Modify: `pyimgano/features/torchvision_backbone.py`

### Task 612: Provide defaults + expose missing knobs for `vision_lscp` / `vision_lscp_spec`
**Design:**
- If `detector_list` / `detector_specs` is None, reuse `_default_lscp_detectors(...)`.
- Add passthrough args to control `local_region_iterations` and `local_min_features` (optional; keeps runtime tunable).

**Files:**
- Modify: `pyimgano/models/lscp.py`

---

## Phase 2 — Final Verification + One Final Commit

### Task 691: Run full unit test suite
Run:
- `pytest -q -o addopts=''`

### Task 692: Run audits
Run:
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 693: Update changelog
**Files:**
- Modify: `CHANGELOG.md`

### Task 694: One final commit
Run:
- `git status --porcelain`
- `git add -A`
- `git commit -m \"feat: industrial v10 (torch tensor embeddings + LSCP defaults)\"`

