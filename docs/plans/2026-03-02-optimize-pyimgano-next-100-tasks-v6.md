# PyImgAno Next 100 Tasks (v6) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** After v5, tighten **pixel-map discovery/contract** (tags + APIs), add a generic, industrial **patch-embedding + classical core → anomaly map** route, and improve defect-bank realism (alpha/Poisson blending) — while staying **offline-safe by default**, dependency-stable, and auditable.

**Architecture (v6 focus):**
- Treat **pixel maps** (`predict_anomaly_map`) as the industrial output interface; image-level scores remain a reduction of maps.
- Keep `BaseDetector` semantics as the “physics law”: **higher score ⇒ more anomalous**.
- Keep “deep embeddings + classical core” as the stable deployable route:
  - image-level: `vision_embedding_core`
  - patch-level (new): `vision_patch_embedding_core_map`
- Keep all deep/foundation models **explicit opt-in** for pretrained weights / checkpoints; default behavior must be offline.

**Tech Stack:** Python, NumPy/SciPy, scikit-learn, OpenCV, Torch/Torchvision (offline by default), optional extras already present (anomalib/open_clip/diffusers/mamba/faiss — all guarded).

---

## Constraints / Non‑Negotiables (Industrial + Repo Policy)

- **No new required dependencies.** Optional extras must be guarded and must not affect `import pyimgano`.
- **No implicit weight downloads** (torchvision/openclip/diffusers/torch.hub). Any pretrained weights must be explicit opt-in.
- **No bundled weights / large assets / datasets** in git.
- **Third-party code policy:** shallow-clone refs for study only; copying code requires compatible license + notices (see `docs/THIRD_PARTY_CODE_POLICY.md`).
- **Git policy for this batch:** **no incremental commits**; only **one final commit** after all tasks pass verification.

---

## Phase 0 — Pixel-Map Contract & Discovery (Tasks 201–230)

### Task 201: Auto-tag pixel-map models when methods exist
**Goal:** prevent drift where a model defines `predict_anomaly_map/get_anomaly_map` but is missing the `pixel_map` tag.

**Files:**
- Modify: `pyimgano/models/registry.py` (enhance `register_model` decorator)

**Verify:**
- `python -c "import pyimgano.models; from pyimgano.models.registry import model_info; print(model_info('ssim_template_map')['supports_pixel_map'])"`

### Task 202: Fix DifferNet pixel-map contract (implement anomaly maps + image-level scoring)
**Files:**
- Modify: `pyimgano/models/differnet.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_pixel_map_audit_strict_v6.py -v`

### Task 203: Add strict audit test for pixel-map registry consistency
**Files:**
- Create: `tests/test_pixel_map_audit_strict_v6.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_pixel_map_audit_strict_v6.py -v`

---

## Phase 1 — Patch-Embedding + Classical Core → Pixel Maps (Tasks 231–270)

### Task 231: Add `vision_patch_embedding_core_map` (generic industrial baseline)
**Design:** torchvision conv patch embedder → stack patches → fit `core_*` detector → patch scores → anomaly map → reduce to image score.

**Files:**
- Create: `pyimgano/models/patch_embedding_core_map.py`
- Modify: `pyimgano/models/__init__.py` (registry import)
- Tests:
  - Create: `tests/test_patch_embedding_core_map_smoke.py`
  - Create: `tests/test_patch_embedding_core_map_defects_e2e.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_patch_embedding_core_map_smoke.py -v`
- `pytest -q -o addopts='' tests/test_patch_embedding_core_map_defects_e2e.py -v`

---

## Phase 2 — Defect Bank Realism (Blending Modes) (Tasks 271–290)

### Task 271: Add alpha/Poisson blend options for defect-bank preset
**Files:**
- Modify: `pyimgano/synthesis/defect_bank.py`
- Modify: `pyimgano/synthesize_cli.py` (surface blend mode via CLI)
- Tests:
  - Modify/Create: `tests/test_synthesis_defect_bank.py`
  - Modify/Create: `tests/test_synthesize_cli_defect_bank.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_synthesis_defect_bank.py -v`
- `pytest -q -o addopts='' tests/test_synthesize_cli_defect_bank.py -v`

---

## Phase 3 — Final Verification + One Final Commit (Tasks 291–300)

### Task 291: Run full unit test suite
Run:
- `pytest -q -o addopts=''`

### Task 292: Run audits
Run:
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 293: Update changelog
Files:
- Modify: `CHANGELOG.md`

### Task 294: One final commit (single commit policy)
Run:
- `git status --porcelain`
- `git add -A`
- `git commit -m \"feat: industrial v6 batch (pixel-map contract, patch-embedding core maps, defect-bank blending)\"`

