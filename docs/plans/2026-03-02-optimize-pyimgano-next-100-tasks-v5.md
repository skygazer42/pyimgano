# PyImgAno Next 100 Tasks (v5) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the next industrial batch after v4: make deep models **safe-by-default offline**, expand `core_*` coverage and semantics, add a **patch/pixel-first** “embeddings + classical core” route that scales, and extend industrial synthesis/robustness — **without adding new required dependencies or implicit weight downloads**.

**Architecture (v5 focus):**
- Keep `BaseDetector` semantics as the “physics law” (**higher score ⇒ more anomalous**; contamination-based thresholding).
- Keep `core_*` detectors **feature-matrix first** (`np.ndarray` / torch tensors convertible to numpy) via `CoreFeatureDetector`.
- Keep `vision_*` detectors **paths/numpy** via feature extractors and pipelines.
- Promote “pixel maps + defects export” as the industrial output format, and keep manifest JSONL as the stable interchange.
- Treat foundation-weight models as **opt-in** (explicit pretrained/checkpoint/embedder); default behavior must be offline.

**Tech Stack:** Python, NumPy/SciPy, scikit-learn, scikit-image, OpenCV, Torch/Torchvision (offline by default), optional extras already present in this repo (diffusers/open_clip/faiss/anomalib/mamba — all guarded).

---

## Constraints / Non‑Negotiables (Industrial + Repo Policy)

- **No new required dependencies.** Optional extras must be guarded and must not affect `import pyimgano`.
- **No implicit weight downloads** (torchvision/openclip/diffusers/torch.hub). Any pretrained weights must be:
  - explicit opt-in (`pretrained=True` or explicit checkpoint path / embedder), and
  - unit-test safe (tests may hard-fail on any attempted download).
- **No bundled weights / large assets / datasets** in git.
- **Third-party code policy:** external repos can be shallow-cloned for study; copying code is allowed only with compatible licenses and mandatory notices/markers (see `docs/THIRD_PARTY_CODE_POLICY.md`).
- **Git policy for this batch:** **no incremental commits**; only **one final commit** after all tasks pass verification.

---

## Web Research Snapshot (2026‑03‑02)

Orientation only; implementation must remain dependency-stable and license-clean.

### Newer LVLM / agent-style directions (study-only in v5)
- IAD-GPT (ICLR 2025 workshop; LVLM-based IAD framing): https://openreview.net/forum?id=I7LkT8pWJ1
- VELM (visual expert language model for anomaly detection; arXiv): https://arxiv.org/abs/2508.00141

### CVPR 2025 method to study (implementable as classical core)
- Odd-One-Out neighbor comparison (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Odd-One-Out_Neighbor_Comparison_for_Robust_Visual_Anomaly_Detection_CVPR_2025_paper.html

---

## Phase 0 — Safety, Governance, Guardrails (Tasks 101–120)

### Task 101: Add a v5 guardrail test: no implicit downloads for selected deep models
**Files:**
- Create: `tests/test_no_implicit_weight_downloads_by_default_deep_models.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_no_implicit_weight_downloads_by_default_deep_models.py -v`

### Task 102: Make torchvision-heavy deep models safe-by-default (`pretrained=False`)
**Files:**
- Modify: `pyimgano/models/patchcore.py`
- Modify: `pyimgano/models/spade.py`
- Modify: `pyimgano/models/padim.py`
- Modify: `pyimgano/models/simplenet.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_no_implicit_weight_downloads_by_default_deep_models.py -v`

### Task 103: Make torch.hub foundation models explicit opt-in (require `embedder=` or `pretrained=True`)
**Files:**
- Modify: `pyimgano/models/anomalydino.py`
- Modify: `pyimgano/models/superad.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_no_implicit_weight_downloads_by_default_deep_models.py -v`
- `pytest -q -o addopts='' tests/test_anomalydino_core.py -v`

### Task 104: Refresh the industrial reference index with v5 snapshot
**Files:**
- Modify: `docs/INDUSTRIAL_REFERENCE_PROJECTS.md`

**Verify:**
- `python -c "import pathlib; pathlib.Path('docs/INDUSTRIAL_REFERENCE_PROJECTS.md').read_text(encoding='utf-8')"`

### Task 105: Expand study clone list (add v5 targets, shallow-only)
**Files:**
- Modify: `tools/clone_reference_repos.sh`

**Verify (optional):**
- `bash tools/clone_reference_repos.sh --dir .cache/pyimgano_refs --jobs 4`

### Task 106: Add a v5 “paper-to-module mapping” doc
**Files:**
- Create: `docs/PAPER_TO_MODULE_MAP_V5.md`

---

## Phase 1 — Coreization Expansion + Semantics Hardening (Tasks 121–150)

### Task 121: Coreize additional embedding-first baselines (cosine/robust/statistical)
**Files:**
- Create: `pyimgano/models/core_knn_cosine_calibrated.py`
- Create: `pyimgano/models/core_cosine_mahalanobis.py`
- Tests: `tests/test_core_knn_cosine_calibrated.py`, `tests/test_core_cosine_mahalanobis.py`

### Task 122: Add score-direction fixes where safe (keep BaseDetector contract)
**Files:**
- Modify: `pyimgano/models/core_knn_cosine.py`
- Modify: `pyimgano/models/core_kde_ratio.py`
- Tests: `tests/test_score_direction_core_models_v5.py`

### Task 123: Add a “core detector stability” contract test on torch inputs + NaNs
**Files:**
- Create: `tests/contracts/test_core_detector_contract_stability_v5.py`

---

## Phase 2 — Patch/Pixel-first “Embeddings + Classical Core” (Tasks 151–175)

### Task 151: Add a lightweight patch memory bank pixel-map baseline (PatchCore-lite-map)
**Files:**
- Create: `pyimgano/models/patchcore_lite_map.py`
- Modify: `pyimgano/models/__init__.py`
- Tests: `tests/test_patchcore_lite_map_smoke.py`, `tests/test_patchcore_lite_map_defects_e2e.py`

### Task 152: Add a torchvision conv-feature patch embedder utility (no downloads by default)
**Files:**
- Create: `pyimgano/features/torchvision_conv_patch_embedder.py`
- Tests: `tests/test_torchvision_conv_patch_embedder.py`

### Task 153: Add a “pixel-map capability” audit for new pixel-map models
**Files:**
- Modify: `tools/audit_registry.py` (or add `tools/audit_pixel_map_models.py`)

---

## Phase 3 — Industrial Synthesis v4 (banked defects + camera artifacts) (Tasks 176–195)

### Task 176: Add camera artifact synthesis (defocus + lens distortion)
**Files:**
- Create: `pyimgano/synthesis/camera_artifacts.py`
- Modify: `pyimgano/synthesis/presets.py`
- Tests: `tests/test_synthesis_camera_artifacts_presets.py`

### Task 177: Add “defect bank” copy/paste + Poisson option (industrial scars/stains)
**Files:**
- Create: `pyimgano/synthesis/defect_bank.py`
- Modify: `pyimgano/synthesize_cli.py` (new `--defect-bank-dir` opt-in)
- Tests: `tests/test_synthesis_defect_bank.py`, `tests/test_synthesize_cli_defect_bank.py`

### Task 178: Add CLI exposure for `--severity-range` and `--num-defects`
**Files:**
- Modify: `pyimgano/synthesize_cli.py`
- Tests: `tests/test_synthesize_cli_severity_num_defects.py`

### Task 195: E2E smoke: defect-bank + patchcore-lite-map → defects export
**Files:**
- Create: `tests/test_e2e_synth_defect_bank_patchcore_lite_map_defects.py`

---

## Phase 4 — Final Verification + One Final Commit (Tasks 196–200)

### Task 196: Run full unit test suite (no coverage addopts)
**Run:**
- `pytest -q -o addopts=''`
**Expected:** all pass (skips allowed only for optional deps).

### Task 197: Run audits (public API / registry / score direction / third-party notices / import costs)
**Run:**
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
**Expected:** OK (heuristic WARNs reviewed and documented if needed).

### Task 198: Update changelog with concrete bullets for shipped v5 items
**Files:**
- Modify: `CHANGELOG.md`

### Task 199: One final commit (single commit policy)
**Run:**
- `git status --porcelain`
- `git add -A`
- `git commit -m \"feat: industrial MVP loop v5 (safe deep defaults, patch/pixel pipelines, synthesis v4)\"`
**Expected:** exactly one new commit.

