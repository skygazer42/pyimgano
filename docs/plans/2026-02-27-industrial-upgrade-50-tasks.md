# PyImgAno Industrial Upgrade (50 Tasks) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add more **industrial-grade algorithms, embeddings, and anomaly synthesis** capabilities while staying dependency-stable (no new heavy runtime deps) and keeping a single final commit at the end.

**Architecture:** Extend three pillars without breaking existing contracts:
1) `core_*` detectors operate on 2D feature matrices and conform to `BaseDetector`.
2) Vision pipelines route `images -> embeddings -> core_*` via feature extractor registry.
3) Synthesis provides deterministic defect masks + blend modes + CLI dataset generation, with end-to-end tests.

**Tech Stack:** Python, NumPy, scikit-learn, OpenCV, scikit-image, PyTorch/TorchVision (optional).

**Commit policy (per user request):** Do **not** commit until all 50 tasks are completed; one final commit at the end.

---

## Phase A — Synthesis Masks + Presets (Tasks 1–20)

### Task 1: Add `random_brush_stroke_mask` (industrial brush/paint strokes)
**Files:** Modify `pyimgano/synthesis/masks.py`  
**Test:** Modify `tests/test_synthesis_masks.py`

### Task 2: Add `random_spatter_mask` (droplets / spatter defects)
**Files:** Modify `pyimgano/synthesis/masks.py`  
**Test:** Modify `tests/test_synthesis_masks.py`

### Task 3: Add `random_edge_band_mask` (edge-wear / border anomalies)
**Files:** Modify `pyimgano/synthesis/masks.py`  
**Test:** Modify `tests/test_synthesis_masks.py`

### Task 4: Add preset `brush`
**Files:** Modify `pyimgano/synthesis/presets.py`  
**Test:** Existing `tests/test_synthesis_presets.py` should pass.

### Task 5: Add preset `spatter`
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 6: Add preset `tape` (rectangular tape / patch)
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 7: Add preset `marker` (broad marker stroke)
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 8: Add preset `burn` (darkened Perlin burn mark)
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 9: Add preset `bubble` (air bubbles / circular bright/dark)
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 10: Add preset `fiber` (thin fibers / hairs)
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 11: Add preset `wrinkle` (waviness / crease)
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 12: Add preset `texture` (self-crop texture injection via Perlin mask)
**Files:** Modify `pyimgano/synthesis/presets.py`  
**Test:** Add a small smoke test in `tests/test_synthesis_presets.py` via registry loop.

### Task 13: Add preset `edge_wear` (edge-band + tint)
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 14: Add determinism regression test for one new preset
**Files:** Modify `tests/test_synthesis_determinism.py`

### Task 15: Add preview smoke coverage for new presets
**Files:** Modify `tests/test_synthesis_preview.py`

### Task 16: Add docs note for new synthesis presets
**Files:** Modify `README.md` (short section) + `docs/README_DOCS.md` (if needed)

### Task 17: Ensure all new preset metas include `preset=<name>`
**Files:** Modify `pyimgano/synthesis/presets.py`

### Task 18: Add ROI-mask edge-case tests for new masks
**Files:** Modify `tests/test_synthesis_roi_mask.py`

### Task 19: Add `__all__` exports for new masks (public API hygiene)
**Files:** Modify `pyimgano/synthesis/masks.py`

### Task 20: Run synthesis unit subset
**Verify:** `pytest -q tests/test_synthesis_masks.py tests/test_synthesis_presets.py`

---

## Phase B — Synthesis CLI + Dataset Workflow (Tasks 21–30)

### Task 21: Add `--presets` (mixed preset sampling) to `pyimgano-synthesize`
**Files:** Modify `pyimgano/synthesize_cli.py`  
**Test:** Modify `tests/test_synthesize_cli_smoke.py`

### Task 22: Add helper `make_preset_mixture(names)` for reuse
**Files:** Modify `pyimgano/synthesis/presets.py`  
**Test:** Add `tests/test_synthesis_preset_mixture.py`

### Task 23: Add `--roi-mask` option to restrict anomalies to ROI
**Files:** Modify `pyimgano/synthesize_cli.py`  
**Test:** Add `tests/test_synthesize_cli_roi_mask.py`

### Task 24: Make CLI emit chosen preset name in manifest metadata when mixing
**Files:** Modify `pyimgano/synthesize_cli.py`

### Task 25: Add E2E smoke: CLI generates dataset with `--presets`
**Files:** Modify `tests/test_synthesis_e2e_manifest_smoke.py`

### Task 26: Ensure CLI remains deterministic for same seed under `--presets`
**Files:** Add `tests/test_synthesize_cli_determinism_mix.py`

### Task 27: Extend `SyntheticAnomalyDataset` docs/examples for preset mixing
**Files:** Modify `pyimgano/datasets/synthetic.py`, `README.md`

### Task 28: Add quick recipe snippet: synthesize -> embedding-core -> score
**Files:** Modify `docs/README_DOCS.md` or add a short `docs/recipes/synth_embedding_core.md`

### Task 29: Add regression test for `--roi-mask` (empty ROI => always normal)
**Files:** `tests/test_synthesize_cli_roi_mask.py`

### Task 30: Run CLI subset
**Verify:** `pytest -q tests/test_synthesize_cli_smoke.py tests/test_synthesize_cli_preview.py`

---

## Phase C — Embedding Extractors (Tasks 31–38)

### Task 31: Add GeM pooling helper (`gem_pool2d`) (torch-only)
**Files:** Create `pyimgano/features/pooling.py`  
**Test:** Add `tests/test_feature_gem_pooling.py`

### Task 32: Add `torchvision_backbone_gem` extractor (layer4 feature-map + GeM)
**Files:** Create `pyimgano/features/torchvision_backbone_gem.py`  
**Test:** Add `tests/test_feature_torchvision_backbone_gem.py`

### Task 33: Export new extractor in `pyimgano/features/__init__.py` (registry discovery)
**Files:** Modify `pyimgano/features/__init__.py`

### Task 34: Add weight-download guard coverage for new extractor
**Files:** Modify `tests/test_no_torchvision_weight_downloads_by_default.py`

### Task 35: Add `gem_p` parameter + validate numeric stability
**Files:** Modify `pyimgano/features/pooling.py`, `pyimgano/features/torchvision_backbone_gem.py`

### Task 36: Add feature extractor tags/metadata for discoverability
**Files:** Modify `pyimgano/features/torchvision_backbone_gem.py`

### Task 37: Add `feature_cli` listing smoke for new extractor
**Files:** Modify `tests/test_cli_feature_discovery.py`

### Task 38: Run feature subset
**Verify:** `pytest -q tests/test_feature_torchvision_backbone_gem.py tests/test_cli_feature_discovery.py`

---

## Phase D — Models (Pixel-Map + New Core Detector) (Tasks 39–47)

### Task 39: Add `ssim_template_map` detector (pixel-map capable)
**Files:** Create `pyimgano/models/ssim_map.py`  
**Test:** Add `tests/test_ssim_map_detector.py`

### Task 40: Add `ssim_struct_map` detector (edges + pixel-map)
**Files:** Modify `pyimgano/models/ssim_map.py`  
**Test:** `tests/test_ssim_map_detector.py`

### Task 41: Add registry auto-import for `ssim_map`
**Files:** Modify `pyimgano/models/__init__.py`

### Task 42: Refactor `vision_crossmad` to be safe-by-default + BaseDetector-thresholded
**Files:** Modify `pyimgano/models/crossmad.py`  
**Test:** Add `tests/test_crossmad_contract.py`

### Task 43: Extend weight-download guard to include `vision_crossmad` fit smoke
**Files:** Modify `tests/test_no_torchvision_weight_downloads_by_default.py`

### Task 44: Add new torch-based core detector: `core_torch_autoencoder`
**Files:** Create `pyimgano/models/torch_autoencoder.py`  
**Test:** Add `tests/test_core_torch_autoencoder.py`

### Task 45: Add vision wrapper: `vision_torch_autoencoder` (feature extractor + core)
**Files:** Modify `pyimgano/models/torch_autoencoder.py`

### Task 46: Add registry auto-import for `torch_autoencoder`
**Files:** Modify `pyimgano/models/__init__.py`

### Task 47: Add model discovery smoke for new models
**Files:** Modify `tests/test_more_models_added.py`

---

## Phase E — Industrial Wrappers + Finalization (Tasks 48–50)

### Task 48: Add embedding-core industrial wrappers (resnet18 defaults + core baselines)
**Files:** Modify `pyimgano/models/industrial_wrappers.py`  
**Test:** Add `tests/test_industrial_wrappers_embeddings.py`

### Task 49: Update changelog + docs + third-party attributions (if any code copied)
**Files:** Modify `CHANGELOG.md`, `docs/INDUSTRIAL_REFERENCE_PROJECTS.md` (optional note)

### Task 50: Full verification + audits + single final commit
**Verify:**
- `pytest -q`
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
**Commit:** One final commit with all changes.

