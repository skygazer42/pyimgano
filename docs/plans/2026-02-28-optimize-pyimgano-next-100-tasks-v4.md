# PyImgAno Next 100 Tasks (v4) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a stable industrial MVP loop centered on **manifest datasets + pixel anomaly maps + defects export**, while expanding `core_*` coverage and adding a small set of **modern deep methods** that fit our existing base contracts and feature/pipeline system — **without adding new required dependencies or implicit weight downloads**.

**Architecture (v4 focus):**
- Keep `BaseDetector` semantics as the “physics law” (higher score ⇒ more anomalous; contamination-based thresholding).
- Keep `core_*` detectors **feature-matrix first** (`np.ndarray` / torch tensors convertible to numpy) via `CoreFeatureDetector`.
- Keep `vision_*` detectors **paths/numpy** via feature extractors and pipelines.
- Treat **pixel maps** as first-class: anomaly maps feed defects extraction (mask + regions), overlays, and industrial outputs.
- Treat **manifest JSONL** as the stable interchange format for datasets, synthetic generation, and evaluation loops.

**Tech Stack:** Python, NumPy/SciPy, scikit-learn, scikit-image, OpenCV, Torch/Torchvision (no implicit downloads), optional extras already present in this repo (diffusers/open_clip/faiss/anomalib/mamba — all guarded).

---

## Constraints / Non‑Negotiables (Industrial + Repo Policy)

- **No new required dependencies.** Optional extras must be guarded and must not affect `import pyimgano`.
- **No implicit weight downloads** (torchvision/openclip/diffusers). Any pretrained weights must be:
  - opt-in (`pretrained=True` or explicit checkpoint path), and
  - test-suite safe (tests must forbid downloads).
- **No bundled weights / large assets / datasets** in git.
- **Third-party code policy:** external repos can be shallow-cloned for study; copying code is allowed only with compatible licenses and mandatory notices/markers (see `docs/THIRD_PARTY_CODE_POLICY.md`).
- **Git policy for this batch:** **no incremental commits**; only **one final commit** after all tasks pass verification.

---

## Web Research Snapshot (2026‑02‑28)

This section is for *orientation only*; the implementation work must remain dependency-stable and license-clean.

### Datasets / Benchmarks to align with

- **MVTec AD 2** (dataset + paper; license: CC BY‑NC‑SA 4.0)
  - https://www.mvtec.com/company/research/datasets/mvtec-ad-2
  - https://arxiv.org/abs/2503.21622
- **VAND @ CVPR 2025** (benchmark framing; SegF1 @ single threshold is common)
  - https://cvpr2025.thecvf.com/virtual/2025/workshop/36626
  - https://openaccess.thecvf.com/CVPR2025_workshops/VAND.html
- **Real‑IAD** (large-scale industrial dataset; access-gated)
  - https://realiad4ad.github.io/Real-IAD/
  - Real‑IAD D3: https://arxiv.org/abs/2504.14221
- **ReinAD** (reinforced inspection; new settings)
  - https://reinad.ai/
- **RAD** (robotic multi-view, viewpoint + lighting variation)
  - https://rad-iad.github.io/ (project page)
  - https://arxiv.org/abs/2411.12179
- **MMAD** (ICLR 2025 benchmark for multimodal anomaly detection)
  - https://github.com/jam-cc/MMAD
- “Awesome / survey indexes” (for discovering additional methods + repos)
  - https://github.com/M-3LAB/awesome-industrial-anomaly-detection
  - https://github.com/IHPCRits/IAD-Survey

### Methods to study (implementation feasibility filtered by our dependency rules)

- **AnomalyAny** (CVPR 2025; diffusion-based anomaly generation ideas — *study-only unless we can make it optional and no-download*)
  - https://github.com/EPFL-IMOS/AnomalyAny
- **AgentIAD** (arXiv 2025; LVLM pipeline ideas — likely study-only in v4)
  - https://arxiv.org/abs/2512.13671
- **AutoIAD** (arXiv 2025; self-improving loop ideas — likely study-only in v4)
  - https://arxiv.org/abs/2508.05503
- **ADClick** (arXiv 2025; interactive correction loop ideas — study-only)
  - https://arxiv.org/abs/2509.05034

### Surveys (for method taxonomy + evaluation patterns)

- 2025 survey (industrial visual anomaly detection): https://arxiv.org/abs/2506.05441
- 2025 survey (deep learning for industrial anomaly detection): https://arxiv.org/abs/2503.05088

---

## Phase 0 — Research, Governance, Guardrails (Tasks 1–15)

### Task 1: Update research index with v4 web snapshot
**Files:**
- Modify: `docs/INDUSTRIAL_REFERENCE_PROJECTS.md`

**Verify:**
- Render check: `python -c "import pathlib; pathlib.Path('docs/INDUSTRIAL_REFERENCE_PROJECTS.md').read_text(encoding='utf-8')"`
**Expected:** no exceptions.

### Task 2: Expand study clone list (v4 targets, shallow-only)
**Files:**
- Modify: `tools/clone_reference_repos.sh`

**Verify:**
- Run (optional): `bash tools/clone_reference_repos.sh --dir .cache/pyimgano_refs --jobs 4`
**Expected:** repos clone or gracefully “skip/fail” without breaking script.

### Task 3: Add a v4 “paper-to-module mapping” doc for planned imports
**Files:**
- Create: `docs/PAPER_TO_MODULE_MAP_V4.md`

**Verify:**
- Smoke: `python -c "import pathlib; assert 'Task' not in pathlib.Path('docs/PAPER_TO_MODULE_MAP_V4.md').read_text()"`
**Expected:** doc is not a todo dump; it maps papers → modules/contracts.

### Task 4: Add a regression test to prevent `pyimgano-infer` from enabling maps accidentally
**Why:** `--defects` must imply maps, but `--include-maps` alone should not imply defects.
**Files:**
- Create: `tests/test_infer_cli_maps_vs_defects_flags.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_infer_cli_maps_vs_defects_flags.py -v`

### Task 5: Add a “no implicit downloads” test for OpenCLIP extractor defaults
**Files:**
- Create: `tests/test_no_openclip_weight_downloads_by_default.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_no_openclip_weight_downloads_by_default.py -v`

### Task 6: Add an audit that `core_*` models accept torch tensors
**Files:**
- Create: `tests/test_core_models_accept_torch_tensor_inputs.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_core_models_accept_torch_tensor_inputs.py -v`

### Task 7: Add docs: “Industrial MVP loop (synthesize → infer → defects)”
**Files:**
- Create: `docs/INDUSTRIAL_MVP_LOOP.md`

**Verify:**
- `python -c "import pathlib; pathlib.Path('docs/INDUSTRIAL_MVP_LOOP.md').read_text(encoding='utf-8')"`

### Task 8: Add a `pyimgano` CLI snippet in docs that uses `ssim_template_map` as pixel baseline
**Files:**
- Modify: `docs/INDUSTRIAL_INFERENCE.md`

**Verify:**
- `python -c "import pathlib; assert 'ssim_template_map' in pathlib.Path('docs/INDUSTRIAL_INFERENCE.md').read_text(encoding='utf-8')"`

### Task 9: Harden third-party notice auditing for copied code (better error UX)
**Files:**
- Modify: `tools/audit_third_party_notices.py`

**Verify:**
- `python tools/audit_third_party_notices.py`
**Expected:** OK or actionable errors.

### Task 10: Add an audit that no `.cache/pyimgano_refs` files are tracked by git
**Files:**
- Create: `tools/audit_no_reference_clones_tracked.py`

**Verify:**
- `python tools/audit_no_reference_clones_tracked.py`
**Expected:** OK.

### Task 11: Add an `__init__`/lazy-import audit for new modules (keep import light)
**Files:**
- Modify: `tools/audit_import_costs.py`

**Verify:**
- `python tools/audit_import_costs.py`
**Expected:** import timing prints; no crashes.

### Task 12: Add docs: “MVTec AD 2 notes (license + evaluation + gotchas)”
**Files:**
- Create: `docs/DATASET_MVTEC_AD2_NOTES.md`

### Task 13: Add docs: “Real-IAD / RAD / ReinAD settings we support”
**Files:**
- Create: `docs/DATASET_REAL_WORLD_IAD_NOTES.md`

### Task 14: Add `pyimgano-benchmark --dataset mvtec_ad2` placeholder error message
**Why:** avoid silent “custom dataset” misinterpretation before implementation lands.
**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_mvtec_ad2_placeholder.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_cli_mvtec_ad2_placeholder.py -v`

### Task 15: Update `CHANGELOG.md` “Unreleased” section for v4 batch scope
**Files:**
- Modify: `CHANGELOG.md`

---

## Phase 1 — Dataset Converters + Manifest Hardening (Tasks 16–35)

### Task 16: Add MVTec AD 2 manifest converter (paths-first)
**Files:**
- Create: `pyimgano/datasets/mvtec_ad2.py`
- Modify: `pyimgano/datasets/__init__.py`
- Test: `tests/test_dataset_mvtec_ad2_manifest_converter.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_dataset_mvtec_ad2_manifest_converter.py -v`

### Task 17: Add CLI entry: `pyimgano-manifest --dataset mvtec_ad2`
**Files:**
- Modify: `pyimgano/manifest_cli.py`
- Test: `tests/test_manifest_cli_mvtec_ad2.py`

### Task 18: Add Real-IAD “layout recognizer” (best-effort) and manifest converter
**Files:**
- Create: `pyimgano/datasets/real_iad.py`
- Test: `tests/test_dataset_real_iad_converter_smoke.py`

### Task 19: Add RAD manifest converter (multi-view metadata in `meta`)
**Files:**
- Create: `pyimgano/datasets/rad.py`
- Test: `tests/test_dataset_rad_converter_smoke.py`

### Task 20: Extend manifest record schema to allow `meta.view_id` and `meta.condition`
**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_meta_fields_roundtrip.py`

### Task 21: Add manifest splitting policy: group-aware split (by `meta.group_id`)
**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_group_split_policy.py`

### Task 22: Add manifest validation: required keys + path existence checks
**Files:**
- Modify: `pyimgano/validate_infer_config_cli.py`
- Create: `pyimgano/datasets/manifest_validate.py`
- Test: `tests/test_manifest_validate.py`

### Task 23: Add `pyimgano-manifest --validate` mode
**Files:**
- Modify: `pyimgano/manifest_cli.py`
- Test: `tests/test_manifest_cli_validate.py`

### Task 24: Add `load_masks="auto"` mode for manifest split loader
**Why:** avoid loading masks when models don’t need them.
**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_load_masks_auto.py`

### Task 25: Add “mask missing” policy for manifest datasets (skip vs error)
**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_missing_masks_policy.py`

### Task 26: Add `pyimgano-benchmark --dataset manifest --pixel` compatibility for mask-less records
**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_manifest_pixel_requires_masks.py`

### Task 27: Add docs: “Manifest meta schema for industrial multi-view”
**Files:**
- Modify: `docs/MANIFEST_DATASET.md`

### Task 28: Add dataset converter registry (name → converter)
**Files:**
- Create: `pyimgano/datasets/converters.py`
- Test: `tests/test_dataset_converters_registry.py`

### Task 29: Wire converter registry into `pyimgano-manifest`
**Files:**
- Modify: `pyimgano/manifest_cli.py`

### Task 30: Add synthetic dataset converter test coverage for `include_masks=True`
**Files:**
- Modify: `tests/test_synthesis_e2e_manifest_smoke.py`

### Task 31: Add “path normalization” helper (Windows + POSIX)
**Files:**
- Create: `pyimgano/utils/path_normalize.py`
- Test: `tests/test_path_normalize.py`

### Task 32: Use path normalization in manifest read/write paths
**Files:**
- Modify: `pyimgano/datasets/manifest.py`

### Task 33: Add `pyimgano-datasets` discovery CLI (list converters + info)
**Files:**
- Create: `pyimgano/datasets_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_datasets_cli_smoke.py`

### Task 34: Add docs: “Dataset converters: mvtec_ad2 / real_iad / rad”
**Files:**
- Create: `docs/DATASET_CONVERTERS.md`

### Task 35: Add minimal examples for converters under `examples/`
**Files:**
- Create: `examples/convert_mvtec_ad2_to_manifest.py`
- Create: `examples/convert_rad_to_manifest.py`

---

## Phase 2 — Pixel‑First Baselines + Defects Pipeline v2 (Tasks 36–55)

### Task 36: Add `vision_template_ncc_map` (normalized cross-correlation anomaly map)
**Files:**
- Create: `pyimgano/models/template_ncc_map.py`
- Modify: `pyimgano/models/__init__.py`
- Test: `tests/test_template_ncc_map_detector.py`

### Task 37: Add `vision_phase_correlation_map` (misalignment-tolerant template baseline)
**Files:**
- Create: `pyimgano/models/phase_corr_map.py`
- Test: `tests/test_phase_corr_map_detector.py`

### Task 38: Add registry tags for pixel-map models (`pixel_map`, `template`)
**Files:**
- Modify: `pyimgano/models/capabilities.py`
- Test: `tests/test_model_capabilities_pixel_map_tags.py`

### Task 39: Add `infer` API support for detectors returning maps via `decision_function` tuple
**Why:** some models naturally produce (score, map).
**Files:**
- Modify: `pyimgano/inference/api.py`
- Test: `tests/test_inference_api_tuple_outputs.py`

### Task 40: Add pixel-threshold calibration: “max SegF1 on val split” (optional supervised)
**Files:**
- Create: `pyimgano/calibration/pixel_threshold_supervised.py`
- Test: `tests/test_pixel_threshold_supervised_segf1.py`

### Task 41: Add `pyimgano-benchmark --pixel-threshold-strategy supervised_segf1`
**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_pixel_threshold_supervised_segf1.py`

### Task 42: Extend defects extraction to emit region solidity / aspect ratio stats
**Files:**
- Modify: `pyimgano/defects/extract.py`
- Test: `tests/test_defects_extract_region_stats.py`

### Task 43: Add defects export option: `--defects-regions-jsonl` (per-image regions file)
**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_defects_regions_jsonl.py`

### Task 44: Add defects export option: `--defects-mask-dilate` (industrial fill)
**Files:**
- Modify: `pyimgano/defects/binary_postprocess.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_defects_mask_dilate.py`

### Task 45: Add ROI behavior tests: ROI applies to regions, but mask export can be “full” or “roi”
**Files:**
- Modify: `pyimgano/infer_cli.py`
- Create: `tests/test_infer_cli_defects_roi_mask_space.py`

### Task 46: Add overlays: include region IDs + score stats in filename (debug UX)
**Files:**
- Modify: `pyimgano/defects/overlays.py`
- Test: `tests/test_defects_overlays_naming.py`

### Task 47: Add defects export stability test on synthetic data (real model)
**Files:**
- Create: `tests/test_e2e_synth_ssim_struct_defects_infer_cli.py`

### Task 48: Add tile-map integration test with defects on high-res synthetic image
**Files:**
- Create: `tests/test_e2e_tiling_defects_smoke.py`

### Task 49: Add docs: “Defects export contract (mask + regions + provenance)”
**Files:**
- Modify: `docs/INDUSTRIAL_INFERENCE.md`

### Task 50: Add CLI reference updates for new defects flags
**Files:**
- Modify: `docs/CLI_REFERENCE.md`

### Task 51: Add `pyimgano-defects` CLI (standalone map→mask→regions tool)
**Files:**
- Create: `pyimgano/defects_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_defects_cli_smoke.py`

### Task 52: Add defect-region filtering by “score quantile within ROI”
**Files:**
- Modify: `pyimgano/defects/extract.py`
- Test: `tests/test_defects_extract_score_quantile_filter.py`

### Task 53: Add postprocess preset `industrial-defects-fp40` to `pyimgano-infer`
**Files:**
- Modify: `pyimgano/cli_presets.py`
- Test: `tests/test_cli_presets_defects_fp40.py`

### Task 54: Add an end-to-end example script for pixel-first defects baseline
**Files:**
- Create: `examples/pixel_first_ssim_defects.py`

### Task 55: Add docs page: “Pixel-first baselines (SSIM/NCC/phase-corr)”
**Files:**
- Create: `docs/RECIPES_PIXEL_FIRST_BASELINES.md`

---

## Phase 3 — Embeddings + Core: Make It the “Default Industrial Route” (Tasks 56–75)

### Task 56: Add `torchvision_backbone` pooling options (`avg|max|gem|cls`)
**Files:**
- Modify: `pyimgano/features/torchvision_backbone.py`
- Test: `tests/test_feature_torchvision_backbone_pooling_modes.py`

### Task 57: Add `torchvision_backbone` “channels-last” option for speed
**Files:**
- Modify: `pyimgano/features/torchvision_backbone.py`
- Test: `tests/test_feature_torchvision_backbone_channels_last.py`

### Task 58: Add embedding extractor “amp=True” option (best-effort)
**Files:**
- Modify: `pyimgano/features/torchvision_backbone.py`
- Test: `tests/test_feature_torchvision_backbone_amp_best_effort.py`

### Task 59: Add embedding extractor “compile=True” option (torch.compile best-effort)
**Files:**
- Modify: `pyimgano/features/torchvision_backbone.py`
- Test: `tests/test_feature_torchvision_backbone_compile_best_effort.py`

### Task 60: Add `torchvision_backbone_patch_tokens` extractor (for patch-based cores)
**Files:**
- Create: `pyimgano/features/torchvision_patch_tokens.py`
- Modify: `pyimgano/features/__init__.py`
- Test: `tests/test_feature_torchvision_patch_tokens.py`

### Task 61: Add `core_knn_cosine` optimized for normalized embeddings
**Files:**
- Create: `pyimgano/models/core_knn_cosine.py`
- Modify: `pyimgano/models/__init__.py`
- Test: `tests/test_core_knn_cosine.py`

### Task 62: Add `core_mahalanobis_shrinkage` (Ledoit-Wolf) for embeddings
**Files:**
- Create: `pyimgano/models/core_mahalanobis_shrinkage.py`
- Test: `tests/test_core_mahalanobis_shrinkage.py`

### Task 63: Add `vision_embedding_core` preset mapping for stable industrial defaults
**Files:**
- Modify: `pyimgano/models/industrial_wrappers.py`
- Test: `tests/test_industrial_wrappers_embedding_core_presets.py`

### Task 64: Add workbench recipe: `industrial-embedding-core-fast`
**Files:**
- Modify: `pyimgano/recipes/industrial_adapt.py`
- Test: `tests/test_workbench_recipe_embedding_core_fast.py`

### Task 65: Add “embedding cache” for torchvision extractors (disk cache keyed by path+mtime)
**Files:**
- Create: `pyimgano/cache/embeddings.py`
- Modify: `pyimgano/features/torchvision_backbone.py`
- Test: `tests/test_embedding_cache_paths.py`

### Task 66: Add CLI support: `pyimgano-features --cache-dir`
**Files:**
- Modify: `pyimgano/feature_cli.py`
- Test: `tests/test_feature_cli_cache_dir.py`

### Task 67: Add end-to-end test: embedding cache speeds up second pass (best-effort)
**Files:**
- Create: `tests/test_embedding_cache_e2e_best_effort.py`

### Task 68: Add core-model contract test: `decision_function` shape + dtype + finite
**Files:**
- Create: `tests/contracts/test_core_detector_contract.py`

### Task 69: Add registry metadata: add `input_modes` and `supports_pixel_map` to CLI output
**Files:**
- Modify: `pyimgano/models/registry.py`
- Test: `tests/test_model_info_payload_has_capabilities.py`

### Task 70: Add docs: “Embeddings + Core” update with pooling modes
**Files:**
- Modify: `docs/RECIPES_EMBEDDINGS_PLUS_CORE.md`

### Task 71: Add example: `torchvision_backbone` + `core_mahalanobis_shrinkage`
**Files:**
- Create: `examples/embedding_plus_core_mahalanobis_shrinkage.py`

### Task 72: Add benchmark smoke: embedding+core runs on manifest dataset
**Files:**
- Create: `tests/test_cli_manifest_embedding_core_smoke.py`

### Task 73: Add docs: “Choosing core_* for embeddings” (ECOD vs LOF vs Mahalanobis)
**Files:**
- Create: `docs/CORE_SELECTION_ON_EMBEDDINGS.md`

### Task 74: Add `core_score_standardizer` as a first-class recommended post-step
**Files:**
- Modify: `docs/CORE_SELECTION_ON_EMBEDDINGS.md`
- Modify: `pyimgano/presets/industrial_classical.py`

### Task 75: Add `pyimgano-benchmark` preset `industrial-embedding-core-balanced`
**Files:**
- Modify: `pyimgano/cli_presets.py`
- Test: `tests/test_cli_presets_embedding_core_balanced.py`

---

## Phase 4 — Modern Deep Methods (Minimal, Contract‑Aligned) (Tasks 76–90)

### Task 76: Add a “reference-based” pixel anomaly map pipeline (query vs template)
**Why:** align with ReinAD/RAD-style inspection where a “golden” reference exists.
**Files:**
- Create: `pyimgano/pipelines/reference_map_pipeline.py`
- Test: `tests/test_reference_map_pipeline_smoke.py`

### Task 77: Add `vision_ref_patch_distance_map` (torchvision features + patch distance map)
**Files:**
- Create: `pyimgano/models/ref_patch_distance.py`
- Test: `tests/test_ref_patch_distance_map_detector.py`

### Task 78: Add `vision_ref_patch_distance_map` tiling support (high-res)
**Files:**
- Modify: `pyimgano/models/ref_patch_distance.py`
- Test: `tests/test_ref_patch_distance_tiling.py`

### Task 79: Add deep model “checkpoint required” capability enforcement in infer CLI
**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_requires_checkpoint_for_checkpoint_models.py`

### Task 80: Add a lightweight VQ-VAE reconstruction baseline (tiny industrial variant)
**Files:**
- Create: `pyimgano/models/vqvae.py`
- Modify: `pyimgano/models/__init__.py`
- Test: `tests/test_vqvae_smoke_tiny.py`

### Task 81: Add a “torch AE on embeddings” preset geared to stability
**Files:**
- Modify: `pyimgano/models/torch_autoencoder.py`
- Test: `tests/test_torch_autoencoder_embeddings_preset.py`

### Task 82: Add `vision_patchcore_online` (incremental memory update) (study-only to start)
**Files:**
- Create: `pyimgano/models/patchcore_online.py`
- Test: `tests/test_patchcore_online_smoke.py`

### Task 83: Add doc: “When to use reference-based pipelines”
**Files:**
- Create: `docs/RECIPES_REFERENCE_BASED_INSPECTION.md`

### Task 84: Add open_clip patch-level anomaly map (optional dep; no downloads by default)
**Files:**
- Create: `pyimgano/models/openclip_patch_map.py`
- Test: `tests/test_openclip_patch_map_optional_import.py`

### Task 85: Add `openclip_patch_map` defects export e2e with explicit “skip if deps missing”
**Files:**
- Create: `tests/test_openclip_patch_map_defects_e2e_optional.py`

### Task 86: Add “paper stub” docs for study-only LVLM methods (AgentIAD/AutoIAD)
**Files:**
- Create: `docs/STUDY_LVLM_ANOMALY_DETECTION.md`

### Task 87: Add a benchmark report schema for “reference-based” results
**Files:**
- Modify: `pyimgano/reporting/report_schema.py`
- Test: `tests/test_report_schema_reference_fields.py`

### Task 88: Add CLI UX: `pyimgano-infer --reference-dir` (for reference pipelines)
**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_reference_dir_flag.py`

### Task 89: Add docs: `pyimgano-infer` reference mode + examples
**Files:**
- Modify: `docs/INDUSTRIAL_INFERENCE.md`

### Task 90: Add “no-download” safety for diffusion extras if used (AnomalyAny-inspired)
**Files:**
- Create: `tests/test_no_diffusers_weight_downloads_by_default.py`

---

## Phase 5 — Synthesis / Augmentation v3 (Shift + Misalignment) (Tasks 91–96)

### Task 91: Add synthesis preset: illumination gradient / vignetting (MVTec AD2-like)
**Files:**
- Create: `pyimgano/synthesis/illumination.py`
- Modify: `pyimgano/synthesis/presets.py`
- Test: `tests/test_synthesis_illumination_preset.py`

### Task 92: Add synthesis preset: slight geometric warp (misalignment stress test)
**Files:**
- Create: `pyimgano/synthesis/warp.py`
- Modify: `pyimgano/synthesis/presets.py`
- Test: `tests/test_synthesis_warp_preset.py`

### Task 93: Add synthesis option: multiple defects per image + severity scalar
**Files:**
- Modify: `pyimgano/synthesis/synthesizer.py`
- Test: `tests/test_synthesis_multiple_defects.py`

### Task 94: Extend `pyimgano-synthesize` to output `meta.severity` and `meta.preset_id`
**Files:**
- Modify: `pyimgano/synthesize_cli.py`
- Test: `tests/test_synthesize_cli_meta_fields.py`

### Task 95: Add dataset wrapper: on-the-fly synthesis with severity curriculum
**Files:**
- Modify: `pyimgano/datasets/synthetic.py`
- Test: `tests/test_synthetic_dataset_wrapper_curriculum.py`

### Task 96: Add end-to-end smoke: synthesize (shift+warp) → pixel-first defects
**Files:**
- Create: `tests/test_e2e_synth_shift_warp_pixel_first_defects.py`

---

## Phase 6 — Final Verification + One Final Commit (Tasks 97–100)

### Task 97: Run full unit test suite (no coverage addopts)
**Run:**
- `pytest -q -o addopts=''`
**Expected:** all pass (skips allowed only for optional deps).

### Task 98: Run audits (public API / registry / score direction / third-party notices)
**Run:**
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
**Expected:** OK (heuristic WARNs reviewed and documented if needed).

### Task 99: Update changelog with concrete bullets for shipped v4 items
**Files:**
- Modify: `CHANGELOG.md`

### Task 100: One final commit (single commit policy)
**Run:**
- `git status --porcelain`
- `git add -A`
- `git commit -m "feat: industrial MVP loop v4 (datasets, pixel baselines, defects v2, embeddings+core, synthesis v3)"`
**Expected:** exactly one new commit.

