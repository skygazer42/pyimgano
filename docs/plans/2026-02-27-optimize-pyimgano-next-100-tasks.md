# PyImgAno Next 100-Task Optimization Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver another large, cohesive upgrade of PyImgAno across (1) more algorithms/models, (2) stronger functional/pipeline capabilities, and (3) richer industrial image enhancement + augmentation/synthesis — while keeping runtime dependencies stable and package size minimal (no new heavy core deps).

**Architecture:** Keep registry-driven API stable (`pyimgano.models.create_model/list_models/model_info`). Prefer adding/modernizing models by composing:
- `BaseDetector` for consistent scoring/thresholding/`predict()` semantics
- `FeatureExtractor` registry for pluggable feature extraction
- `vision_feature_pipeline` for “extractor + core detector” composition

**Tech Stack:** Python, NumPy, SciPy, scikit-learn, OpenCV, PyTorch, torchvision (already in deps). Optional integrations must be best-effort and guarded (no mandatory new deps).

**Commit policy (per user request):** Do **not** commit until all 100 tasks are completed; one final commit at the end.

**Review checkpoints:** Tasks marked **[REVIEW]** should be skimmed by the user before implementation (API surface, CLI changes, deprecations/aliases).

---

## Phase 0 — Industrial References + Guardrails (Tasks 1–10)

### Task 1: Web Research “Industrial AD” Repos/Papers + Build Reference Index
**Files:**
- Create: `docs/INDUSTRIAL_REFERENCE_PROJECTS.md`
**Notes:** Include short summaries + “what we borrow conceptually”. Code copy is allowed only when
license-compatible and notices are preserved.
**Verify:** `rg -n "GitHub|paper|license" docs/INDUSTRIAL_REFERENCE_PROJECTS.md | cat`

### Task 2: Add a Local-Only Reference Clone Helper (Shallow Clone)
**Files:**
- Create: `tools/clone_reference_repos.sh`
- Modify: `.gitignore` (ignore `/.cache/pyimgano_refs/`)
**Verify:** `bash tools/clone_reference_repos.sh --help`

### Task 3: Add/Verify Optional-Dependency Utilities (Centralized Error Messages)
**Files:**
- Verify/Modify: `pyimgano/utils/optional_deps.py`
- Test: `tests/test_optional_deps_utils.py`
**Verify:** `pytest -q -o addopts='' tests/test_optional_deps_utils.py`

### Task 4: Replace Ad-hoc Optional Imports To Use `optional_deps.require` (Targeted)
**Files:**
- Modify: `pyimgano/models/__init__.py`
- Modify: `pyimgano/models/winclip.py`
**Verify:** `pytest -q -o addopts='' tests/test_models_import_optional.py`

### Task 5: Add “No Network In Unit Tests” Guard For New Extractors/Synthesis
**Files:**
- Create: `tests/test_no_network_assumptions.py`
**Verify:** `pytest -q -o addopts='' tests/test_no_network_assumptions.py`

### Task 6: Add Stable Hash Utilities (For Caches + Determinism)
**Files:**
- Create: `pyimgano/utils/hash_utils.py`
- Test: `tests/test_hash_utils.py`
**Verify:** `pytest -q -o addopts='' tests/test_hash_utils.py`

### Task 7: Add Numpy-Array Feature Cache (Optional, Local Disk)
**Files:**
- Create: `pyimgano/cache/array_features.py`
- Test: `tests/test_array_feature_cache.py`
**Verify:** `pytest -q -o addopts='' tests/test_array_feature_cache.py`

### Task 8: Extend `BaseVisionDetector` To Cache Numpy Inputs (Best-effort)
**Files:**
- Modify: `pyimgano/models/baseml.py`
- Test: `tests/test_feature_cache_numpy_inputs.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_cache_numpy_inputs.py`

### Task 9: Add “Core/Feature” Capability Tags To Registry Metadata (Non-breaking)
**Files:**
- Modify: `pyimgano/models/registry.py`
- Test: `tests/test_model_info_payload.py`
**Verify:** `pytest -q -o addopts='' tests/test_model_info_payload.py`

### Task 10: Add Doc: “How Classical Pipelines Work” (Detector + Extractor + Cache)
**Files:**
- Create: `docs/ARCHITECTURE_CLASSICAL_PIPELINES.md`
**Verify:** `rg -n "BaseDetector|FeatureExtractor|vision_feature_pipeline" docs/ARCHITECTURE_CLASSICAL_PIPELINES.md | cat`

---

## Phase 1 — Register Core Models (Features-In → Scores-Out) (Tasks 11–35)

> Goal: expose more algorithms as `core_*` registry models that accept `np.ndarray` feature matrices directly, while keeping implementation centered on `BaseDetector` semantics.

### Task 11: Add Helper Base: `CoreFeatureDetector` (Thin `BaseDetector` Adapter)
**Files:**
- Create: `pyimgano/models/core_feature_base.py`
- Test: `tests/test_core_feature_base_contract.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_feature_base_contract.py`

### Task 12: Register `core_knn` (Wrap Existing `CoreKNN`)
**Files:**
- Modify: `pyimgano/models/knn.py`
- Test: `tests/test_core_knn_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_knn_model.py`

### Task 13: Register `core_iforest` (Wrap `CoreIForest`)
**Files:**
- Modify: `pyimgano/models/iforest.py`
- Test: `tests/test_core_iforest_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_iforest_model.py`

### Task 14: Register `core_ecod` (Wrap `CoreECOD`)
**Files:**
- Modify: `pyimgano/models/ecod.py`
- Test: `tests/test_core_ecod_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_ecod_model.py`

### Task 15: Register `core_copod` (Wrap `CoreCOPOD`)
**Files:**
- Modify: `pyimgano/models/copod.py`
- Test: `tests/test_core_copod_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_copod_model.py`

### Task 16: Register `core_pca` (Wrap the PCA Detector Core)
**Files:**
- Modify: `pyimgano/models/pca.py`
- Test: `tests/test_core_pca_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_pca_model.py`

### Task 17: Register `core_kde` (KernelDensity) (New Lightweight Core)
**Files:**
- Modify: `pyimgano/models/kde.py`
- Test: `tests/test_core_kde_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_kde_model.py`

### Task 18: Register `core_gmm` (Wrap `CoreGMM`)
**Files:**
- Modify: `pyimgano/models/gmm.py`
- Test: `tests/test_core_gmm_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_gmm_model.py`

### Task 19: Register `core_hbos` (Wrap `CoreHBOS`)
**Files:**
- Modify: `pyimgano/models/hbos.py`
- Test: `tests/test_core_hbos_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_hbos_model.py`

### Task 20: Register `core_mcd` (Wrap `CoreMCD`)
**Files:**
- Modify: `pyimgano/models/mcd.py`
- Test: `tests/test_core_mcd_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_mcd_model.py`

### Task 21: Register `core_ocsvm` (Wrap `CoreOCSVM`)
**Files:**
- Modify: `pyimgano/models/ocsvm.py`
- Test: `tests/test_core_ocsvm_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_ocsvm_model.py`

### Task 22: Register `core_abod` (Wrap `CoreABOD`)
**Files:**
- Modify: `pyimgano/models/abod.py`
- Test: `tests/test_core_abod_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_abod_model.py`

### Task 23: Register `core_cof` (Wrap `CoreCOF`)
**Files:**
- Modify: `pyimgano/models/cof.py`
- Test: `tests/test_core_cof_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_cof_model.py`

### Task 24: Register `core_loci` (Wrap `CoreLOCI`)
**Files:**
- Modify: `pyimgano/models/loci.py`
- Test: `tests/test_core_loci_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_loci_model.py`

### Task 25: Register `core_inne` (Wrap `CoreINNE`)
**Files:**
- Modify: `pyimgano/models/inne.py`
- Test: `tests/test_core_inne_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_inne_model.py`

### Task 26: Register `core_mad` (Wrap `CoreMAD`)
**Files:**
- Modify: `pyimgano/models/mad.py`
- Test: `tests/test_core_mad_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_mad_model.py`

### Task 27: Register `core_qmcd` (Wrap `CoreQMCD`)
**Files:**
- Modify: `pyimgano/models/qmcd.py`
- Test: `tests/test_core_qmcd_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_qmcd_model.py`

### Task 28: Register `core_sampling` (Wrap `CoreSampling`)
**Files:**
- Modify: `pyimgano/models/sampling.py`
- Test: `tests/test_core_sampling_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_sampling_model.py`

### Task 29: Register `core_feature_bagging` (Wrap `CoreFeatureBagging`)
**Files:**
- Modify: `pyimgano/models/feature_bagging.py`
- Test: `tests/test_core_feature_bagging_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_feature_bagging_model.py`

### Task 30: Register `core_suod` (Wrap `CoreSUOD`)
**Files:**
- Modify: `pyimgano/models/suod.py`
- Test: `tests/test_core_suod_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_suod_model.py`

### Task 31: Register `core_rgraph` (Wrap `CoreRGraph`)
**Files:**
- Modify: `pyimgano/models/rgraph.py`
- Test: `tests/test_core_rgraph_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_rgraph_model.py`

### Task 32: Register `core_loda` + `core_cblof` As Core Models (If Not Already)
**Files:**
- Modify: `pyimgano/models/loda.py`
- Modify: `pyimgano/models/cblof.py`
- Test: `tests/test_core_loda_cblof_models.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_loda_cblof_models.py`

### Task 33: Add CLI Filtering For “core” Models (tags)
**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_model_tag_filtering.py`
**Verify:** `pytest -q -o addopts='' tests/test_cli_model_tag_filtering.py`

### Task 34: Add Docs: “Classical (core_*) Models” Quick Reference
**Files:**
- Create: `docs/CORE_MODELS.md`
**Verify:** `rg -n \"core_knn|core_iforest\" docs/CORE_MODELS.md | cat`

### Task 35: Add One Unified Smoke Test For All `core_*` Classical Models
**Files:**
- Create: `tests/test_core_models_registry_smoke.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_models_registry_smoke.py`

---

## Phase 2 — Modernize Legacy “Structure/Template” Models (Tasks 36–55)

### Task 36: Add `StructuralFeaturesExtractor` (Modern, Deterministic)
**Files:**
- Create: `pyimgano/features/structural.py`
- Test: `tests/test_feature_structural.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_structural.py`

### Task 37: Add `core_lof` (Local Outlier Factor, novelty-mode)
**Files:**
- Create: `pyimgano/models/lof_core.py`
- Test: `tests/test_core_lof_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_lof_model.py`

### Task 38: Add `vision_lof` Wrapper (Extractor + `core_lof`)
**Files:**
- Create: `pyimgano/models/lof_native.py`
- Test: `tests/test_vision_lof_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_vision_lof_model.py`

### Task 39: **[REVIEW]** Replace `lof_structure` Implementation With `vision_lof` Alias (Keep Name Stable)
**Files:**
- Modify: `pyimgano/models/lof.py`
- Test: `tests/test_legacy_lof_structure_alias.py`
**Verify:** `pytest -q -o addopts='' tests/test_legacy_lof_structure_alias.py`

### Task 40: **[REVIEW]** Replace `isolation_forest_struct` With Structural Extractor + `vision_iforest` Core
**Files:**
- Modify: `pyimgano/models/Isolationforest.py`
- Test: `tests/test_legacy_isolation_forest_struct_alias.py`
**Verify:** `pytest -q -o addopts='' tests/test_legacy_isolation_forest_struct_alias.py`

### Task 41: Implement `core_kmeans` (Distance-to-Nearest-Centroid Score)
**Files:**
- Create: `pyimgano/models/kmeans_core.py`
- Test: `tests/test_core_kmeans_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_kmeans_model.py`

### Task 42: **[REVIEW]** Rebuild `kmeans_anomaly` Using `StructuralFeaturesExtractor` + `core_kmeans`
**Files:**
- Modify: `pyimgano/models/k_means.py`
- Test: `tests/test_legacy_kmeans_anomaly_alias.py`
**Verify:** `pytest -q -o addopts='' tests/test_legacy_kmeans_anomaly_alias.py`

### Task 43: Implement `core_dbscan` (Noise Probability / Distance Score)
**Files:**
- Create: `pyimgano/models/dbscan_core.py`
- Test: `tests/test_core_dbscan_model.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_dbscan_model.py`

### Task 44: **[REVIEW]** Rebuild `dbscan_anomaly` Using Structural Extractor + `core_dbscan`
**Files:**
- Modify: `pyimgano/models/dbscan.py`
- Test: `tests/test_legacy_dbscan_anomaly_alias.py`
**Verify:** `pytest -q -o addopts='' tests/test_legacy_dbscan_anomaly_alias.py`

### Task 45: Implement `core_ssim_template` (Template-Match Reconstruction Score)
**Files:**
- Create: `pyimgano/models/ssim_template_core.py`
- Test: `tests/test_core_ssim_template.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_ssim_template.py`

### Task 46: **[REVIEW]** Rebuild `ssim_template` Model On `core_ssim_template` (Keep Name Stable)
**Files:**
- Modify: `pyimgano/models/ssim.py`
- Test: `tests/test_legacy_ssim_template_alias.py`
**Verify:** `pytest -q -o addopts='' tests/test_legacy_ssim_template_alias.py`

### Task 47: Implement `core_ssim_struct` (SSIM + Structural Features)
**Files:**
- Create: `pyimgano/models/ssim_struct_core.py`
- Test: `tests/test_core_ssim_struct.py`
**Verify:** `pytest -q -o addopts='' tests/test_core_ssim_struct.py`

### Task 48: **[REVIEW]** Rebuild `ssim_struct` Model On `core_ssim_struct`
**Files:**
- Modify: `pyimgano/models/ssim_struct.py`
- Test: `tests/test_legacy_ssim_struct_alias.py`
**Verify:** `pytest -q -o addopts='' tests/test_legacy_ssim_struct_alias.py`

### Task 49: Add Compatibility Tests: Legacy Models Still Registered + Behave Like BaseDetector
**Files:**
- Create: `tests/test_legacy_models_contract.py`
**Verify:** `pytest -q -o addopts='' tests/test_legacy_models_contract.py`

### Task 50: Add Docs: “Structure/Template Baselines” (When To Use)
**Files:**
- Create: `docs/STRUCTURE_TEMPLATE_BASELINES.md`
**Verify:** `rg -n \"lof_structure|ssim_template\" docs/STRUCTURE_TEMPLATE_BASELINES.md | cat`

### Task 51: Update `docs/COMPARISON.md` To Mention New Core/Legacy Modernization
**Files:**
- Modify: `docs/COMPARISON.md`
**Verify:** `rg -n \"core_\" docs/COMPARISON.md | head`

### Task 52: Update `docs/ALGORITHM_SELECTION_GUIDE.md` (Add “structure/template” route)
**Files:**
- Modify: `docs/ALGORITHM_SELECTION_GUIDE.md`
**Verify:** `rg -n \"structure|template\" docs/ALGORITHM_SELECTION_GUIDE.md | head`

### Task 53: Add Benchmark Presets For Structure/Template Models
**Files:**
- Modify: `pyimgano/benchmark.py`
- Test: `tests/test_benchmark_presets.py`
**Verify:** `pytest -q -o addopts='' tests/test_benchmark_presets.py`

### Task 54: Ensure `tools/generate_model_index.py` Groups Core + Legacy Aliases Nicely
**Files:**
- Modify: `tools/generate_model_index.py`
**Verify:** `python tools/generate_model_index.py`

### Task 55: Regenerate `docs/MODEL_INDEX.md`
**Files:**
- Modify: `docs/MODEL_INDEX.md`
**Verify:** `rg -n \"core_knn|core_iforest|core_lof\" docs/MODEL_INDEX.md | head`

---

## Phase 3 — Torch Embedding Feature Extractors (Tasks 56–75)

### Task 56: Add `TorchvisionBackboneExtractor` (Global Pool Embeddings)
**Files:**
- Create: `pyimgano/features/torchvision_backbone.py`
- Test: `tests/test_feature_torchvision_backbone.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_torchvision_backbone.py`

### Task 57: Add `TorchvisionMultiLayerExtractor` (Intermediate Feature Concat)
**Files:**
- Create: `pyimgano/features/torchvision_multilayer.py`
- Test: `tests/test_feature_torchvision_multilayer.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_torchvision_multilayer.py`

### Task 58: Add `PatchGridExtractor` (Patch-Level Embeddings → Fixed Vector)
**Files:**
- Create: `pyimgano/features/patch_grid.py`
- Test: `tests/test_feature_patch_grid.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_patch_grid.py`

### Task 59: **[REVIEW]** Add Optional `OpenCLIPExtractor` (If `open_clip_torch` Installed)
**Files:**
- Create: `pyimgano/features/openclip_embed.py`
- Test: `tests/test_feature_openclip_optional.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_openclip_optional.py`

### Task 60: Register New Extractors + Tags (`embeddings`, `torch`, `deep-features`)
**Files:**
- Modify: `pyimgano/features/__init__.py`
- Modify: `pyimgano/features/registry.py`
- Test: `tests/test_feature_registry.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_registry.py`

### Task 61: Improve `pyimgano-features` CLI To Support Device/Batching
**Files:**
- Modify: `pyimgano/feature_cli.py`
- Test: `tests/test_feature_cli_smoke.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_cli_smoke.py`

### Task 62: Add “Embeddings + Classical” Tutorial
**Files:**
- Create: `docs/TUTORIAL_EMBEDDINGS_PLUS_CORE.md`
**Verify:** `rg -n \"TorchvisionBackboneExtractor|core_knn\" docs/TUTORIAL_EMBEDDINGS_PLUS_CORE.md | cat`

### Task 63: Add Example: Run `vision_feature_pipeline` With Torchvision Embeddings
**Files:**
- Create: `examples/torchvision_embeddings_classical_demo.py`
**Verify:** `python examples/torchvision_embeddings_classical_demo.py --help`

### Task 64: Add Workbench Config Export Coverage For New Extractor Specs
**Files:**
- Test: `tests/test_workbench_feature_pipeline.py`
**Verify:** `pytest -q -o addopts='' tests/test_workbench_feature_pipeline.py`

### Task 65: Add Feature Cache Coverage For Embedding Extractors (Paths + Numpy)
**Files:**
- Create: `tests/test_feature_cache_embeddings.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_cache_embeddings.py`

### Task 66: Add `core_kpca` / `core_pca_md` Docs Example With Embeddings
**Files:**
- Modify: `docs/TUTORIAL_CLASSICAL_ON_EMBEDDINGS.md`
**Verify:** `rg -n \"core_kpca|core_pca_md\" docs/TUTORIAL_CLASSICAL_ON_EMBEDDINGS.md | cat`

### Task 67: Add Benchmark Preset “embeddings-fast” (core + embeddings)
**Files:**
- Modify: `pyimgano/benchmark.py`
- Test: `tests/test_benchmark_presets.py`
**Verify:** `pytest -q -o addopts='' tests/test_benchmark_presets.py`

### Task 68: Add Deterministic Torch Inference Helpers For Extractors
**Files:**
- Create: `pyimgano/utils/torch_infer.py`
- Test: `tests/test_torch_infer_utils.py`
**Verify:** `pytest -q -o addopts='' tests/test_torch_infer_utils.py`

### Task 69: Use `torch_infer` Helpers In New Torch Extractors
**Files:**
- Modify: `pyimgano/features/torchvision_backbone.py`
- Modify: `pyimgano/features/torchvision_multilayer.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_torchvision_backbone.py`

### Task 70: Add `feature_info` Fields: Output Dim Hints When Available
**Files:**
- Modify: `pyimgano/features/registry.py`
- Test: `tests/test_feature_extractor_protocol.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_extractor_protocol.py`

### Task 71: Add Extractor-Level `get_params/set_params` For JSON Roundtrip
**Files:**
- Modify: `pyimgano/features/base.py`
- Test: `tests/test_feature_extractor_config_resolution.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_extractor_config_resolution.py`

### Task 72: Add `vision_feature_pipeline` Support For Passing Precomputed Feature Matrices **[REVIEW]**
**Files:**
- Modify: `pyimgano/models/feature_pipeline.py`
- Test: `tests/test_feature_pipeline_precomputed.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_pipeline_precomputed.py`

### Task 73: Add CLI `--input-mode features` For Benchmark When Model Supports It **[REVIEW]**
**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_cli_features_input_mode.py`
**Verify:** `pytest -q -o addopts='' tests/test_cli_features_input_mode.py`

### Task 74: Add Docs For `--input-mode features`
**Files:**
- Modify: `docs/CLI_REFERENCE.md`
**Verify:** `rg -n \"input-mode\\s+features\" docs/CLI_REFERENCE.md | cat`

### Task 75: Add Robustness Benchmark Coverage For `features` Input Mode
**Files:**
- Modify: `pyimgano/robustness/benchmark.py`
- Test: `tests/test_robustness_benchmark.py`
**Verify:** `pytest -q -o addopts='' tests/test_robustness_benchmark.py`

---

## Phase 4 — Industrial Anomaly Synthesis + Augmentations (Tasks 76–95)

### Task 76: Add `pyimgano/synthesis` Package Skeleton
**Files:**
- Create: `pyimgano/synthesis/__init__.py`
- Test: `tests/test_synthesis_import.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_import.py`

### Task 77: Implement 2D Perlin Noise (No Extra Deps)
**Files:**
- Create: `pyimgano/synthesis/perlin.py`
- Test: `tests/test_synthesis_perlin.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_perlin.py`

### Task 78: Implement Mask Primitives (Blobs / Ellipses / Scratches)
**Files:**
- Create: `pyimgano/synthesis/masks.py`
- Test: `tests/test_synthesis_masks.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_masks.py`

### Task 79: Implement Blend Ops (Alpha + Poisson via OpenCV `seamlessClone`)
**Files:**
- Create: `pyimgano/synthesis/blend.py`
- Test: `tests/test_synthesis_blend.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_blend.py`

### Task 80: Implement CutPaste Variants (normal / scar / 3-way)
**Files:**
- Create: `pyimgano/synthesis/cutpaste.py`
- Test: `tests/test_synthesis_cutpaste.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_cutpaste.py`

### Task 81: Implement `AnomalySynthesizer` (Configurable Pipeline)
**Files:**
- Create: `pyimgano/synthesis/synthesizer.py`
- Test: `tests/test_synthesis_synthesizer.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_synthesizer.py`

### Task 82: Add Industrial Synthesis Presets (scratch/stain/pit/glare)
**Files:**
- Create: `pyimgano/synthesis/presets.py`
- Test: `tests/test_synthesis_presets.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_presets.py`

### Task 83: Add Deterministic Seeding For Synthesis Pipeline
**Files:**
- Modify: `pyimgano/synthesis/synthesizer.py`
- Test: `tests/test_synthesis_determinism.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_determinism.py`

### Task 84: Add Dataset Wrapper: Generate Synthetic Anomalies On-The-Fly
**Files:**
- Create: `pyimgano/datasets/synthetic.py`
- Test: `tests/test_dataset_synthetic_wrapper.py`
**Verify:** `pytest -q -o addopts='' tests/test_dataset_synthetic_wrapper.py`

### Task 85: **[REVIEW]** Add CLI `pyimgano-synthesize` (Write Images + Masks + Manifest)
**Files:**
- Create: `pyimgano/synthesize_cli.py`
- Modify: `pyproject.toml` (add script entry)
- Test: `tests/test_synthesize_cli_smoke.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesize_cli_smoke.py`

### Task 86: Add Docs: Synthetic Anomaly Generation Cookbook
**Files:**
- Create: `docs/SYNTHETIC_ANOMALY_GENERATION.md`
**Verify:** `rg -n \"CutPaste|Perlin|seamlessClone\" docs/SYNTHETIC_ANOMALY_GENERATION.md | cat`

### Task 87: Add Example: Generate A Tiny Synthetic Dataset (For CI/Debug)
**Files:**
- Create: `examples/synthesis_generate_dataset_demo.py`
**Verify:** `python examples/synthesis_generate_dataset_demo.py --help`

### Task 88: Add Robustness Integration: Treat Synthesis Preset As A Corruption Source
**Files:**
- Modify: `pyimgano/robustness/types.py`
- Modify: `pyimgano/robustness/benchmark.py`
- Test: `tests/test_robustness_synthesis_corruption.py`
**Verify:** `pytest -q -o addopts='' tests/test_robustness_synthesis_corruption.py`

### Task 89: Add Industrial Recipe That Enables Synthesis During Training (Best-effort)
**Files:**
- Modify: `pyimgano/recipes/builtin/__init__.py`
- Modify: `pyimgano/recipes/builtin/classical_recipes.py`
- Test: `tests/test_recipes_smoke.py`
**Verify:** `pytest -q -o addopts='' tests/test_recipes_smoke.py`

### Task 90: Add Preprocessing Bridge: Synthesis Outputs Respect ROI Masks (If Provided)
**Files:**
- Modify: `pyimgano/synthesis/synthesizer.py`
- Test: `tests/test_synthesis_roi_mask.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_roi_mask.py`

### Task 91: Add “Industrial Surface” Augmentations (Vibration Blur / Stripe Noise / Dust)
**Files:**
- Create: `pyimgano/synthesis/industrial_noise.py`
- Test: `tests/test_synthesis_industrial_noise.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_industrial_noise.py`

### Task 92: Add Preview Utility: Grid Visualization For Synth + Masks (Matplotlib Optional)
**Files:**
- Create: `pyimgano/synthesis/preview.py`
- Test: `tests/test_synthesis_preview.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_preview.py`

### Task 93: Update `docs/INDUSTRIAL_PREPROCESSING_COOKBOOK.md` With Synthesis Section
**Files:**
- Modify: `docs/INDUSTRIAL_PREPROCESSING_COOKBOOK.md`
**Verify:** `rg -n \"synthesis\" docs/INDUSTRIAL_PREPROCESSING_COOKBOOK.md | cat`

### Task 94: Add Type Exports For Synthesis Config Objects
**Files:**
- Modify: `pyimgano/__init__.py`
- Test: `tests/test_tools_audit_public_api.py`
**Verify:** `pytest -q -o addopts='' tests/test_tools_audit_public_api.py`

### Task 95: Add End-to-End Smoke: Train/Infer On Tiny Synthetic Manifest Dataset
**Files:**
- Create: `tests/test_synthesis_e2e_manifest_smoke.py`
**Verify:** `pytest -q -o addopts='' tests/test_synthesis_e2e_manifest_smoke.py`

---

## Phase 5 — Industrial Enhancement Additions + Final Release Hygiene (Tasks 96–100)

### Task 96: Add Retinex Illumination Normalization (MSRCR-lite)
**Files:**
- Create: `pyimgano/preprocessing/retinex.py`
- Modify: `pyimgano/preprocessing/__init__.py`
- Test: `tests/test_retinex.py`
**Verify:** `pytest -q -o addopts='' tests/test_retinex.py`

### Task 97: Update `pyimgano/preprocessing/industrial_presets.py` With Retinex Preset
**Files:**
- Modify: `pyimgano/preprocessing/industrial_presets.py`
- Test: `tests/test_industrial_presets.py`
**Verify:** `pytest -q -o addopts='' tests/test_industrial_presets.py`

### Task 98: Run Full Test Suite + Audits
**Verify:**
- `pytest -q -o addopts=''`
- `python tools/audit_public_api.py && python tools/audit_registry.py`

### Task 99: Update `CHANGELOG.md` With “Next 100 Tasks” Summary
**Files:**
- Modify: `CHANGELOG.md`
**Verify:** `rg -n \"Next 100\" CHANGELOG.md | cat`

### Task 100: Single Final Commit (All Changes)
**Command:**
```bash
git status -sb
git add -A
git commit -m "feat: next 100-task optimization (core models, embeddings, synthesis)"
```
