# PyImgAno 100-Task Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a large, cohesive upgrade of PyImgAno across (1) algorithms/models, (2) functional utilities/pipelines, and (3) image enhancement & feature extraction, while keeping runtime dependencies stable and package size minimal.

**Architecture:** Keep the existing registry as the stable user-facing API (`pyimgano.models.create_model/list_models/model_info`). Add new algorithms as native implementations that conform to `BaseDetector` / `BaseDeepLearningDetector`. Expand capabilities primarily via lightweight NumPy/SciPy/sklearn/OpenCV/Torch code (no new heavy deps). Improve correctness, determinism, and performance with targeted utilities + tests.

**Tech Stack:** Python, NumPy, SciPy, scikit-learn, scikit-image, OpenCV, PyTorch.

**Commit policy (per user request):** Do **not** commit until all 100 tasks are completed; one final commit at the end.

---

## Phase 0 — Hygiene & PyOD Text Cleanup (Tasks 1–10)

### Task 1: Replace PyOD Example In `docs/cblof.md` With Native `vision_cblof`
**Files:**
- Modify: `docs/cblof.md`
**Verify:** `python -m py_compile pyimgano/models/cblof.py`

### Task 2: Replace PyOD Example In `docs/loda.md` With Native `core_loda`
**Files:**
- Modify: `docs/loda.md`
**Verify:** `python -m py_compile pyimgano/models/loda.py`

### Task 3: Remove Remaining Misleading “pyod” References In Runtime Comments
**Files:**
- Modify: `pyimgano/models/baseCv.py`
**Verify:** `pytest -q -o addopts='' tests/test_no_pyod_imports.py`

### Task 4: Audit `docs/` For “pyod” Mentions In Algorithm Pages; Replace With `pyimgano` Examples
**Files:**
- Modify: `docs/*.md` (algorithm pages only; keep migration/changelog references)
**Verify:** `rg -n "\\bpyod\\b" docs | cat`

### Task 5: Tighten “No PyOD Imports” Guardrails For New Code (Already In Place)
**Files:**
- Verify only: `tests/test_no_pyod_imports.py`
**Verify:** `pytest -q -o addopts='' tests/test_no_pyod_imports.py`

### Task 6: Standardize Logging (No `print()` In Library Code) For Classical Detectors
**Files:**
- Modify: `pyimgano/models/*.py` (targeted set; start with CBLOF/LODA)
**Verify:** `pytest -q -o addopts='' tests/test_models_import_optional.py`

### Task 7: Add `pyimgano/utils/logging.py` Helper (Project Logger + Verbose Mapping)
**Files:**
- Create: `pyimgano/utils/logging.py`
- Test: `tests/test_logging_utils.py`
**Verify:** `pytest -q -o addopts='' tests/test_logging_utils.py`

### Task 8: Add Deterministic Random-State Utility
**Files:**
- Create: `pyimgano/utils/random_state.py`
- Test: `tests/test_random_state_utils.py`
**Verify:** `pytest -q -o addopts='' tests/test_random_state_utils.py`

### Task 9: Add Score-Normalization Utility (zscore/minmax/rank/quantile)
**Files:**
- Create: `pyimgano/utils/score_normalization.py`
- Test: `tests/test_score_normalization.py`
**Verify:** `pytest -q -o addopts='' tests/test_score_normalization.py`

### Task 10: Add `pyimgano/utils/typing.py` Helpers (NDArray aliases, Input types)
**Files:**
- Create: `pyimgano/utils/typing.py`
- Test: `tests/test_typing_utils_import.py`
**Verify:** `pytest -q -o addopts='' tests/test_typing_utils_import.py`

---

## Phase 1 — Core Detector Contract & Utilities (Tasks 11–30)

### Task 11: Add `BaseDetector.fit_predict` Convenience
**Files:**
- Modify: `pyimgano/models/base_detector.py`
- Test: `tests/test_base_detector_fit_predict.py`
**Verify:** `pytest -q -o addopts='' tests/test_base_detector_fit_predict.py`

### Task 12: Add `BaseDetector.score_samples` Alias To `decision_function`
**Files:**
- Modify: `pyimgano/models/base_detector.py`
- Test: `tests/test_base_detector_score_samples.py`
**Verify:** `pytest -q -o addopts='' tests/test_base_detector_score_samples.py`

### Task 13: Add `BaseDetector.get_params/set_params` (sklearn-compatible)
**Files:**
- Modify: `pyimgano/models/base_detector.py`
- Test: `tests/test_base_detector_params.py`
**Verify:** `pytest -q -o addopts='' tests/test_base_detector_params.py`

### Task 14: Add `pyimgano/models/mixins.py` For Shared Validation (contamination, n_jobs, etc.)
**Files:**
- Create: `pyimgano/models/mixins.py`
- Test: `tests/test_model_mixins.py`
**Verify:** `pytest -q -o addopts='' tests/test_model_mixins.py`

### Task 15: Refactor `CoreCBLOF` To Inherit `BaseDetector` (Remove Duplicate Threshold Logic)
**Files:**
- Modify: `pyimgano/models/cblof.py`
- Test: `tests/test_cblof_native_contract.py`
**Verify:** `pytest -q -o addopts='' tests/test_cblof_native_contract.py`

### Task 16: Refactor `CoreLODA` To Inherit `BaseDetector` (Remove Duplicate Threshold Logic)
**Files:**
- Modify: `pyimgano/models/loda.py`
- Test: `tests/test_loda_native_contract.py`
**Verify:** `pytest -q -o addopts='' tests/test_loda_native_contract.py`

### Task 17: Remove `check_parameter` Duplicates In Legacy Classical Detectors (Use `utils/param_check.py`)
**Files:**
- Modify: `pyimgano/models/cblof.py`
- Modify: `pyimgano/models/loda.py`
- Test: `tests/test_param_check.py`
**Verify:** `pytest -q -o addopts='' tests/test_param_check.py`

### Task 18: Add Unified “Feature Extractor” Protocol + Runtime Validation
**Files:**
- Create: `pyimgano/features/protocols.py`
- Modify: `pyimgano/models/baseml.py`
- Test: `tests/test_feature_extractor_protocol.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_extractor_protocol.py`

### Task 19: Add `pyimgano/features/__init__.py` Public Surface
**Files:**
- Create: `pyimgano/features/__init__.py`
- Test: `tests/test_features_import.py`
**Verify:** `pytest -q -o addopts='' tests/test_features_import.py`

### Task 20: Add `FeatureExtractor` Base Class With Optional Fit/Transform Semantics
**Files:**
- Create: `pyimgano/features/base.py`
- Test: `tests/test_feature_extractor_base.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_extractor_base.py`

### Task 21: Add “IdentityExtractor” To `pyimgano/features` (Canonical)
**Files:**
- Create: `pyimgano/features/identity.py`
- Modify: `pyimgano/detectors/__init__.py` (reuse canonical IdentityExtractor)
- Test: `tests/test_identity_extractor.py`
**Verify:** `pytest -q -o addopts='' tests/test_identity_extractor.py`

### Task 22: Add Registry Support For Feature Extractors (List + Create)
**Files:**
- Create: `pyimgano/features/registry.py`
- Test: `tests/test_feature_registry.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_registry.py`

### Task 23: Add CLI Flags To List Feature Extractors (Non-breaking)
**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke.py`
**Verify:** `pytest -q -o addopts='' tests/test_cli_smoke.py`

### Task 24: Add Optional Disk Cache For Extractor Outputs (Paths Input)
**Files:**
- Create: `pyimgano/cache/feature_vectors.py`
- Modify: `pyimgano/models/baseml.py`
- Test: `tests/test_feature_cache_vectors.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_cache_vectors.py`

### Task 25: Add “Batching” Support For Extractors (Large Dataset Throughput)
**Files:**
- Modify: `pyimgano/features/base.py`
- Test: `tests/test_feature_extractor_batching.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_extractor_batching.py`

### Task 26: Add `pyimgano/utils/parallel.py` (Joblib Threading Helper)
**Files:**
- Create: `pyimgano/utils/parallel.py`
- Test: `tests/test_parallel_utils.py`
**Verify:** `pytest -q -o addopts='' tests/test_parallel_utils.py`

### Task 27: Speed Up `pyimgano/models/__init__.py` Auto-Import With Known-Optional Modules
**Files:**
- Modify: `pyimgano/models/__init__.py`
- Test: `tests/test_models_import_optional.py`
**Verify:** `pytest -q -o addopts='' tests/test_models_import_optional.py`

### Task 28: Add `model_info` Fields For Input Modes (paths/numpy/features)
**Files:**
- Modify: `pyimgano/models/registry.py`
- Modify: `pyimgano/models/capabilities.py`
- Test: `tests/test_model_info_payload.py`
**Verify:** `pytest -q -o addopts='' tests/test_model_info_payload.py`

### Task 29: Add Score Calibration Helpers (Unsupervised Rank Calibration)
**Files:**
- Create: `pyimgano/calibration/rank_calibration.py`
- Test: `tests/test_rank_calibration.py`
**Verify:** `pytest -q -o addopts='' tests/test_rank_calibration.py`

### Task 30: Add Score-Ensemble Weighting Modes (mean/max/trimmed-mean)
**Files:**
- Modify: `pyimgano/models/score_ensemble.py`
- Test: `tests/test_score_ensemble.py`
**Verify:** `pytest -q -o addopts='' tests/test_score_ensemble.py`

---

## Phase 2 — Add Lightweight Classical Detectors (Tasks 31–60)

### Task 31: Implement LoOP Core (`core_loop`) (Local Outlier Probability)
**Files:**
- Create: `pyimgano/models/loop.py`
- Test: `tests/test_loop_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_loop_detector.py`

### Task 32: Add Vision Wrapper (`vision_loop`) Using Feature Extractor
**Files:**
- Modify: `pyimgano/models/loop.py`
- Test: `tests/test_loop_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_loop_detector.py`

### Task 33: Implement LDOF Core (`core_ldof`) (Local Distance-based Outlier Factor)
**Files:**
- Create: `pyimgano/models/ldof.py`
- Test: `tests/test_ldof_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_ldof_detector.py`

### Task 34: Add Vision Wrapper (`vision_ldof`)
**Files:**
- Modify: `pyimgano/models/ldof.py`
- Test: `tests/test_ldof_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_ldof_detector.py`

### Task 35: Implement ODIN Core (`core_odin`) (kNN indegree)
**Files:**
- Create: `pyimgano/models/odin.py`
- Test: `tests/test_odin_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_odin_detector.py`

### Task 36: Add Vision Wrapper (`vision_odin`)
**Files:**
- Modify: `pyimgano/models/odin.py`
- Test: `tests/test_odin_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_odin_detector.py`

### Task 37: Implement RRCF Core (`core_rrcf`) (Robust Random Cut Forest)
**Files:**
- Create: `pyimgano/models/rrcf.py`
- Test: `tests/test_rrcf_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_rrcf_detector.py`

### Task 38: Add Vision Wrapper (`vision_rrcf`)
**Files:**
- Modify: `pyimgano/models/rrcf.py`
- Test: `tests/test_rrcf_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_rrcf_detector.py`

### Task 39: Implement Half-Space Trees Core (`core_hst`)
**Files:**
- Create: `pyimgano/models/hst.py`
- Test: `tests/test_hst_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_hst_detector.py`

### Task 40: Add Vision Wrapper (`vision_hst`)
**Files:**
- Modify: `pyimgano/models/hst.py`
- Test: `tests/test_hst_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_hst_detector.py`

### Task 41: Add Lightweight “Mahalanobis” Detector (`core_mahalanobis`) (Mean/Cov)
**Files:**
- Create: `pyimgano/models/mahalanobis.py`
- Test: `tests/test_mahalanobis_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_mahalanobis_detector.py`

### Task 42: Add Vision Wrapper (`vision_mahalanobis`)
**Files:**
- Modify: `pyimgano/models/mahalanobis.py`
- Test: `tests/test_mahalanobis_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_mahalanobis_detector.py`

### Task 43: Add kNN “Distance-to-Centroid” Detector (`core_dtc`)
**Files:**
- Create: `pyimgano/models/dtc.py`
- Test: `tests/test_dtc_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_dtc_detector.py`

### Task 44: Add Vision Wrapper (`vision_dtc`)
**Files:**
- Modify: `pyimgano/models/dtc.py`
- Test: `tests/test_dtc_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_dtc_detector.py`

### Task 45: Add “Robust Z-Score” Detector (`core_rzscore`) (Median/MAD)
**Files:**
- Create: `pyimgano/models/rzscore.py`
- Test: `tests/test_rzscore_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_rzscore_detector.py`

### Task 46: Add Vision Wrapper (`vision_rzscore`)
**Files:**
- Modify: `pyimgano/models/rzscore.py`
- Test: `tests/test_rzscore_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_rzscore_detector.py`

### Task 47: Add “POT Thresholding” Utility (Extreme Value) For Score -> Label
**Files:**
- Create: `pyimgano/calibration/pot_threshold.py`
- Test: `tests/test_pot_threshold.py`
**Verify:** `pytest -q -o addopts='' tests/test_pot_threshold.py`

### Task 48: Add Optional POT Threshold Mode To BaseDetector
**Files:**
- Modify: `pyimgano/models/base_detector.py`
- Test: `tests/test_base_detector_pot_threshold.py`
**Verify:** `pytest -q -o addopts='' tests/test_base_detector_pot_threshold.py`

### Task 49: Add “kNN Graph Degree” Detector (`core_knn_degree`)
**Files:**
- Create: `pyimgano/models/knn_degree.py`
- Test: `tests/test_knn_degree_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_knn_degree_detector.py`

### Task 50: Add Vision Wrapper (`vision_knn_degree`)
**Files:**
- Modify: `pyimgano/models/knn_degree.py`
- Test: `tests/test_knn_degree_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_knn_degree_detector.py`

### Task 51: Add “Distance Correlation” Detector (`core_dcorr`) (Feature Independence)
**Files:**
- Create: `pyimgano/models/dcorr.py`
- Test: `tests/test_dcorr_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_dcorr_detector.py`

### Task 52: Add Vision Wrapper (`vision_dcorr`)
**Files:**
- Modify: `pyimgano/models/dcorr.py`
- Test: `tests/test_dcorr_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_dcorr_detector.py`

### Task 53: Add Simple “PCA Mahalanobis” Detector (`core_pca_md`)
**Files:**
- Create: `pyimgano/models/pca_md.py`
- Test: `tests/test_pca_md_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_pca_md_detector.py`

### Task 54: Add Vision Wrapper (`vision_pca_md`)
**Files:**
- Modify: `pyimgano/models/pca_md.py`
- Test: `tests/test_pca_md_detector.py`
**Verify:** `pytest -q -o addopts='' tests/test_pca_md_detector.py`

### Task 55: Add `vision_*` Registration Tags For New Classical Detectors
**Files:**
- Modify: `pyimgano/models/*.py` (new detectors)
**Verify:** `python tools/generate_model_index.py`

### Task 56: Add Smoke Tests For New Detectors In CLI `--list-models`
**Files:**
- Modify: `tests/test_more_models_added.py`
**Verify:** `pytest -q -o addopts='' tests/test_more_models_added.py`

### Task 57: Add `docs/<algo>.md` Pages For New Detectors (LoOP/LDOF/ODIN/RRCF/HST)
**Files:**
- Create: `docs/loop.md`
- Create: `docs/ldof.md`
- Create: `docs/odin.md`
- Create: `docs/rrcf.md`
- Create: `docs/hst.md`
**Verify:** `python -m py_compile pyimgano/models/loop.py`

### Task 58: Update `docs/ALGORITHM_SELECTION_GUIDE.md` With New Baselines
**Files:**
- Modify: `docs/ALGORITHM_SELECTION_GUIDE.md`
**Verify:** `rg -n \"vision_loop|vision_ldof|vision_rrcf\" docs/ALGORITHM_SELECTION_GUIDE.md`

### Task 59: Add Benchmark Presets For New Classical Detectors
**Files:**
- Modify: `pyimgano/benchmark.py`
- Test: `tests/test_benchmark_workflows.py`
**Verify:** `pytest -q -o addopts='' tests/test_benchmark_workflows.py`

### Task 60: Add Robustness/Corruption Benchmarks For Classical Detectors (Paths->Numpy)
**Files:**
- Modify: `pyimgano/robustness/*`
- Test: `tests/test_robust_benchmark.py`
**Verify:** `pytest -q -o addopts='' tests/test_robust_benchmark.py`

---

## Phase 3 — Feature Extractors Zoo (Tasks 61–80)

### Task 61: Add `HOGExtractor`
**Files:**
- Create: `pyimgano/features/hog.py`
- Test: `tests/test_feature_hog.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_hog.py`

### Task 62: Add `LBPExtractor`
**Files:**
- Create: `pyimgano/features/lbp.py`
- Test: `tests/test_feature_lbp.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_lbp.py`

### Task 63: Add `GaborBankExtractor`
**Files:**
- Create: `pyimgano/features/gabor.py`
- Test: `tests/test_feature_gabor.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_gabor.py`

### Task 64: Add `ColorHistogramExtractor` (HSV/LAB)
**Files:**
- Create: `pyimgano/features/color_hist.py`
- Test: `tests/test_feature_color_hist.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_color_hist.py`

### Task 65: Add `EdgeStatsExtractor` (Canny/Sobel stats)
**Files:**
- Create: `pyimgano/features/edge_stats.py`
- Test: `tests/test_feature_edge_stats.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_edge_stats.py`

### Task 66: Add `FFTLowFreqExtractor`
**Files:**
- Create: `pyimgano/features/fft_lowfreq.py`
- Test: `tests/test_feature_fft_lowfreq.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_fft_lowfreq.py`

### Task 67: Add `PatchStatsExtractor` (Mean/Std/Skew/Kurt per patch grid)
**Files:**
- Create: `pyimgano/features/patch_stats.py`
- Test: `tests/test_feature_patch_stats.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_patch_stats.py`

### Task 68: Add `MultiExtractor` (Concat multiple extractors)
**Files:**
- Create: `pyimgano/features/multi.py`
- Test: `tests/test_feature_multi.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_multi.py`

### Task 69: Add `PCAProjector` Extractor (Fit on train, transform test)
**Files:**
- Create: `pyimgano/features/pca_projector.py`
- Test: `tests/test_feature_pca_projector.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_pca_projector.py`

### Task 70: Add `StandardScalerExtractor` (Fit/Transform)
**Files:**
- Create: `pyimgano/features/scaler.py`
- Test: `tests/test_feature_scaler.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_scaler.py`

### Task 71: Register All Extractors In Feature Registry
**Files:**
- Modify: `pyimgano/features/__init__.py`
- Modify: `pyimgano/features/registry.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_registry.py`

### Task 72: Add Docs Page `docs/FEATURE_EXTRACTORS.md`
**Files:**
- Create: `docs/FEATURE_EXTRACTORS.md`
**Verify:** `rg -n \"HOGExtractor\" docs/FEATURE_EXTRACTORS.md`

### Task 73: Add Example Script `examples/feature_extractors_demo.py`
**Files:**
- Create: `examples/feature_extractors_demo.py`
**Verify:** `python examples/feature_extractors_demo.py --help`

### Task 74: Add Feature Extractor Smoke Test With Synthetic Images
**Files:**
- Create: `tests/test_features_smoke_images.py`
**Verify:** `pytest -q -o addopts='' tests/test_features_smoke_images.py`

### Task 75: Add `vision_*` Preset Recipes Using New Extractors + Classical Detectors
**Files:**
- Modify: `pyimgano/recipes/*`
- Test: `tests/test_recipes_smoke.py`
**Verify:** `pytest -q -o addopts='' tests/test_recipes_smoke.py`

### Task 76: Add `VisionFeaturePipeline` (Extractor + Detector) Helper
**Files:**
- Create: `pyimgano/pipelines/feature_pipeline.py`
- Test: `tests/test_feature_pipeline.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_pipeline.py`

### Task 77: Add Exportable JSON Config For Feature Pipelines (Workbench Integration)
**Files:**
- Modify: `pyimgano/workbench/*`
- Test: `tests/test_workbench_feature_pipeline.py`
**Verify:** `pytest -q -o addopts='' tests/test_workbench_feature_pipeline.py`

### Task 78: Add Minimal Tutorial In Docs For “Classical On Embeddings”
**Files:**
- Create: `docs/TUTORIAL_CLASSICAL_ON_EMBEDDINGS.md`
**Verify:** `rg -n \"vision_knn\" docs/TUTORIAL_CLASSICAL_ON_EMBEDDINGS.md`

### Task 79: Add “Feature Precompute” CLI For Large Datasets
**Files:**
- Create: `pyimgano/feature_cli.py`
- Modify: `pyproject.toml` (add script entry)
- Test: `tests/test_feature_cli_smoke.py`
**Verify:** `pytest -q -o addopts='' tests/test_feature_cli_smoke.py`

### Task 80: Add `pyimgano.features` Public API Exports In `pyimgano/__init__.py`
**Files:**
- Modify: `pyimgano/__init__.py`
- Test: `tests/test_tools_audit_public_api.py`
**Verify:** `pytest -q -o addopts='' tests/test_tools_audit_public_api.py`

---

## Phase 4 — Image Enhancement & Industrial Preprocessing (Tasks 81–95)

### Task 81: Vectorize `frequency_filter` (Remove O(HW) Python Loops)
**Files:**
- Modify: `pyimgano/preprocessing/advanced_operations.py`
- Test: `tests/test_frequency_filter_vectorized.py`
**Verify:** `pytest -q -o addopts='' tests/test_frequency_filter_vectorized.py`

### Task 82: Add Guided Filter (NumPy/OpenCV box-filter implementation)
**Files:**
- Create: `pyimgano/preprocessing/guided_filter.py`
- Test: `tests/test_guided_filter.py`
**Verify:** `pytest -q -o addopts='' tests/test_guided_filter.py`

### Task 83: Add Perona–Malik Anisotropic Diffusion
**Files:**
- Create: `pyimgano/preprocessing/anisotropic_diffusion.py`
- Test: `tests/test_anisotropic_diffusion.py`
**Verify:** `pytest -q -o addopts='' tests/test_anisotropic_diffusion.py`

### Task 84: Add Rolling-Ball Background Subtraction (Industrial Surfaces)
**Files:**
- Create: `pyimgano/preprocessing/background.py`
- Test: `tests/test_background_subtraction.py`
**Verify:** `pytest -q -o addopts='' tests/test_background_subtraction.py`

### Task 85: Add “Shading Correction” Preset (Background + CLAHE)
**Files:**
- Modify: `pyimgano/preprocessing/industrial_presets.py`
- Test: `tests/test_industrial_presets.py`
**Verify:** `pytest -q -o addopts='' tests/test_industrial_presets.py`

### Task 86: Add Fast “Local Contrast Normalization” (LCN) For Grayscale
**Files:**
- Modify: `pyimgano/preprocessing/enhancer.py`
- Test: `tests/test_local_contrast_normalization.py`
**Verify:** `pytest -q -o addopts='' tests/test_local_contrast_normalization.py`

### Task 87: Add Torch GPU Gaussian Blur (Fallback to CPU)
**Files:**
- Modify: `pyimgano/preprocessing/enhancer.py`
- Test: `tests/test_torch_gaussian_blur.py`
**Verify:** `pytest -q -o addopts='' tests/test_torch_gaussian_blur.py`

### Task 88: Add “Defect Amplification” Morphology Preset (Tophat + Edge)
**Files:**
- Modify: `pyimgano/preprocessing/industrial_presets.py`
- Test: `tests/test_industrial_presets.py`
**Verify:** `pytest -q -o addopts='' tests/test_industrial_presets.py`

### Task 89: Add JPEG Artifact Robust Preprocess (Denoise + Deblock Approx)
**Files:**
- Modify: `pyimgano/preprocessing/enhancer.py`
- Test: `tests/test_jpeg_robust_preprocess.py`
**Verify:** `pytest -q -o addopts='' tests/test_jpeg_robust_preprocess.py`

### Task 90: Add `ImagePreprocessor` Support For OpenCV Backend (Optional)
**Files:**
- Modify: `pyimgano/utils/image_ops.py`
- Test: `tests/test_image_ops_cv_backend.py`
**Verify:** `pytest -q -o addopts='' tests/test_image_ops_cv_backend.py`

### Task 91: Add “Tile + Blend” Helper For High-Res Industrial Images
**Files:**
- Create: `pyimgano/preprocessing/tiling.py`
- Test: `tests/test_tiling_blend.py`
**Verify:** `pytest -q -o addopts='' tests/test_tiling_blend.py`

### Task 92: Add “Mask-Aware Enhancement” (Only Enhance ROI)
**Files:**
- Modify: `pyimgano/preprocessing/mixin.py`
- Test: `tests/test_mask_aware_enhancement.py`
**Verify:** `pytest -q -o addopts='' tests/test_mask_aware_enhancement.py`

### Task 93: Add `docs/INDUSTRIAL_PREPROCESSING_COOKBOOK.md`
**Files:**
- Create: `docs/INDUSTRIAL_PREPROCESSING_COOKBOOK.md`
**Verify:** `rg -n \"shading\" docs/INDUSTRIAL_PREPROCESSING_COOKBOOK.md`

### Task 94: Add Robustness Tests For Preprocessing Ops (Shapes/Dtypes/Finite)
**Files:**
- Create: `tests/test_preprocessing_robustness.py`
**Verify:** `pytest -q -o addopts='' tests/test_preprocessing_robustness.py`

### Task 95: Add `pyimgano.preprocessing` Public API Exports For New Ops
**Files:**
- Modify: `pyimgano/preprocessing/__init__.py`
**Verify:** `python -c \"from pyimgano.preprocessing import *; print('ok')\"`

---

## Phase 5 — Final Integration & Release Hygiene (Tasks 96–100)

### Task 96: Regenerate `docs/MODEL_INDEX.md` And Ensure New Models Appear
**Files:**
- Modify: `docs/MODEL_INDEX.md`
**Verify:** `python tools/generate_model_index.py`

### Task 97: Run Full Test Suite
**Verify:** `pytest -q -o addopts=''`

### Task 98: Run Public API + Registry Audits
**Verify:** `python tools/audit_public_api.py && python tools/audit_registry.py`

### Task 99: Update `CHANGELOG.md` With Optimization Summary
**Files:**
- Modify: `CHANGELOG.md`
**Verify:** `rg -n \"optimize\" CHANGELOG.md | cat`

### Task 100: Single Final Commit (All Changes)
**Files:** (all changed)
**Command:**
```bash
git status -sb
git add -A
git commit -m "feat: 100-task optimization (models, features, preprocessing)"
```

