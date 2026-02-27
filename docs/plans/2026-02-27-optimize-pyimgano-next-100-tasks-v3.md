# PyImgAno Next 100 Tasks (v3) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand PyImgAno’s industrial anomaly detection toolkit with more native `core_*` algorithms, modernize legacy/deep models to our base-class contract, and strengthen industrial synthesis + embedding pipelines **without adding heavy new dependencies or bundling model weights**.

**Architecture:** Keep our own base contracts (`BaseDetector`, `BaseVisionDetector`, `BaseVisionDeepDetector`, `CoreFeatureDetector`) as the center of gravity. External repos are used for study only (shallow clone). New capabilities are added as small, testable, registry-registered modules.

**Tech Stack:** Python, NumPy/SciPy, scikit-learn, OpenCV, Torch/Torchvision (no implicit weight downloads), scikit-image, pytest.

---

## Constraints / Non-Negotiables

- No `pyod` runtime dependency. No thin wrappers like `from pyod import ...`.
- Implementations must conform to our contracts:
  - `BaseDetector` semantics (higher score = more anomalous; contamination thresholding; `predict()` returns {0,1}).
  - `core_*` models consume `np.ndarray` feature matrices (or torch tensors convertible to numpy).
  - `vision_*` models consume image paths and/or in-memory `np.ndarray` images via feature extractors / dataset wrappers.
- Avoid expanding package size:
  - No bundled weights, no big binary assets in repo.
  - New optional deps must be guarded and never required for `import pyimgano`.
- **Git policy for this batch:** do not commit incrementally; only one final commit after all 100 tasks are completed + verified.

---

## Phase 0 — Research, Governance, Guardrails (Tasks 1–15)

### Task 1: Refresh Industrial Reference Index (web search + study targets)
**Files:**
- Modify: `docs/INDUSTRIAL_REFERENCE_PROJECTS.md`

**Outcome:**
- Add 15–25 more repos (frameworks + methods + augmentation toolkits) with license notes + “what to learn”.

### Task 2: Expand Shallow Clone Helper Repo List
**Files:**
- Modify: `tools/clone_reference_repos.sh`

**Outcome:**
- Add more study repos (e.g. RealNet, RD++, STFPM, PUAD, etc.) while keeping clones shallow and gitignored.

### Task 3: Add Third-Party Code Policy Doc
**Files:**
- Create: `docs/THIRD_PARTY_CODE_POLICY.md`

**Outcome:**
- Define how to copy code legally (license-compatible only), how to keep notices, how to annotate files (`UPSTREAM:` header).

### Task 4: Add Third-Party Notices Folder + Template
**Files:**
- Create: `third_party/README.md`
- Create: `third_party/NOTICE.md`

**Outcome:**
- A canonical place to store upstream license texts / notices when code is copied.

### Task 5: Add “Upstream Notice” Audit Tool
**Files:**
- Create: `tools/audit_third_party_notices.py`

**Outcome:**
- Scan for `UPSTREAM:` markers and ensure a corresponding notice exists under `third_party/`.

### Task 6: Stronger “No Network Downloads” Guardrail for Torchvision Weights
**Files:**
- Create: `tests/test_no_torchvision_weight_downloads_by_default.py`

**Outcome:**
- Monkeypatch `torch.hub.load_state_dict_from_url` to hard-fail; instantiate selected models/extractors with defaults; ensure no download attempted.

### Task 7: Add Safe Torchvision Backbone Loader Utility (single place)
**Files:**
- Create: `pyimgano/utils/torchvision_safe.py`
- Modify: `pyimgano/features/torchvision_backbone.py`
- Modify: `pyimgano/features/torchvision_multilayer.py`

**Outcome:**
- Centralize “pretrained=False by default” and `weights=None` handling across torchvision versions.

### Task 8: Add Import-Cost Audit Script (best-effort)
**Files:**
- Create: `tools/audit_import_costs.py`

**Outcome:**
- Rough timing for `import pyimgano`, `import pyimgano.models`, and selected deep modules to prevent accidental import-time explosions.

### Task 9: Registry Metadata Hygiene Pass
**Files:**
- Modify: `pyimgano/models/capabilities.py`
- Modify: selected model modules lacking `metadata.description`

**Outcome:**
- Ensure each registry entry has description + stable tags; reduce “unknown” entries.

### Task 10: Documentation for “Core vs Vision vs Deep” Path (v3)
**Files:**
- Modify: `docs/ARCHITECTURE_CLASSICAL_PIPELINES.md`
- Create: `docs/ARCHITECTURE_DEEP_CONTRACTS.md`

**Outcome:**
- Document industrial recommended route: `torch embeddings -> core_* classical -> calibration`.

### Task 11: Add Minimal Determinism Helpers for Torch Training/Eval
**Files:**
- Create: `pyimgano/utils/torch_determinism.py`
- Modify: `pyimgano/utils/torch_infer.py`

**Outcome:**
- One helper to set seeds and deterministic flags for quick reproducible tests/bench runs.

### Task 12: Extend Optional-Dependency Utilities (error UX)
**Files:**
- Modify: `pyimgano/utils/optional_deps.py`
- Test: `tests/test_optional_deps_utils.py` (extend)

**Outcome:**
- Better error messages + “install extra” hints.

### Task 13: Harden “No PyOD” Rule (docs allowed, code forbidden)
**Files:**
- Modify: `tests/test_no_pyod_imports.py` (tighten path filtering)

**Outcome:**
- Keep docs references allowed, but ensure no imports in runtime code.

### Task 14: Add “Public API Surface” Smoke for New Modules
**Files:**
- Modify: `tools/audit_public_api.py`

**Outcome:**
- Ensure new modules don’t accidentally export unstable internal names.

### Task 15: Update CHANGELOG Stub for Next Release
**Files:**
- Modify: `CHANGELOG.md`

**Outcome:**
- Add “Unreleased” section describing the upcoming batch at a high level.

---

## Phase 1 — Core Algorithms Expansion (Tasks 16–50)

### Task 16: Register `core_sos` Wrapper (feature-matrix SOS)
**Files:**
- Modify: `pyimgano/models/sos.py`

### Task 17: Register `core_sod` Wrapper
**Files:**
- Modify: `pyimgano/models/sod.py`

### Task 18: Register `core_rod` Wrapper
**Files:**
- Modify: `pyimgano/models/rod.py`

### Task 19: Register `core_imdd` Wrapper
**Files:**
- Modify: `pyimgano/models/imdd.py`

### Task 20: Register `core_lmdd` Wrapper (aliasing IMDD mechanics)
**Files:**
- Modify: `pyimgano/models/lmdd.py`

### Task 21: Add `core_elliptic_envelope` (native robust covariance scoring)
**Files:**
- Create: `pyimgano/models/elliptic_envelope.py`
- Test: covered by `tests/test_core_models_registry_smoke.py`

### Task 22: Add `vision_elliptic_envelope` Wrapper
**Files:**
- Modify: `pyimgano/models/elliptic_envelope.py`

### Task 23: Add `core_mst_outlier` (MST-based outlier score)
**Files:**
- Create: `pyimgano/models/mst_outlier.py`

### Task 24: Add `vision_mst_outlier` Wrapper
**Files:**
- Modify: `pyimgano/models/mst_outlier.py`

### Task 25: Add `core_lid` (Local Intrinsic Dimensionality outlier score)
**Files:**
- Create: `pyimgano/models/lid.py`

### Task 26: Add `vision_lid` Wrapper
**Files:**
- Modify: `pyimgano/models/lid.py`

### Task 27: Add `core_cook_distance` (linear influence baseline)
**Files:**
- Create: `pyimgano/models/cook_distance.py`

### Task 28: Add `core_studentized_residual` baseline (robust regression residual)
**Files:**
- Create: `pyimgano/models/studentized_residual.py`

### Task 29: Add `core_quantile_forest`-like lightweight approximation (sklearn ExtraTrees)
**Files:**
- Create: `pyimgano/models/extra_trees_density.py`

### Task 30: Add `core_random_projection_knn` (RP + kNN distance)
**Files:**
- Create: `pyimgano/models/random_projection_knn.py`

### Task 31: Add `core_rbf_kde_ratio` (density ratio via KDE)
**Files:**
- Create: `pyimgano/models/kde_ratio.py`

### Task 32: Add `core_neighborhood_entropy` (graph entropy outlier score)
**Files:**
- Create: `pyimgano/models/neighborhood_entropy.py`

### Task 33: Add `core_score_standardizer` Wrapper (rank/zscore/robust)
**Files:**
- Create: `pyimgano/models/core_score_standardizer.py`
- Create: `pyimgano/calibration/score_standardization.py`
- Test: `tests/test_score_standardizer.py`

### Task 34: Add `vision_score_standardizer` Wrapper (vision -> standardized score)
**Files:**
- Create: `pyimgano/models/vision_score_standardizer.py`
- Test: `tests/test_vision_score_standardizer.py`

### Task 35: Ensemble Spec Resolver (detectors as names/spec dicts)
**Files:**
- Create: `pyimgano/models/ensemble_spec.py`
- Test: `tests/test_ensemble_spec.py`

### Task 36: Upgrade `VisionScoreEnsemble` to Support Spec Inputs (no breaking)
**Files:**
- Modify: `pyimgano/models/score_ensemble.py`
- Test: `tests/test_score_ensemble_specs.py`

### Task 37: Add `core_score_ensemble` (feature-matrix detector ensemble)
**Files:**
- Create: `pyimgano/models/core_score_ensemble.py`
- Test: `tests/test_core_score_ensemble.py`

### Task 38: Add `vision_lscp_spec` Wrapper (spec-based LSCP)
**Files:**
- Modify: `pyimgano/models/lscp.py`
- Test: `tests/test_lscp_spec_inputs.py`

### Task 39: Add `vision_suod_spec` Wrapper (spec-based SUOD)
**Files:**
- Modify: `pyimgano/models/suod.py`
- Test: `tests/test_suod_spec_inputs.py`

### Task 40: Add `vision_feature_pipeline` Docs + Examples (core detector names)
**Files:**
- Modify: `docs/ARCHITECTURE_CLASSICAL_PIPELINES.md`
- Create: `examples/feature_pipeline_core_detectors.py`

### Task 41: Add New “core_*” Registry Smoke Coverage for Added Models
**Files:**
- Modify: `tests/test_core_models_registry_smoke.py` (ensure new models included; skip only deep)

### Task 42: Add “core model info” payload tests for new models
**Files:**
- Modify: `tests/test_model_info_payload.py`

### Task 43: Add Core Algorithm Numerical-Stability Tests (NaN/Inf guards)
**Files:**
- Create: `tests/test_core_models_numerical_stability.py`

### Task 44: Add `core_*` Save/Load for Backends that Support It (joblib)
**Files:**
- Create: `pyimgano/models/serialization.py`
- Test: `tests/test_core_serialization.py`

### Task 45: Add “Score Direction” Audit (higher=more anomalous)
**Files:**
- Create: `tools/audit_score_direction.py`

### Task 46: Add `vision_*` Wrappers for New Cores via `VisionFeaturePipeline`
**Files:**
- Create: `pyimgano/models/industrial_wrappers.py`

### Task 47: Add Industrial Preset Model Configs (JSON-ready)
**Files:**
- Create: `pyimgano/presets/industrial_classical.py`
- Modify: `pyimgano/cli_presets.py`

### Task 48: CLI: Allow Selecting Model by Preset Name
**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_presets_roundtrip.py`

### Task 49: Add Docs: “Deep Embedding + Core” Industrial Route
**Files:**
- Create: `docs/INDUSTRIAL_EMBEDDING_PLUS_CORE.md`

### Task 50: Add Benchmarks Stub for New Core Models
**Files:**
- Modify: `benchmarks/run_bench.py` (or existing harness)

---

## Phase 2 — Deep Models Modernization (Tasks 51–80)

### Task 51: Deep Model Contract Doc + Shared Helpers
**Files:**
- Create: `pyimgano/models/deep_contract.py`
- Create: `docs/ARCHITECTURE_DEEP_CONTRACTS.md`

### Task 52: Modernize `efficient_ad` -> `vision_efficientad` (no prints; BaseVisionDeepDetector)
**Files:**
- Modify: `pyimgano/models/efficientad.py`
- Test: `tests/test_vision_efficientad_smoke.py`

### Task 53: Modernize `ae_resnet_unet` to Contract (`fit(X)` and `decision_function(X)`)
**Files:**
- Modify: `pyimgano/models/ae.py`
- Test: `tests/test_ae_resnet_unet_contract.py`

### Task 54: Modernize `vae_conv` to Contract
**Files:**
- Modify: `pyimgano/models/vae.py`
- Test: `tests/test_vae_conv_contract.py`

### Task 55: Remove Import-Time `matplotlib` in Deep Models (lazy/optional)
**Files:**
- Modify: `pyimgano/models/ae.py`
- Modify: `pyimgano/models/vae.py`

### Task 56: Ensure All Deep Models Default to `pretrained=False` (no downloads)
**Files:**
- Modify: multiple deep model modules
- Test: relies on Task 6 guard

### Task 57: Add `tiny=True` Mode for Selected Deep Models (fast unit tests)
**Files:**
- Modify: `pyimgano/models/draem.py`
- Modify: `pyimgano/models/fastflow.py`
- Modify: `pyimgano/models/stfpm.py`
- Modify: `pyimgano/models/reverse_distillation.py`
- Tests: per-model smoke tests

### Task 58: Deep Save/Load Best-Effort (state_dict + metadata)
**Files:**
- Create: `pyimgano/models/deep_io.py`
- Test: `tests/test_deep_io_roundtrip.py`

### Task 59: Unify `decision_function` Return Types for Deep Models (np.ndarray float64)
**Files:**
- Modify: selected deep models
- Test: `tests/test_deep_decision_function_shapes.py`

### Task 60: Add Optional Pixel-Level Output (`anomaly_map`) for Models that Support It
**Files:**
- Modify: selected deep models
- Test: `tests/test_deep_anomaly_map_shapes.py`

### Task 61: Make Deep Models Accept In-Memory Arrays via `VisionArrayDataset`
**Files:**
- Modify: selected deep models (dataset usage)
- Test: `tests/test_deep_models_numpy_inputs.py`

### Task 62: Add `VisionEmbeddingCoreDetector` (deep extractor + core detector model)
**Files:**
- Create: `pyimgano/models/vision_embedding_core.py`
- Test: `tests/test_vision_embedding_core_smoke.py`

### Task 63: Add `vision_patchcore_lite` (embedding + approximate NN without faiss)
**Files:**
- Create: `pyimgano/models/patchcore_lite.py`
- Test: `tests/test_patchcore_lite_smoke.py`

### Task 64: Add `vision_padim_lite` (embedding + Gaussian stats)
**Files:**
- Create: `pyimgano/models/padim_lite.py`
- Test: `tests/test_padim_lite_smoke.py`

### Task 65: Add `vision_student_teacher_lite` (STFPM-like lite)
**Files:**
- Create: `pyimgano/models/student_teacher_lite.py`
- Test: `tests/test_student_teacher_lite_smoke.py`

### Task 66: Add Postprocess: Morphology for Anomaly Maps (industrial cleanup)
**Files:**
- Create: `pyimgano/postprocess/morphology.py`
- Test: `tests/test_postprocess_morphology.py`

### Task 67: Add Postprocess: Connected Components + Region Scoring
**Files:**
- Create: `pyimgano/postprocess/connected_components.py`
- Test: `tests/test_connected_components.py`

### Task 68: Add “Pixel → Image score” reducers (max/mean/topk/area)
**Files:**
- Create: `pyimgano/postprocess/reducers.py`
- Test: `tests/test_postprocess_reducers.py`

### Task 69: Add Deep Model Registry Tags (“pixel”, “image”, “fewshot”, etc.)
**Files:**
- Modify: selected model registry decorators

### Task 70: Add Deep Model Smoke Suite (construct + 1 step fit on tiny data)
**Files:**
- Create: `tests/test_deep_models_tiny_smoke.py`

### Task 71: Add Deep Training Progress Logging (no prints)
**Files:**
- Modify: selected deep models

### Task 72: Add Deep Mixed Precision Toggle (optional, safe defaults)
**Files:**
- Create: `pyimgano/utils/torch_amp.py`
- Modify: selected deep models

### Task 73: Add `pyimgano-train` Support for New Deep Models (config surface)
**Files:**
- Modify: `pyimgano/train_cli.py`
- Modify: config schema docs

### Task 74: Add “No GPU Required” Fallback for Unit Tests (CPU-only paths)
**Files:**
- Modify: deep tests to force `device=cpu`

### Task 75: Add Robustness Corruptions Dataset Wrapper (industrial)
**Files:**
- Create: `pyimgano/datasets/corruptions.py`
- Test: `tests/test_corruptions_dataset.py`

### Task 76: Add Robustness CLI Mode: Evaluate Under Corruptions
**Files:**
- Modify: `pyimgano/robust_cli.py`
- Test: `tests/test_robust_cli_smoke.py`

### Task 77: Add `vision_winclip_lite` Safety Defaults (no implicit model download)
**Files:**
- Modify: `pyimgano/models/winclip.py`
- Test: existing optional tests

### Task 78: Add Feature Cache Support in `BaseVisionDeepDetector` (best-effort)
**Files:**
- Modify: `pyimgano/models/baseCv.py`
- Create: `pyimgano/cache/deep_embeddings.py`

### Task 79: Add Doc: “Deep Models: Recommended vs Experimental”
**Files:**
- Create: `docs/DEEP_MODELS_STATUS.md`

### Task 80: Add Benchmarks: Deep Tiny Mode Speed + Memory
**Files:**
- Modify: `benchmarks/`

---

## Phase 3 — Embedding + Pipeline Enhancements (Tasks 81–90)

### Task 81: Add ViT Token Feature Extractor (torchvision)
**Files:**
- Create: `pyimgano/features/torchvision_vit_tokens.py`
- Test: `tests/test_torchvision_vit_tokens_extractor.py`

### Task 82: Add Embedding Normalization Extractor (L2 + power)
**Files:**
- Create: `pyimgano/features/normalize.py`
- Test: `tests/test_feature_normalize_extractor.py`

### Task 83: Add Feature-Extractor Composition DSL (simple pipeline spec)
**Files:**
- Modify: `pyimgano/features/multi.py`
- Test: `tests/test_feature_pipeline_specs.py`

### Task 84: Add `pyimgano-features` Manifest Mode (read manifest JSONL)
**Files:**
- Modify: `pyimgano/feature_cli.py`
- Test: `tests/test_feature_cli_manifest_mode.py`

### Task 85: Add “Export Features + IDs” helper (stable ordering)
**Files:**
- Create: `pyimgano/features/export.py`
- Test: `tests/test_feature_export.py`

### Task 86: Add Embedding Cache Controls to Feature CLI
**Files:**
- Modify: `pyimgano/feature_cli.py`
- Test: `tests/test_feature_cli_cache_dir.py`

### Task 87: Add Docs: “Embeddings + Core Recipes” (industrial presets)
**Files:**
- Create: `docs/RECIPES_EMBEDDINGS_PLUS_CORE.md`

### Task 88: Add Example: “Torchvision Multi-layer + core_ecod”
**Files:**
- Create: `examples/embedding_plus_core_ecod.py`

### Task 89: Add Example: “OpenCLIP + core_knn (optional)”
**Files:**
- Create: `examples/openclip_plus_core_knn.py`

### Task 90: Add E2E Test: Embedding+Core Pipeline on Generated Synthetic Dataset
**Files:**
- Create: `tests/test_e2e_synth_embedding_core_pipeline.py`

---

## Phase 4 — Synthesis / Augmentation v2 (Tasks 91–98)

### Task 91: Add New Presets: Rust/Corrosion, Oil Stain, Crack, Pitting
**Files:**
- Modify: `pyimgano/synthesis/presets.py`
- Create: `pyimgano/synthesis/defects.py`
- Test: `tests/test_synthesis_presets_more.py`

### Task 92: Add Scratch Generator v2 (curved + multi-stroke)
**Files:**
- Modify: `pyimgano/synthesis/presets.py`
- Modify: `pyimgano/synthesis/masks.py`

### Task 93: Add Texture-Source Bank + Paste (anomalib-like)
**Files:**
- Create: `pyimgano/synthesis/sources.py`
- Modify: `pyimgano/synthesis/synthesizer.py`
- Test: `tests/test_synthesis_sources_bank.py`

### Task 94: Add CLI Preview Mode for Synthesis (grid output)
**Files:**
- Modify: `pyimgano/synthesize_cli.py`
- Test: `tests/test_synthesize_cli_preview.py`

### Task 95: Add CLI “from manifest” mode (augment existing dataset)
**Files:**
- Modify: `pyimgano/synthesize_cli.py`
- Test: `tests/test_synthesize_cli_from_manifest.py`

### Task 96: Add Dataset Wrapper: Synthetic-on-the-fly (train-time augmentation)
**Files:**
- Modify: `pyimgano/datasets/synthetic.py`
- Test: `tests/test_synthetic_dataset_wrapper_v2.py`

### Task 97: Add Robust Corruptions Integration (synthesis used as corruption)
**Files:**
- Modify: `pyimgano/datasets/corruptions.py`
- Test: `tests/test_corruptions_with_synthesis.py`

### Task 98: Add Docs: Synthetic Pipeline + Manifests + ROI masks
**Files:**
- Modify: `docs/SYNTHETIC_ANOMALY_GENERATION.md`

---

## Phase 5 — Final Verification + Single Commit (Tasks 99–100)

### Task 99: Full Verification Sweep
**Run:**
- `pytest -q -o addopts=''`
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`

### Task 100: Update Release Notes + One Final Commit
**Files:**
- Modify: `CHANGELOG.md`

**Commit:**
- Stage everything and create exactly one commit for this batch.

