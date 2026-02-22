# Industrial Robustness + VAND-Style SegF1 Benchmark (Design)

**Date:** 2026-02-22  
**Status:** Approved (user chose default threshold strategy = option 1)  
**Owner:** @codex (with user approval)

## Background

`pyimgano` already supports many industrial anomaly detection algorithms and pixel-map evaluation
(pixel AUROC / pixel AP / AUPRO). However, recent “industrial robustness” benchmarks increasingly
care about a **single deployment-ready threshold** and **robustness to drift** (lighting, compression,
blur, glare, geometric jitter).

The VAND 3.0 CVPR 2025 challenge emphasizes:

- **Pixel-level SegF1** as a primary metric
- **Single global threshold** (deployment-style)
- Distribution shift settings (e.g., MVTec AD 2 has “lighting shift” splits)

This design adds a reproducible “robustness benchmark harness” to `pyimgano` and makes SegF1 a
first-class pixel metric, without bloating the core dependency footprint.

## Web Research Notes (high-signal references)

- VAND 3.0 CVPR 2025 challenge overview and metric focus (SegF1 single-threshold, robustness).  
  Ref: https://sites.google.com/view/vand30cvpr2025/challenge
- MVTec AD 2 dataset split structure (`train`, `validation`, `test_public`, `test_private`, etc.).  
  Ref: https://www.mvtec.com/company/research/datasets/mvtec-ad-2
- SuperAD (2025) describes a strong “foundation-feature + training-free memory bank” direction.  
  Ref: https://arxiv.org/abs/2505.19750
- SNARM (2025) explores self-referential learning and Mamba-style fusion for anomaly detection.  
  Ref: https://arxiv.org/abs/2508.01591
- RoBiS (VAND 3.0-related) repo notes practical robustness tricks (HR cropping, adaptive binarization,
  augmentation, optional SAM refinement).  
  Ref: https://github.com/pangdatangtt/RoBiS

## Goals

1. Add **pixel SegF1** as a first-class metric:
   - computed from `(pixel_labels, pixel_scores, threshold)`
   - uses **one threshold** for the whole evaluation (split/category)
2. Add a **deploy-friendly pixel-threshold calibration** default:
   - default strategy: **`normal_pixel_quantile`** (user chose option 1)
   - uses *normal pixels only* (no GT required for calibration)
3. Add a deterministic **industrial drift corruption suite** aligned with production failure modes:
   - lighting/exposure/white-balance
   - JPEG artifacts
   - blur (defocus/motion-ish)
   - glare / specular highlight
   - geometric jitter (rotation/translation/scale)
4. Add a **robustness benchmark runner** that:
   - evaluates clean + corruptions using the **same calibrated threshold**
   - reports a minimal “production” metric set:
     - `pixel_segf1`
     - `bg_fpr` (false-positive rate on background pixels, GT==0)
     - `latency_ms_per_image` (best-effort)
5. Keep the API consistent:
   - Works for any detector that already supports pixel maps (`predict_anomaly_map` or `get_anomaly_map`)
   - Leverages existing tiling + post-processing

## Non-Goals

- Shipping pretrained weights inside the wheel.
- Reproducing every upstream repo’s full training pipeline.
- Adding heavyweight new mandatory dependencies.

## Proposed UX

### 1) Standard benchmark (adds SegF1)

```bash
pyimgano-benchmark \
  --dataset mvtec_ad2 \
  --root /datasets/mvtec_ad2 \
  --category capsule \
  --model vision_patchcore \
  --pixel \
  --pixel-segf1 \
  --pixel-threshold-strategy normal_pixel_quantile \
  --pixel-normal-quantile 0.999 \
  --output runs/ad2_capsule_patchcore.json
```

### 2) Robustness benchmark (clean + drift corruptions)

```bash
pyimgano-robust-benchmark \
  --dataset mvtec_ad2 \
  --root /datasets/mvtec_ad2 \
  --category capsule \
  --model vision_patchcore \
  --pixel \
  --pixel-segf1 \
  --pixel-threshold-strategy normal_pixel_quantile \
  --pixel-normal-quantile 0.999 \
  --corruptions industrial_drift \
  --severity 1 2 3 4 5 \
  --seed 0 \
  --output runs/ad2_capsule_patchcore_robust.json
```

## Threshold Strategy (Pixel)

### Default (chosen): `normal_pixel_quantile`

- **Input:** pixel anomaly maps on a *calibration set of normal images*.
- **Output:** threshold = quantile(`pixel_scores_normal`, `q`)
- Calibration set selection priority:
  1. dataset-provided validation normal split (if available)
  2. otherwise: deterministic hold-out split from `train/good` (e.g. 80/20, fixed seed)

This avoids the common kNN “self-match” degeneracy when calibrating on the same images used to build
the memory bank (important for PatchCore-like / DINO-kNN detectors).

### Optional: `best_f1_public` (leaderboard-only)

An opt-in mode that picks the threshold maximizing SegF1 on the public test split using GT masks.
This is explicitly **not** deploy-friendly and not the default.

## Robustness / Corruption Suite

Add a deterministic corruption set with severity 1–5, fixed RNG seed:

1. **lighting**: brightness/contrast/gamma + mild WB shift
2. **jpeg**: encode/decode at quality levels
3. **blur**: Gaussian blur (kernel/sigma by severity)
4. **glare**: synthetic specular blob(s) using existing defect synthesis helpers
5. **geo_jitter**: small affine warp; transforms mask consistently

Each corruption is implemented as:

- `apply(image: np.ndarray, mask: Optional[np.ndarray], severity: int, rng) -> (image, mask)`

## Architecture / Components

1. **Metrics**
   - Extend `pyimgano.evaluation` with:
     - `compute_pixel_segf1(pixel_labels, pixel_scores, threshold)`
     - `compute_bg_fpr(pixel_labels, pixel_scores, threshold)`
     - helpers for threshold calibration from normal pixels

2. **Robustness harness**
   - New `pyimgano/robustness/` package:
     - `corruptions.py` (implement deterministic corruptions)
     - `benchmark.py` (run clean + corrupted evaluation, produce JSON)

3. **Pipeline integration**
   - Extend `pyimgano.pipelines.mvtec_visa.evaluate_split()` (or add a new pipeline wrapper) to:
     - compute pixel maps
     - calibrate pixel threshold (if requested)
     - compute SegF1 + bg FPR using the single threshold

4. **CLI**
   - Extend `pyimgano-benchmark` to expose SegF1 + threshold strategy flags
   - Add `pyimgano-robust-benchmark` for corruption suite evaluation

5. **Models (optional, v0.5.0 scope permitting)**
   - Add a “foundation baseline” model entry:
     - `vision_superad` (DINOv2 patch-kNN, training-free baseline)
   - Add an optional Mamba-flavored model entry:
     - `vision_snarm` (experimental; optional dependency: `pyimgano[mamba]`)

If implementation scope becomes tight, the benchmark/methodology changes are higher priority than
shipping additional detectors, because they benefit every existing detector immediately.

## Testing Strategy

- Unit tests for:
  - SegF1 correctness on small synthetic masks/scores
  - bg FPR correctness
  - deterministic corruption outputs (seeded, severity mapping)
  - mask-warp correctness for `geo_jitter`
- CLI smoke tests:
  - `--pixel-segf1` flag path works (monkeypatch pipeline calls)
  - robust benchmark JSON schema stable

## Documentation

- Add a dedicated doc:
  - `docs/ROBUSTNESS_BENCHMARK.md` (what it is, how to run, what metrics mean)
- Update:
  - `docs/INDUSTRIAL_INFERENCE.md` (recommend robust eval workflow)
  - `docs/ALGORITHM_SELECTION_GUIDE.md` (when to care about SegF1 + drift)
  - `CHANGELOG.md`

