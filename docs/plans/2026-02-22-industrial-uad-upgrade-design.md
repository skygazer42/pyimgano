# Industrial UAD Upgrade: Datasets + Camera-Robust Inference + Zero-Shot Baselines (Design)

**Date:** 2026-02-22

## Context

`pyimgano` is already a broad visual anomaly detection toolkit (many algorithms + preprocessing utilities),
but “industrial” usage tends to stress a few recurring pain points:

1. **Newer industrial benchmarks** beyond classic MVTec AD / VisA (e.g. MVTec LOCO AD, MVTec AD 2) are
   increasingly used to represent real defect modes (logical anomalies, multi-split evaluation).
2. Industrial inspection images are often **high-resolution** (2K/4K), while many detectors assume
   ~224–518 px inputs. Downscaling hurts tiny defects, and naïve resize can mis-localize.
3. Many production lines have **camera + lighting drift** (white-balance, exposure, glare, compression),
   which creates false positives unless preprocessing/postprocessing is industrial-aware.
4. Recent research trends show strong interest in **foundation-model / VLM / CLIP-style** zero-shot and
   few-shot anomaly detection, plus “inference-first” workflows where training happens elsewhere and
   checkpoints are consumed by a stable evaluator.

This design focuses on **practical industrial accuracy and robustness** while keeping `pyimgano`
install-friendly and consistent with its current “numpy-first inference” direction.

## Web Research Notes (high-signal references)

- **MVTec LOCO AD** (logical + structural anomalies) publishes a clear dataset structure and JSON config
  format. The dataset root contains `train/*/good`, `test/*/{good,logical_anomalies,structural_anomalies}`
  and pixel GT masks under `ground_truth/*/{logical_anomalies,structural_anomalies}`. A `config.json` can
  describe `train` / `validation` / `test` splits. (Source: MVTec LOCO AD dataset docs.)
- **MVTec AD 2** (released 2025) introduces explicit splits under each object category:
  `train`, `validation`, `test_public`, `test_private`, `test_private_mixed`. (Source: MVTec AD 2 dataset
  docs / structure description.)
- **AnomalyCLIP (ICLR 2024)** is a popular CLIP-based anomaly detection/localization repo and highlights
  the “VLM prompts + patch/region localization” direction.
- **Dinomaly (CVPR 2025)** is integrated in anomalib v2.1.0, reinforcing the “checkpoint-first evaluation”
  workflow for modern models.
- **Tiled/patch-wise inference** is a common practical solution for high-res defect localization; recent
  work like tiled ensembles provides further motivation.

## Goals

1. **Add industrial dataset coverage**
   - Add loaders for:
     - `mvtec_loco` (MVTec LOCO AD)
     - `mvtec_ad2` (MVTec AD 2; at least train/val/test_public support)
   - Ensure existing loaders (e.g. `btad`) are reachable from CLI and pipelines.

2. **Make high-resolution inference a first-class workflow**
   - Provide a tiling wrapper that can:
     - run any “numpy-capable” detector on tiles,
     - aggregate image-level scores (max/mean/top-k),
     - stitch pixel anomaly maps back to full resolution (optional).
   - Keep the interface simple and reproducible (explicit tile size/stride/padding).

3. **Improve camera/lighting robustness via presets**
   - Add an “industrial camera robust” preprocessing preset geared to:
     - white balance / color constancy
     - illumination normalization
     - compression + blur + noise robustness
     - optional ROI masking
   - Expose preset construction in Python API and (optionally) CLI.

4. **Add modern, practical “latest” algorithm entry points**
   - Add a lightweight **OpenCLIP prompt-patch baseline** (AnomalyCLIP-inspired, but clearly labeled as a
     baseline) that:
     - uses `open_clip_torch` (optional extra) for embeddings,
     - produces both image scores and anomaly maps,
     - supports numpy inputs.
   - Expand anomalib checkpoint aliases for “latest” models (e.g. `vision_dinomaly_anomalib`) so users can
     benchmark trained checkpoints without reimplementing training in `pyimgano`.

5. **Numpy-first API robustness**
   - Make `pyimgano.inference.infer(..., include_maps=True)` work for more models by handling detectors
     that expect either `list[np.ndarray]` *or* batched `np.ndarray`.

6. **Docs + examples that match industrial usage**
   - Add clear “from zero → benchmark → inference → tiling” docs and at least one runnable example script.

## Non-Goals

- Shipping pretrained weights inside the wheel (weights should live outside the package and be mounted or
  downloaded to a cache directory).
- Re-implementing full training pipelines for large models (e.g. AnomalyCLIP training) inside `pyimgano`.
- Guaranteeing parity with every upstream repo’s metrics defaults (we target stable, documented behavior).

## Approaches Considered

### Approach A (Recommended, chosen): “Checkpoint-first + dataset/preprocess upgrades”

Add dataset loaders, tiling inference, robust preprocessing presets, and optional backends/wrappers.

Pros:
- Fast to ship, high practical value for industry
- Keeps core stable; heavy deps remain optional
- Matches how teams actually work (train elsewhere, evaluate here)

Cons:
- Some “latest” SOTA methods are only available as wrappers/aliases, not fully native training

### Approach B: “Full anomalib parity mode”

Lean heavily on anomalib for datasets/models and make `pyimgano` mostly an evaluation frontend.

Pros:
- Minimal reimplementation of models

Cons:
- Strong coupling to anomalib versions; harder to keep API stable

### Approach C: “Native reimplementation of latest papers”

Implement full training + inference for AnomalyCLIP/Dinomaly/etc.

Pros:
- Self-contained algorithms

Cons:
- Large engineering scope; higher risk; not aligned with lightweight packaging goals

## Proposed UX

### Benchmarking newer datasets

```bash
pyimgano-benchmark \
  --dataset mvtec_loco \
  --root /datasets/mvtec_loco \
  --category bottle \
  --model vision_patchcore \
  --device cuda \
  --pixel \
  --pixel-postprocess \
  --output runs/loco_bottle_patchcore.json
```

```bash
pyimgano-benchmark \
  --dataset mvtec_ad2 \
  --root /datasets/mvtec_ad2 \
  --category capsule \
  --model vision_dinomaly_anomalib \
  --checkpoint-path /checkpoints/dinomaly.ckpt \
  --device cuda \
  --pixel \
  --output runs/ad2_capsule_dinomaly.json
```

### High-resolution inference via tiling (Python)

```python
from pyimgano.models import create_model
from pyimgano.inference import infer
from pyimgano.inference.tiling import TiledDetector

det = create_model("vision_patchcore", device="cuda")
det.fit(train_images_np, input_format="rgb_u8_hwc")

tiled = TiledDetector(detector=det, tile_size=512, stride=384, score_reduce="max")
results = infer(tiled, [img_np], input_format="rgb_u8_hwc", include_maps=True)
```

## Architecture / Components

1. **Datasets**
   - Add `MVTecLOCODataset` and `MVTecAD2Dataset` in `pyimgano/utils/datasets.py` (and/or `pyimgano/datasets/`),
     plus extend `load_dataset()` and CLI choices.

2. **Tiling**
   - New module `pyimgano/inference/tiling.py`:
     - `iter_tiles(...)` generator + stitching utilities
     - `TiledDetector` wrapper implementing `fit`, `decision_function`, and optional `predict_anomaly_map`

3. **Industrial preprocessing presets**
   - New module `pyimgano/preprocessing/industrial_presets.py`:
     - white balance (gray-world / max-RGB)
     - illumination normalization (homomorphic filter / optional)
     - industrial robustness augmentation preset
   - Export presets via `pyimgano.preprocessing.__init__`.

4. **Modern model entry points**
   - New model: `vision_openclip_prompt_patch` (name TBD) built on `pyimgano/models/openclip_backend.py`
     helpers (optional dep: `open_clip` via `pyimgano[clip]`).
   - Add anomalib checkpoint aliases:
     - `vision_dinomaly_anomalib`
     - `vision_efficientad_anomalib`
     - potentially others that anomalib supports well

5. **Inference API compatibility improvements**
   - Improve `pyimgano/inference/api.py` to try both:
     - list inputs (`list[np.ndarray]`)
     - batched inputs (`np.ndarray` with shape `(N,H,W,C)`)
     when extracting scores/maps.

## Testing Strategy

- Unit tests for:
  - dataset path enumeration for LOCO/AD2 (using tmp directories with a minimal fake structure)
  - tiling stitching correctness (shape + overlap blending)
  - inference API fallback behavior for map extraction
  - industrial preprocessing functions (type/shape stability + deterministic paths)

## Documentation

- Update `README.md` (and/or `docs/`) with:
  - dataset setup expectations
  - tiling usage
  - “optional deps” guidance for OpenCLIP/anomalib

