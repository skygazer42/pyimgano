# Industrial UAD Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve `pyimgano` for industrial anomaly detection by adding newer datasets (LOCO/AD2), high-res tiling inference, camera-robust preprocessing presets, and modern “latest” entry points (OpenCLIP + anomalib aliases) while keeping the numpy-first API consistent.

**Architecture:** Extend the dataset factory + benchmark CLI to include LOCO/AD2, add a tiling wrapper in `pyimgano.inference`, add industrial presets in `pyimgano.preprocessing`, and harden inference/map extraction to support more detector input conventions.

**Tech Stack:** NumPy, OpenCV, PyTorch (existing), optional `open_clip_torch` (OpenCLIP), optional `anomalib`.

---

### Task 1: Add MVTec LOCO dataset loader (paths-first)

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Test: `tests/test_datasets_loco.py`

**Step 1: Write failing test**
- Create a minimal fake LOCO structure:
  - `root/<cat>/train/good/*.png`
  - `root/<cat>/test/good/*.png`
  - `root/<cat>/test/logical_anomalies/<defect>/*.png`
  - `root/<cat>/ground_truth/logical_anomalies/<defect>/*.png` (or `_mask.png`)
  - Same for `structural_anomalies`
- Assert `get_train_paths()` and `get_test_paths()` return consistent counts and mask shapes.

**Step 2: Implement loader**
- Add `MVTecLOCODataset` with:
  - `CATEGORIES = ["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors"]`
  - Robust discovery: accept either “flat” or “one-level nested” anomaly dirs.
  - Mask lookup fallback order:
    1) same relative path under `ground_truth`
    2) same name with `_mask` suffix
    3) zero mask

**Step 3: Run targeted tests**
- Run: `pytest tests/test_datasets_loco.py -q`

---

### Task 2: Wire LOCO into dataset factory + CLI choices

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/pipelines/mvtec_visa.py`
- Test: `tests/test_cli_smoke.py` (or new focused test)

**Steps**
- Add `mvtec_loco` to `load_dataset()` mapping.
- Extend CLI `--dataset` choices to include `mvtec_loco` and `btad` (already supported by factory).
- Extend pipeline dataset union/type to include LOCO (and BTAD if desired).

---

### Task 3: Add MVTec AD 2 dataset loader (test_public + masks)

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Test: `tests/test_datasets_mvtec_ad2.py`

**Steps**
- Add `MVTecAD2Dataset` with split selection:
  - `split="test_public"` default for evaluation (has GT)
  - training uses `train/good`
- Implement `get_train_paths()` and `get_test_paths()`:
  - Test images under `<split>/{good,bad}`
  - Masks under `<split>/ground_truth/bad` (with `_mask` suffix handling)

---

### Task 4: Wire AD2 into dataset factory + pipeline + CLI

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/pipelines/mvtec_visa.py`
- Test: `tests/test_pipeline_smoke.py` (or new focused test)

**Steps**
- Add `mvtec_ad2` to `load_dataset()` mapping.
- Add `mvtec_ad2` to CLI dataset choices.
- Ensure `load_benchmark_split()` can pass split kwargs (e.g. `split="test_public"`) via `load_dataset`.

---

### Task 5: Add tiling inference wrapper for high-resolution industrial images

**Files:**
- Create: `pyimgano/inference/tiling.py`
- Test: `tests/test_inference_tiling.py`

**Steps**
- Implement:
  - `iter_tile_coords(h, w, tile_size, stride)` (cover full image, include last tile)
  - `extract_tile(image, y0, x0, tile_size, pad_mode="reflect")`
  - `stitch_maps(tiles, maps, out_shape, reduce="max"|"mean")`
  - `TiledDetector(detector, tile_size, stride, score_reduce="max")`
    - `fit()` delegates to wrapped detector (no tiling for fit)
    - `decision_function()` tiles each image, reduces tile scores
    - `predict_anomaly_map()` tiles, stitches

---

### Task 6: Harden `infer(include_maps=True)` to support list-vs-batch detector APIs

**Files:**
- Modify: `pyimgano/inference/api.py`
- Test: `tests/test_inference_api.py` (add new test)

**Steps**
- For numpy inputs, try calling:
  1) detector methods with `list[np.ndarray]`
  2) detector methods with `np.stack(list)` (batched ndarray)
- Do this for both `decision_function` (best-effort) and `predict_anomaly_map`.

---

### Task 7: Make OpenCLIP patch embedder accept numpy inputs (paths OR arrays)

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_patch_tokens_optional.py` (extend) or add new test

**Steps**
- Extend `OpenCLIPViTPatchEmbedder.embed()` to accept `Union[str, np.ndarray]`:
  - If ndarray: assume RGB/u8/HWC and convert to PIL Image.
  - Preserve `original_size` from array.

---

### Task 8: Enable numpy-first OpenCLIP promptscore + patchknn detectors

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_promptscore.py` / `tests/test_openclip_promptscore_core.py`

**Steps**
- Change `VisionOpenCLIPPromptScore._embed` to accept `Union[str, np.ndarray]`.
- Ensure `decision_function()` + `get_anomaly_map()` work with arrays.

---

### Task 9: Add “industrial camera robust” preprocessing utilities (white balance + illumination)

**Files:**
- Create: `pyimgano/preprocessing/industrial_presets.py`
- Modify: `pyimgano/preprocessing/__init__.py`
- Test: `tests/test_preprocessing_industrial.py`

**Steps**
- Implement:
  - `gray_world_white_balance(image)`
  - `max_rgb_white_balance(image)`
  - `homomorphic_filter(image, cutoff=..., gain=...)` (optional but useful)
- Keep contract stable: preserve dtype where reasonable, output shape unchanged.

---

### Task 10: Add industrial robustness augmentations (JPEG + vignetting + exposure drift)

**Files:**
- Modify: `pyimgano/preprocessing/augmentation.py`
- Modify: `pyimgano/preprocessing/augmentation_pipeline.py`
- Modify: `pyimgano/preprocessing/__init__.py`
- Test: `tests/test_augmentation_registry.py` (or new test)

**Steps**
- Add ops:
  - `jpeg_compress(image, quality=...)`
  - `vignette(image, strength=...)`
  - `random_channel_gain(image, ...)` (color temperature-ish)
- Add preset `get_industrial_camera_robust_augmentation()`.

---

### Task 11: Add anomalib checkpoint aliases for “latest” models

**Files:**
- Modify: `pyimgano/models/anomalib_backend.py`
- Test: `tests/test_anomalib_backend_optional.py` (registry presence)

**Steps**
- Add registry aliases (same implementation via inheritance):
  - `vision_dinomaly_anomalib`
  - `vision_efficientad_anomalib`
  - (optional) `vision_reverse_distillation_anomalib`
- Ensure metadata tags include model name + `requires_checkpoint=True`.

---

### Task 12: Extend benchmark CLI dataset list + help text

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke.py`

**Steps**
- Add dataset choices: `btad`, `mvtec_loco`, `mvtec_ad2`.
- Improve `--help` dataset description (brief).

---

### Task 13: Add docs + examples for industrial workflows

**Files:**
- Modify: `README.md`
- Create: `examples/industrial_tiling_infer.py`

**Steps**
- Add sections:
  - “Datasets: LOCO/AD2 (local paths)”
  - “High-res tiling inference”
  - “Optional deps: OpenCLIP/anomalib”

---

### Task 14: Add dataset smoke tests for LOCO/AD2 pipeline integration

**Files:**
- Create: `tests/test_pipeline_loco_smoke.py`
- Create: `tests/test_pipeline_ad2_smoke.py`

**Steps**
- Use fake tmp dataset folders.
- Run `load_benchmark_split()` and ensure it returns consistent path lists + labels/masks.

---

### Task 15: Add tiling + postprocess integration test

**Files:**
- Modify: `tests/test_pipeline_pixel_scores.py` (or create new focused test)

**Steps**
- Use a dummy detector exposing `predict_anomaly_map`.
- Wrap in `TiledDetector` and verify maps stitch + evaluation runs.

---

### Task 16: Add docs on weight management (no weights in wheel)

**Files:**
- Modify: `docs/PUBLISHING.md` or create `docs/WEIGHTS.md`

**Steps**
- Document recommended pattern:
  - download to `~/.cache/pyimgano/...` or user-controlled dir
  - pass checkpoint paths explicitly

---

### Task 17: Ensure `pyimgano.infer_cli` supports tiling (optional)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_smoke.py`

**Steps**
- Add args:
  - `--tile-size`, `--tile-stride`, `--tile-reduce`
- When set, wrap detector with `TiledDetector` before calling `infer()`.

---

### Task 18: Version bump + changelog entry

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

**Steps**
- Bump to `0.3.0` (feature release).
- Add changelog bullets for LOCO/AD2 + tiling + OpenCLIP numpy support.

---

### Task 19: Run the full test suite (best effort)

**Steps**
- Run: `pytest -q`
- If optional deps are missing, ensure optional tests skip rather than fail.

---

### Task 20: Commit, tag, and push to `main`

**Steps**
- Commit with a single message (per preference) after all tasks, e.g.:
  - `feat: industrial UAD datasets + tiling + camera robustness`
- Create tag: `v0.3.0`
- Push: `git push origin main --tags`

