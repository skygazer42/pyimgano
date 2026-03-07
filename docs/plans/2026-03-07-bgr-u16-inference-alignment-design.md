# BGR/u16 Inference Alignment (Algorithm UX + Data Processing) — Design

**Date:** 2026-03-07  
**Status:** Approved (user selected approach A)  
**Owner:** @codex

## Problem Statement

`pyimgano`’s industrial inference contract is:

- in-memory images must declare an explicit `ImageFormat`
- images are canonicalized to **RGB / uint8 / HWC**

This is documented in `docs/INDUSTRIAL_INFERENCE.md` and implemented in
`pyimgano.inputs.normalize_numpy_image`.

However, there are three real-world gaps that hurt “PyOD-like” UX and deployment robustness:

1. **Missing `u16_max` plumbing in inference API**
   - `normalize_numpy_image(..., u16_max=...)` exists (supports 12-bit sensors in `uint16`)
   - but `infer()/infer_iter()/calibrate_threshold()` don’t accept `u16_max`, forcing users to
     pre-normalize manually.

2. **Deployment wrappers decode paths in 8-bit only**
   - `TiledDetector` loads with `cv2.imread(str(path))` (8-bit default) and therefore drops 16-bit.
   - `PreprocessingDetector` loads with `PIL.Image.open(...).convert("RGB")`, forcing 8-bit.
   - These wrappers are used in CLI/workbench flows when tiling or illumination/contrast
     preprocessing is enabled.

3. **The common industrial numpy input is OpenCV-style `bgr_u8_hwc`**
   - Calling `infer(..., input_format=...)` works, but the experience is not “smooth”:
     users must always pass `input_format=ImageFormat.BGR_U8_HWC`.

## Goals

1. Improve **algorithm experience** for common OpenCV pipelines:
   - provide a BGR-first convenience API without weakening the strict input contract.
2. Improve **data processing** for industrial bit-depth inputs:
   - preserve 16-bit data during path decoding for tiling/preprocessing wrappers.
3. Keep changes **additive** and **low-risk**:
   - preserve the “explicit input_format, no guessing” rule.
   - do not change core detector semantics.

## Non-Goals

- Add automatic guessing of numpy image format.
- Change default color semantics of existing classical feature extractors in this change-set.
- Add new heavy dependencies; keep OpenCV/Pillow optional and lazily imported.

## Proposed Changes

### A) Inference API: thread `u16_max` through normalization

Update:

- `pyimgano.inference.api._normalize_inputs(...)`
- `calibrate_threshold(...)`
- `infer(...)`
- `infer_iter(...)`

Add parameter:

- `u16_max: int | None = None`

Behavior:

- for numpy inputs, call:
  - `normalize_numpy_image(item, input_format=fmt, u16_max=u16_max)`
- for path inputs, unchanged (paths forwarded to detectors as-is).

### B) Convenience API: BGR-first wrappers

Add additive helpers:

- `infer_bgr(detector, inputs, **kwargs)` → calls `infer(..., input_format="bgr_u8_hwc")`
- `infer_iter_bgr(...)`
- `calibrate_threshold_bgr(...)`

Rationale:

- keeps strict explicit formats while making the most common real-world case ergonomic.

### C) Deployment wrappers: robust path decoding with `IMREAD_UNCHANGED`

Introduce a small internal helper used by wrappers that must decode paths into numpy:

- decode with OpenCV `cv2.imread(path, cv2.IMREAD_UNCHANGED)`
- infer declared `ImageFormat` based on `(dtype, ndim, channels)`:
  - `(H,W), uint8` → `GRAY_U8_HW`
  - `(H,W), uint16` → `GRAY_U16_HW`
  - `(H,W,3), uint8` → `BGR_U8_HWC`
  - `(H,W,3), uint16` → `BGR_U16_HWC`
  - `(H,W,4)` → drop alpha to 3 channels then treat as BGR
- canonicalize using `normalize_numpy_image(..., u16_max=u16_max)` to RGB/u8/HWC

Apply to:

- `pyimgano.inference.tiling.TiledDetector` path loader
- `pyimgano.inference.preprocessing.PreprocessingDetector` path loader

### D) CLI: add `--u16-max` for wrapper decode

Add an optional CLI arg:

- `pyimgano-infer --u16-max 4095`

Used only when the CLI itself decodes images (tiling/preprocessing wrappers).

Default behavior remains unchanged:

- `None` → `normalize_numpy_image` uses `u16_max=65535`.

## Error Handling

- If OpenCV is missing and a wrapper needs it (tiling always, preprocessing when `--u16-max` is
  provided or when a non-8-bit image is encountered), raise a clear `ImportError` with install hint.
- If an image has unsupported dtype/shape, raise a clear `ValueError` describing the observed
  `(shape, dtype)`.

## Testing Strategy

Add unit tests that:

1. verify inference API normalization respects `u16_max` for `GRAY_U16_HW`.
2. verify `infer_bgr()` performs the BGR→RGB channel swap (observable via a dummy detector).
3. verify `TiledDetector` and `PreprocessingDetector` can load a 16-bit grayscale PNG path and
   normalize it to canonical RGB/u8/HWC with `u16_max=4095` (score becomes 255 for max pixels).

## Rollout Notes

- This is an additive change; existing call sites keep working.
- A future follow-up can address classical feature extractor color convention consistency
  (path vs numpy) without entangling this high-ROI industrial fix.

