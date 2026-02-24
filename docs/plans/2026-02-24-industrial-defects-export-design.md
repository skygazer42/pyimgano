# PyImgAno Industrial Defects Export (Mask + Regions) — Design

**Date:** 2026-02-24  
**Status:** Approved (user: image-level + pixel-level; export mask + regions; ROI support)  

## Context

`pyimgano` already supports:

- image-level scoring (`decision_function`) with an industrial threshold calibration default
- optional anomaly maps (`get_anomaly_map` / `predict_anomaly_map`)
- workbench artifacts + deploy-style `artifacts/infer_config.json`

However, industrial users typically need more than a heatmap file:

- a **binary defect mask** (for overlay + QC rules)
- **instances/regions** (bbox/area/centroid) for downstream alarm/analytics
- **ROI** gating so fixtures/background do not dominate false positives
- clear **provenance** for both score thresholds and pixel thresholds

This design adds an explicit, auditable “defects export” layer that builds on
existing anomaly-map infrastructure without changing default behavior.

## Goals

### 1) `pyimgano-infer` can export industrial defect structures

When enabled, inference output should include:

- binary mask artifact (PNG/Npy) + metadata
- regions/instances extracted from connected components
- pixel threshold provenance (how mask threshold was chosen)

### 2) Pixel threshold resolution is stable and auditable

Pixel thresholds must be explicit about:

- strategy used (fixed vs quantile calibration)
- calibration source (explicit value, infer-config, or train-dir quantile)
- calibration sample counts

### 3) ROI support without breaking existing score semantics

ROI should:

- default to affecting **defects** output only (mask/regions)
- not silently change image-level `score`/`label` behavior unless explicitly requested
- be defined in a resolution-independent way for deployment

### 4) Keep the default behavior stable

- Without the new `--defects` flag, `pyimgano-infer` behaves as before.
- No new mandatory dependencies.
- All changes should be additive and guarded by flags/config.

## Non-goals

- No time-series/video tracking (single-frame contract only).
- No full COCO instance segmentation export.
- No large “model zoo expansion” as a primary objective for this milestone.
- No HTTP service layer.

## Proposed Interfaces

### 1) `pyimgano-infer` output schema (JSONL)

Each JSONL line continues to represent a single input.

Existing fields (unchanged):

- `index`, `input`, `score`
- `label` (0/1) when `threshold_` is available
- `anomaly_map.path/shape/dtype` when maps are requested and exported

New `defects` block (only when enabled):

```json
{
  "defects": {
    "space": { "type": "anomaly_map", "shape": [256, 256] },
    "pixel_threshold": 0.73,
    "pixel_threshold_provenance": { "method": "...", "source": "...", "...": "..." },
    "mask": { "path": "masks/000000_x.png", "shape": [256, 256], "dtype": "uint8", "encoding": "png" },
    "regions": [
      {
        "id": 1,
        "bbox_xyxy": [12, 34, 80, 120],
        "area": 1532,
        "centroid_xy": [45.2, 77.8],
        "score_max": 0.98,
        "score_mean": 0.41
      }
    ],
    "map_stats_roi": { "max": 0.98, "mean": 0.12 }
  }
}
```

Coordinate space:

- All mask/region coordinates are in the anomaly-map grid (`anomaly_map.shape`).
- ROI is applied in that same space to avoid ambiguous mapping to original image sizes.

### 2) Pixel threshold resolution rules

When `--defects` is enabled, a pixel threshold is required. Resolution order:

1) CLI explicit value: `--pixel-threshold FLOAT` (method: fixed, source: explicit)
2) infer-config value: `infer_config["defects"]["pixel_threshold"]` (source: infer_config)
3) Auto-calibrate from `--train-dir` using `normal_pixel_quantile` (method: normal_pixel_quantile)
4) Otherwise: error with actionable guidance (no silent fallback)

Auto-calibration default:

- `q=0.999` on normal pixels in calibration maps.
- Calibration maps use the same tiling and postprocess settings as inference.

### 3) ROI definition and behavior

ROI definition (phase 1):

- Rectangle ROI in normalized coordinates:
  - `roi_xyxy_norm = [x1, y1, x2, y2]` in `[0,1]`
  - Interpreted relative to the anomaly-map width/height

Behavior:

- ROI affects `defects.mask` and `defects.regions` by zeroing anomaly-map pixels outside ROI.
- ROI does not change image-level `score/label` by default.

### 4) CLI flags (minimal but industrial)

New `pyimgano-infer` flags:

- `--defects` (enables defects export; implies maps are needed)
- `--save-masks DIR`
- `--mask-format png|npy` (default: png)
- `--pixel-threshold FLOAT` (fixed threshold)
- `--pixel-threshold-strategy normal_pixel_quantile|fixed|infer_config`
- `--pixel-normal-quantile FLOAT` (default: 0.999)
- `--defect-min-area INT` (default: 0)
- `--defect-open-ksize INT` (default: 0)
- `--defect-close-ksize INT` (default: 0)
- `--defect-fill-holes` (default: false)
- `--defect-max-regions INT` (default: unlimited)
- ROI:
  - `--roi-xyxy-norm x1 y1 x2 y2` (optional)

### 5) `infer_config.json` contract (deployment)

Extend `artifacts/infer_config.json` with an optional `defects` block:

```json
{
  "defects": {
    "enabled": true,
    "pixel_threshold": null,
    "pixel_threshold_strategy": "normal_pixel_quantile",
    "pixel_normal_quantile": 0.999,
    "mask_format": "png",
    "roi_xyxy_norm": null,
    "min_area": 0,
    "open_ksize": 0,
    "close_ksize": 0,
    "fill_holes": false,
    "max_regions": null
  }
}
```

`pyimgano-train --export-infer-config` should export this block as part of the
deployable inference payload, and `pyimgano-infer --infer-config` should apply it.

## Implementation Notes

### 1) New module for defect extraction

Add a small, dependency-light module that:

- converts anomaly maps → masks
- applies binary postprocess (optional)
- extracts connected components and region stats
- applies ROI gating

This keeps CLI logic thin and testable.

### 2) Provenance

Add `pixel_threshold_provenance` alongside existing `threshold_provenance`.

## Testing Strategy

Focus on deterministic tests using dummy detectors returning controlled maps:

- ROI gating correctness
- pixel threshold resolution order
- mask morphology + min-area behavior
- region extraction geometry/stats correctness
- CLI output schema contains required keys when `--defects` is used
- infer-config export/import roundtrip for defects config

## Acceptance Criteria

1) `pyimgano-infer --defects` writes mask artifacts and emits regions.
2) Pixel threshold is always present with provenance, or the CLI errors clearly.
3) ROI affects mask/regions only (default), and is stable across deployments.
4) `pyimgano-train --export-infer-config` exports a minimal deployable defects config.
5) All changes are additive and covered by unit tests.

