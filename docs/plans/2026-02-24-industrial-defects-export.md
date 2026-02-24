# Industrial Defects Export (Mask + Regions + ROI) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an industrial-grade “defects export” layer to `pyimgano` so inference can emit a binary defect mask + connected-component regions (bbox/area/centroid/region scores), with ROI gating and fully auditable pixel-threshold provenance, while keeping default behavior unchanged unless explicitly enabled.

**Architecture:** Implement a small, dependency-light core module for defect extraction from anomaly maps (ROI gating → binarize → binary postprocess → regions). Integrate it into `pyimgano-infer` behind a `--defects` flag and export/load its config via workbench `infer_config.json` for deploy-style usage. All outputs are in anomaly-map coordinate space by default.

**Tech Stack:** Python, NumPy, OpenCV (cv2), (optional) SciPy for hole filling; existing `pyimgano` CLI/workbench/reporting stack.

---

## Milestone 01 (Tasks 1–10): Defects core (library-first, no CLI behavior change)

### Task 1: Add ROI helpers (normalized rectangle ROI)

**Files:**
- Create: `pyimgano/defects/roi.py`
- Test: `tests/test_defects_roi.py`

**Step 1: Write the failing test**

Create `tests/test_defects_roi.py` with:

```python
import numpy as np
import pytest

from pyimgano.defects.roi import clamp_roi_xyxy_norm, roi_mask_from_xyxy_norm


def test_clamp_roi_xyxy_norm_orders_and_clamps():
    roi = clamp_roi_xyxy_norm([1.2, -0.1, 0.7, 0.3])
    assert roi == pytest.approx([0.7, 0.0, 1.0, 0.3])


def test_roi_mask_from_xyxy_norm_shape_and_coverage():
    mask = roi_mask_from_xyxy_norm((10, 20), [0.25, 0.2, 0.75, 0.8])
    assert mask.shape == (10, 20)
    assert mask.dtype == np.uint8
    # ROI should include some pixels and exclude some pixels.
    assert int(mask.sum()) > 0
    assert int(mask.sum()) < int(mask.size)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_defects_roi.py -v`  
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

Implement `pyimgano/defects/roi.py`:

- `clamp_roi_xyxy_norm(roi) -> [x1,y1,x2,y2]` (clamp to `[0,1]`, reorder if needed)
- `roi_mask_from_xyxy_norm(shape_hw, roi) -> uint8 mask` where 1=inside ROI

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_defects_roi.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/defects/roi.py tests/test_defects_roi.py
git commit -m "feat(defects): add normalized ROI helpers"
```

---

### Task 2: Add anomaly-map ROI gating + ROI stats helpers

**Files:**
- Create: `pyimgano/defects/map_ops.py`
- Test: `tests/test_defects_map_ops.py`

**Step 1: Write the failing test**

Create `tests/test_defects_map_ops.py`:

```python
import numpy as np

from pyimgano.defects.map_ops import apply_roi_to_map, compute_roi_stats


def test_apply_roi_to_map_zeros_outside_roi():
    m = np.ones((4, 4), dtype=np.float32)
    out = apply_roi_to_map(m, roi_xyxy_norm=[0.5, 0.0, 1.0, 1.0])
    assert out.shape == (4, 4)
    assert float(out[:, :2].max()) == 0.0
    assert float(out[:, 2:].min()) == 1.0


def test_compute_roi_stats_returns_max_and_mean():
    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    stats = compute_roi_stats(m, roi_xyxy_norm=[0.0, 0.0, 0.5, 1.0])
    assert set(stats) == {"max", "mean"}
    assert stats["max"] >= stats["mean"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_defects_map_ops.py -v`  
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `apply_roi_to_map(anomaly_map, roi_xyxy_norm|None) -> float32 map`
- `compute_roi_stats(anomaly_map, roi_xyxy_norm|None) -> {"max":..., "mean":...}`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_defects_map_ops.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/defects/map_ops.py tests/test_defects_map_ops.py
git commit -m "feat(defects): add ROI gating + ROI stats for anomaly maps"
```

---

### Task 3: Add binary mask generation from anomaly maps

**Files:**
- Create: `pyimgano/defects/mask.py`
- Test: `tests/test_defects_mask.py`

**Step 1: Write the failing test**

Create `tests/test_defects_mask.py`:

```python
import numpy as np

from pyimgano.defects.mask import anomaly_map_to_binary_mask


def test_anomaly_map_to_binary_mask_thresholds_to_uint8_255():
    m = np.asarray([[0.1, 0.9]], dtype=np.float32)
    mask = anomaly_map_to_binary_mask(m, pixel_threshold=0.5)
    assert mask.dtype == np.uint8
    assert mask.tolist() == [[0, 255]]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_defects_mask.py -v`  
Expected: FAIL

**Step 3: Write minimal implementation**

Implement `anomaly_map_to_binary_mask(map, pixel_threshold)` returning `{0,255}`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_defects_mask.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/defects/mask.py tests/test_defects_mask.py
git commit -m "feat(defects): add binary mask generation from anomaly maps"
```

---

### Task 4: Add binary mask postprocess (open/close + min-area)

**Files:**
- Create: `pyimgano/defects/binary_postprocess.py`
- Test: `tests/test_defects_binary_postprocess.py`

**Step 1: Write the failing test**

Create `tests/test_defects_binary_postprocess.py`:

```python
import numpy as np

from pyimgano.defects.binary_postprocess import postprocess_binary_mask


def test_postprocess_binary_mask_removes_small_components():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1, 1] = 255  # tiny
    mask[5:8, 5:8] = 255  # big
    out = postprocess_binary_mask(mask, min_area=4, open_ksize=0, close_ksize=0, fill_holes=False)
    assert int(out[1, 1]) == 0
    assert int(out[6, 6]) == 255
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_defects_binary_postprocess.py -v`  
Expected: FAIL

**Step 3: Write minimal implementation**

Implement `postprocess_binary_mask(mask_u8, min_area, open_ksize, close_ksize, fill_holes)` using:

- OpenCV morphology for open/close (ksize==0 → skip)
- connected components area filter
- optional hole filling (keep best-effort; can be added in Task 5)

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_defects_binary_postprocess.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/defects/binary_postprocess.py tests/test_defects_binary_postprocess.py
git commit -m "feat(defects): add binary mask postprocess (morphology + min-area)"
```

---

### Task 5: Add optional hole filling behavior (fill_holes=True)

**Files:**
- Modify: `pyimgano/defects/binary_postprocess.py`
- Test: `tests/test_defects_binary_postprocess.py`

**Step 1: Write the failing test**

Append to `tests/test_defects_binary_postprocess.py`:

```python
import numpy as np


def test_postprocess_binary_mask_fill_holes_fills_internal_hole():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    mask[4:6, 4:6] = 0  # hole
    out = postprocess_binary_mask(mask, min_area=0, open_ksize=0, close_ksize=0, fill_holes=True)
    assert int(out[5, 5]) == 255
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_defects_binary_postprocess.py -v`  
Expected: FAIL

**Step 3: Implement hole filling**

Implement hole filling using one of:

- `scipy.ndimage.binary_fill_holes` (preferred if available)
- OR an OpenCV flood-fill based alternative (no new deps)

**Step 4: Run test**

Run: `pytest tests/test_defects_binary_postprocess.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/defects/binary_postprocess.py tests/test_defects_binary_postprocess.py
git commit -m "feat(defects): support fill_holes in binary mask postprocess"
```

---

### Task 6: Add connected-component region extraction (bbox/area/centroid)

**Files:**
- Create: `pyimgano/defects/regions.py`
- Test: `tests/test_defects_regions.py`

**Step 1: Write the failing test**

Create `tests/test_defects_regions.py`:

```python
import numpy as np

from pyimgano.defects.regions import extract_regions_from_mask


def test_extract_regions_from_mask_finds_bbox_and_area():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    regions = extract_regions_from_mask(mask)
    assert len(regions) == 1
    r = regions[0]
    assert r["bbox_xyxy"] == [3, 2, 6, 4]
    assert r["area"] == int(3 * 4)
```

**Step 2: Run test**

Run: `pytest tests/test_defects_regions.py -v`  
Expected: FAIL

**Step 3: Implement region extraction**

Use `cv2.connectedComponentsWithStats` and compute:

- bbox in xyxy (inclusive or exclusive must match test; use inclusive coords in output)
- area
- centroid (from `centroids` output)

**Step 4: Run test**

Run: `pytest tests/test_defects_regions.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/defects/regions.py tests/test_defects_regions.py
git commit -m "feat(defects): extract connected-component regions from binary mask"
```

---

### Task 7: Add region scoring from anomaly maps (score_max/score_mean)

**Files:**
- Modify: `pyimgano/defects/regions.py`
- Test: `tests/test_defects_regions.py`

**Step 1: Write failing test**

Append to `tests/test_defects_regions.py`:

```python
import numpy as np


def test_extract_regions_from_mask_adds_scores_when_map_provided():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    amap = np.zeros((4, 4), dtype=np.float32)
    amap[2, 2] = 0.9
    regions = extract_regions_from_mask(mask, anomaly_map=amap)
    r = regions[0]
    assert r["score_max"] == 0.9
    assert 0.0 < r["score_mean"] <= r["score_max"]
```

**Step 2: Run test (fails)**

Run: `pytest tests/test_defects_regions.py -v`

**Step 3: Implement scoring**

If `anomaly_map` provided:

- compute max/mean over pixels where mask==255 for each component

**Step 4: Run test**

Run: `pytest tests/test_defects_regions.py -v`

**Step 5: Commit**

```bash
git add pyimgano/defects/regions.py tests/test_defects_regions.py
git commit -m "feat(defects): compute region score_max/score_mean from anomaly map"
```

---

### Task 8: Add an end-to-end defects extractor from anomaly maps

**Files:**
- Create: `pyimgano/defects/extract.py`
- Test: `tests/test_defects_extract.py`

**Step 1: Write failing test**

Create `tests/test_defects_extract.py`:

```python
import numpy as np

from pyimgano.defects.extract import extract_defects_from_anomaly_map


def test_extract_defects_from_anomaly_map_returns_mask_and_regions():
    amap = np.zeros((8, 8), dtype=np.float32)
    amap[2:5, 3:6] = 1.0
    out = extract_defects_from_anomaly_map(
        amap,
        pixel_threshold=0.5,
        roi_xyxy_norm=None,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        min_area=0,
        max_regions=None,
    )
    assert out["mask"].shape == (8, 8)
    assert len(out["regions"]) == 1
    assert out["space"]["type"] == "anomaly_map"
```

**Step 2: Run test**

Run: `pytest tests/test_defects_extract.py -v`  
Expected: FAIL

**Step 3: Implement minimal extractor**

Wire together:

- ROI gating
- binarize
- binary postprocess
- region extraction + scoring
- include `space` and `map_stats_roi`

**Step 4: Run test**

Run: `pytest tests/test_defects_extract.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/defects/extract.py tests/test_defects_extract.py
git commit -m "feat(defects): add end-to-end extraction from anomaly maps"
```

---

### Task 9: Add `pyimgano.defects` package init and public surface

**Files:**
- Create: `pyimgano/defects/__init__.py`
- Test: `tests/test_defects_public_api.py`

**Step 1: Write failing test**

Create `tests/test_defects_public_api.py`:

```python
from pyimgano.defects import extract_defects_from_anomaly_map


def test_defects_public_imports():
    assert callable(extract_defects_from_anomaly_map)
```

**Step 2: Run test**

Run: `pytest tests/test_defects_public_api.py -v`  
Expected: FAIL

**Step 3: Implement**

Add `pyimgano/defects/__init__.py` re-exporting key helpers.

**Step 4: Run test**

Run: `pytest tests/test_defects_public_api.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/defects/__init__.py tests/test_defects_public_api.py
git commit -m "feat(defects): add defects subpackage public API"
```

---

### Task 10: Tag Milestone 01 and push

**Files:**
- None

**Step 1: Verify a focused test subset**

Run: `pytest tests/test_defects_*.py -v`  
Expected: PASS

**Step 2: Tag**

```bash
git tag milestone-defects-01
git push
git push --tags
```

---

## Milestone 02 (Tasks 11–20): `pyimgano-infer` defects output (opt-in)

### Task 11: Add CLI flags for defects export (parser only)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_from_run_errors.py`

**Step 1: Write failing test**

Add a parser smoke assertion (e.g. call `infer_cli.main([...])` with `--help` or invalid args) to ensure flags are accepted.

**Step 2: Run test**

Run: `pytest tests/test_infer_cli_from_run_errors.py -v`  
Expected: FAIL until flags exist

**Step 3: Implement**

Add flags:

- `--defects`
- `--save-masks`
- `--mask-format`
- `--pixel-threshold`
- `--pixel-threshold-strategy`
- `--pixel-normal-quantile`
- `--defect-min-area`, `--defect-open-ksize`, `--defect-close-ksize`, `--defect-fill-holes`, `--defect-max-regions`
- `--roi-xyxy-norm`

**Step 4: Run test**

Run: `pytest tests/test_infer_cli_from_run_errors.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/infer_cli.py tests/test_infer_cli_from_run_errors.py
git commit -m "feat(infer-cli): add defects export flags"
```

---

### Task 12: Implement mask saving helpers (png/npy)

**Files:**
- Create: `pyimgano/defects/io.py`
- Test: `tests/test_defects_io.py`

**Step 1: Write failing test**

Create `tests/test_defects_io.py` verifying `save_binary_mask_png` writes a readable PNG and `save_binary_mask_npy` writes a `.npy`.

**Step 2: Run test (fails)**

Run: `pytest tests/test_defects_io.py -v`

**Step 3: Implement**

- `save_binary_mask(mask_u8, path, format)` plus helpers for png/npy.

**Step 4: Run test**

Run: `pytest tests/test_defects_io.py -v`

**Step 5: Commit**

```bash
git add pyimgano/defects/io.py tests/test_defects_io.py
git commit -m "feat(defects): add binary mask IO helpers (png/npy)"
```

---

### Task 13: Add pixel-threshold provenance helper (fixed vs quantile)

**Files:**
- Create: `pyimgano/defects/pixel_threshold.py`
- Test: `tests/test_defects_pixel_threshold.py`

**Step 1: Write failing test**

Add tests for:

- explicit threshold returns `(thr, provenance)` with `source="explicit"`
- quantile strategy returns provenance including `q` and `calibration_count`

**Step 2: Run test**

Run: `pytest tests/test_defects_pixel_threshold.py -v`

**Step 3: Implement**

Implement a small resolver that follows the priority order:

- explicit CLI
- infer-config
- train-dir normal-pixel quantile
- else error

**Step 4: Run test**

Run: `pytest tests/test_defects_pixel_threshold.py -v`

**Step 5: Commit**

```bash
git add pyimgano/defects/pixel_threshold.py tests/test_defects_pixel_threshold.py
git commit -m "feat(defects): add pixel threshold resolver + provenance"
```

---

### Task 14: Integrate defects extraction into `pyimgano-infer` output records

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_smoke.py`

**Step 1: Write failing test**

Extend `tests/test_infer_cli_smoke.py` with a dummy detector returning `get_anomaly_map`, then run:

- `--defects --save-masks ...`

Assert JSONL record includes:

- `defects.mask.path`
- `defects.regions` list
- `defects.pixel_threshold` + `defects.pixel_threshold_provenance`

**Step 2: Run test (fails)**

Run: `pytest tests/test_infer_cli_smoke.py -v`

**Step 3: Implement**

In `infer_cli.main()`:

- force `include_maps` if `--defects`
- resolve pixel threshold (explicit/infer_config/train_dir quantile)
- call `extract_defects_from_anomaly_map`
- save masks; add `defects` block to record

**Step 4: Run test**

Run: `pytest tests/test_infer_cli_smoke.py -v`

**Step 5: Commit**

```bash
git add pyimgano/infer_cli.py tests/test_infer_cli_smoke.py
git commit -m "feat(infer-cli): emit defects (mask + regions) in JSONL output"
```

---

### Task 15: Add ROI flag support in CLI and ensure it gates defects only

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_smoke.py`

**Step 1: Write failing test**

Add a test where anomaly_map has a hot spot outside ROI and assert:

- regions empty or mask zeros
- score/label unchanged

**Step 2: Run test**

Run: `pytest tests/test_infer_cli_smoke.py -v`

**Step 3: Implement**

Pass `roi_xyxy_norm` into defect extraction; do not alter `score/label`.

**Step 4: Run test**

Run: `pytest tests/test_infer_cli_smoke.py -v`

**Step 5: Commit**

```bash
git add pyimgano/infer_cli.py tests/test_infer_cli_smoke.py
git commit -m "feat(infer-cli): apply ROI gating to defects output only"
```

---

### Task 16: Add `--defect-max-regions` enforcement and deterministic sorting

**Files:**
- Modify: `pyimgano/defects/extract.py`
- Modify: `pyimgano/defects/regions.py`
- Test: `tests/test_defects_extract.py`

**Step 1: Write failing test**

Create many small regions and assert:

- output region count is limited
- ordering is stable (prefer `score_max` desc, then area desc, then id asc)

**Step 2: Run test**

Run: `pytest tests/test_defects_extract.py -v`

**Step 3: Implement**

Apply sorting + slicing.

**Step 4: Run test**

Run: `pytest tests/test_defects_extract.py -v`

**Step 5: Commit**

```bash
git add pyimgano/defects/extract.py pyimgano/defects/regions.py tests/test_defects_extract.py
git commit -m "feat(defects): add deterministic region sorting + max_regions limit"
```

---

### Task 17: Document `--defects` in CLI docs

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `docs/INDUSTRIAL_INFERENCE.md`

**Step 1: Write failing doc check**

Add a small doc audit test if one exists; otherwise update docs directly.

**Step 2: Update docs**

Add:

- schema snippet
- flags explanation
- ROI notes

**Step 3: Commit**

```bash
git add docs/CLI_REFERENCE.md docs/INDUSTRIAL_INFERENCE.md
git commit -m "docs: document defects export (mask + regions + ROI) for pyimgano-infer"
```

---

### Task 18: Add a dedicated infer-config mode smoke for defects

**Files:**
- Test: `tests/test_infer_cli_infer_config.py`

**Step 1: Write failing test**

Extend the infer-config tests to include a `defects` block and validate the CLI emits defects.

**Step 2: Run test**

Run: `pytest tests/test_infer_cli_infer_config.py -v`

**Step 3: Implement minimal glue**

Ensure infer-cli reads `payload["defects"]` and maps it to runtime settings.

**Step 4: Run test**

Run: `pytest tests/test_infer_cli_infer_config.py -v`

**Step 5: Commit**

```bash
git add tests/test_infer_cli_infer_config.py pyimgano/infer_cli.py
git commit -m "test(infer-cli): cover defects settings when using --infer-config"
```

---

### Task 19: Tag Milestone 02 and push

**Files:**
- None

**Step 1: Focused tests**

Run: `pytest tests/test_defects_*.py tests/test_infer_cli_*.py -v`

**Step 2: Tag**

```bash
git tag milestone-defects-02
git push
git push --tags
```

---

### Task 20: (Buffer) Fix any schema/back-compat edge found by tests

**Files:**
- Modify: as needed (keep changes minimal)

**Step 1: Add a failing regression test**

**Step 2: Implement minimal fix**

**Step 3: Run focused tests**

**Step 4: Commit**

```bash
git add -A
git commit -m "fix(defects): stabilize schema/back-compat edge case"
```

---

## Milestone 03 (Tasks 21–30): Workbench + infer-config alignment

### Task 21: Add `DefectsConfig` to workbench config schema

**Files:**
- Modify: `pyimgano/workbench/config.py`
- Test: `tests/test_workbench_config.py`

**Step 1: Write failing test**

Add config parsing coverage for:

- `defects.enabled`
- ROI, pixel threshold strategy, morph settings

**Step 2: Run test**

Run: `pytest tests/test_workbench_config.py -v`

**Step 3: Implement**

Add `DefectsConfig` dataclass and parse it in `WorkbenchConfig.from_dict`.

**Step 4: Run test**

Run: `pytest tests/test_workbench_config.py -v`

**Step 5: Commit**

```bash
git add pyimgano/workbench/config.py tests/test_workbench_config.py
git commit -m "feat(workbench): add defects config block to WorkbenchConfig"
```

---

### Task 22: Export defects block in `artifacts/infer_config.json`

**Files:**
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_export_infer_config.py`

**Step 1: Write failing test**

Extend export test to assert `payload["defects"]` exists and matches expected defaults.

**Step 2: Run test**

Run: `pytest tests/test_workbench_export_infer_config.py -v`

**Step 3: Implement**

Add `defects` to `build_infer_config_payload`.

**Step 4: Run test**

Run: `pytest tests/test_workbench_export_infer_config.py -v`

**Step 5: Commit**

```bash
git add pyimgano/workbench/runner.py tests/test_workbench_export_infer_config.py
git commit -m "feat(workbench): export defects settings in infer_config.json"
```

---

### Task 23: Propagate per-category defects overrides through `select_infer_category`

**Files:**
- Modify: `pyimgano/inference/config.py`
- Test: `tests/test_infer_config_loader.py`

**Step 1: Write failing test**

Add a per-category payload and assert `defects` or per-category pixel threshold settings propagate.

**Step 2: Run test**

Run: `pytest tests/test_infer_config_loader.py -v`

**Step 3: Implement**

Propagate `defects` similarly to threshold/checkpoint.

**Step 4: Run test**

Run: `pytest tests/test_infer_config_loader.py -v`

**Step 5: Commit**

```bash
git add pyimgano/inference/config.py tests/test_infer_config_loader.py
git commit -m "feat(infer-config): propagate defects settings during category selection"
```

---

### Task 24: Ensure infer-cli applies infer-config defects defaults (no extra flags required)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_infer_config.py`

**Step 1: Write failing test**

Run infer-cli with `--infer-config` containing `defects.enabled=true` and assert defects emitted without `--defects`.

**Step 2: Run test**

Run: `pytest tests/test_infer_cli_infer_config.py -v`

**Step 3: Implement**

If infer-config has `defects.enabled`, treat it as enabling defects output.

**Step 4: Run test**

Run: `pytest tests/test_infer_cli_infer_config.py -v`

**Step 5: Commit**

```bash
git add pyimgano/infer_cli.py tests/test_infer_cli_infer_config.py
git commit -m "feat(infer-cli): enable defects output via infer-config defaults"
```

---

### Task 25: Add docs section “ship infer_config + masks” for deployment

**Files:**
- Modify: `docs/WORKBENCH.md`

**Step 1: Update docs**

Add a short “ship artifacts” section:

- `infer_config.json`
- checkpoint (optional)
- maps + masks directories

**Step 2: Commit**

```bash
git add docs/WORKBENCH.md
git commit -m "docs(workbench): describe defects deployment artifacts"
```

---

### Task 26: Add an example workbench config with defects enabled + ROI

**Files:**
- Create: `examples/configs/industrial_adapt_defects_roi.json`

**Step 1: Add config**

Include:

- `defects.enabled=true`
- `defects.roi_xyxy_norm`
- reasonable morph defaults

**Step 2: Commit**

```bash
git add examples/configs/industrial_adapt_defects_roi.json
git commit -m "docs(examples): add workbench config template for defects + ROI"
```

---

### Task 27: Tag Milestone 03 and push

**Step 1: Focused tests**

Run: `pytest tests/test_workbench_*infer_config*.py tests/test_infer_config_loader.py -v`

**Step 2: Tag**

```bash
git tag milestone-defects-03
git push
git push --tags
```

---

### Task 28: (Buffer) Harden error messages for missing maps/threshold

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_from_run_errors.py`

**Step 1: Add failing test**

Ensure `--defects` on a detector without maps errors with actionable guidance.

**Step 2: Implement**

Improve error text.

**Step 3: Run test + commit**

---

### Task 29: (Buffer) Add schema notes for defects in docs

**Files:**
- Modify: `docs/CLI_REFERENCE.md`

---

### Task 30: (Buffer) Minor refactor to keep CLI code readable

**Files:**
- Modify: `pyimgano/infer_cli.py`

---

## Milestone 04 (Tasks 31–40): Industrial polish + release

### Task 31: Add JSON schema-style example to `docs/INDUSTRIAL_INFERENCE.md`

**Files:**
- Modify: `docs/INDUSTRIAL_INFERENCE.md`

**Commit:** `docs: add defects JSONL schema example`

---

### Task 32: Add a small “defects quickstart” in README

**Files:**
- Modify: `README.md`

**Commit:** `docs: add defects export quickstart`

---

### Task 33: Add `threshold_provenance` + `pixel_threshold_provenance` stability test for infer-config roundtrip

**Files:**
- Test: `tests/test_infer_cli_infer_config.py`

---

### Task 34: Ensure mask output encoding is stable (0/255) and documented

**Files:**
- Modify: `pyimgano/defects/io.py`
- Modify: `docs/CLI_REFERENCE.md`

---

### Task 35: Add `defects.space` + mapping note for consumers

**Files:**
- Modify: `pyimgano/defects/extract.py`
- Modify: `docs/INDUSTRIAL_INFERENCE.md`

---

### Task 36: Add release notes to changelog

**Files:**
- Modify: `CHANGELOG.md`

---

### Task 37: Bump version (patch) and tag release

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`

---

### Task 38: Tag Milestone 04 and push

```bash
git tag milestone-defects-04
git push
git push --tags
```

---

### Task 39: Create final release tag (e.g. `v0.6.7`)

```bash
git tag v0.6.7
git push --tags
```

---

### Task 40: Post-release doc cleanup (optional)

**Files:**
- Modify: `docs/WORKBENCH.md`
- Modify: `docs/CLI_REFERENCE.md`

