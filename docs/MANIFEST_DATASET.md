# Manifest Dataset (JSONL)

Industrial anomaly detection projects often can’t (or shouldn’t) force all data
into MVTec/VisA folder layouts. A **manifest dataset** lets you describe samples
with a single JSONL file and still use the `pyimgano` workbench recipes to
produce standard artifacts.

This is designed for **algorithm engineers** who want:

- one file that can span multiple categories
- deterministic, leakage-safe splitting (optional `group_id`)
- optional pixel masks for localization metrics
- portable relative paths

---

## JSONL Schema (v1)

One JSON object per line.

### Required fields

- `image_path: str`
- `category: str`

### Optional fields

- `split: "train" | "val" | "test"`
  - If omitted, auto-split applies (see Split Policy).
- `label: 0 | 1`
  - If `split=="train"` or `split=="val"` and `label` is omitted → treated as `0`.
  - If `split=="test"` and `label` is omitted → **invalid** (explicit test requires label).
- `mask_path: str`
  - Optional pixel mask path (grayscale). Used for pixel-level metrics.
- `group_id: str`
  - Optional group identifier (part/lot/video clip/etc.). If present, we split
    whole groups to avoid leakage.
- `meta: object`
  - Optional metadata object; treated as opaque user payload.

### Example

```jsonl
{"image_path":"cats/bottle/img_0001.png","category":"bottle","split":"train","group_id":"lotA"}
{"image_path":"cats/bottle/img_0101.png","category":"bottle","split":"test","label":0,"group_id":"lotB"}
{"image_path":"cats/bottle/img_0201.png","category":"bottle","split":"test","label":1,"mask_path":"masks/bottle/img_0201.png","group_id":"lotB","meta":{"line":"L1"}}
{"image_path":"/abs/path/cats/cable/img_0005.png","category":"cable","label":1,"mask_path":"/abs/path/masks/cable/img_0005.png"}
```

Notes:

- Blank lines are ignored.
- Lines starting with `#` are treated as comments and ignored.

---

## Path Resolution

If `image_path` / `mask_path` is absolute, it is used as-is.

If it is relative, `pyimgano` resolves in this order:

1) relative to the manifest file directory  
2) if not found, relative to `dataset.root` (fallback)

This makes manifests portable across machines (repo-relative) while still
supporting legacy “root-relative” paths.

---

## Split Policy (Auto Split)

If a record provides `split`, it is respected (with validation).

If `split` is missing, the default policy is **benchmark-style**:

- anomalies (`label==1`) → `test`
- normals (`label==0` or missing) → split between `train` and `test` by
  `test_normal_fraction`

### Group-aware splitting (`group_id`)

If `group_id` is present:

- groups are indivisible units (no leakage)
- if any record in a group is anomalous (`label==1`), the whole group goes to `test`
- remaining all-normal groups are split deterministically

### Scope

When `dataset.category=="all"`, auto-splitting is done **per-category**
(`scope="category"`). This keeps metrics comparable with benchmark conventions.

---

## Masks + Pixel Metrics

Pixel metrics (pixel AUROC / AUPRO) require masks.

- If at least one mask exists and **all anomaly test samples have masks**, masks
  are loaded and pixel metrics can be computed.
- If any anomaly test sample is missing a mask:
  - pixel metrics are skipped
  - the workbench report includes an explicit reason

Masks are loaded as grayscale, resized to `dataset.resize` using nearest-neighbor,
and binarized with threshold `>127`.

---

## Workbench Config Example

```json
{
  "recipe": "industrial-adapt",
  "seed": 123,
  "dataset": {
    "name": "manifest",
    "root": "/path/to/fallback_root",
    "manifest_path": "/path/to/manifest.jsonl",
    "category": "all",
    "resize": [256, 256],
    "input_mode": "paths",
    "split_policy": {
      "mode": "benchmark",
      "scope": "category",
      "seed": 123,
      "test_normal_fraction": 0.2
    }
  },
  "model": {
    "name": "vision_patchcore",
    "device": "cuda",
    "preset": "industrial-fast",
    "pretrained": true,
    "contamination": 0.1
  },
  "output": {
    "save_run": true,
    "per_image_jsonl": true
  }
}
```

Run:

```bash
pyimgano-train --config cfg.json
```

---

## Generate a Manifest (Custom Layout)

If your data already matches the built-in `custom` dataset layout, you can
generate a manifest automatically:

```bash
pyimgano-manifest --root /path/to/custom_dataset --out manifest.jsonl --include-masks
```

This scans:

- `train/normal/*`
- `test/normal/*`
- `test/anomaly/*`
- optional masks: `ground_truth/anomaly/<stem>_mask.png`

and writes a stable, sorted JSONL output.
