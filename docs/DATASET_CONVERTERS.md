# Dataset Converters (to JSONL manifest)

This page documents the built-in **dataset → manifest** converters.

Goal: make it easy to move from “whatever on-disk layout you have” to a stable
JSONL manifest that the rest of the tooling can consume (`pyimgano-benchmark`,
workbench configs, synthesis-from-manifest, etc.).

Discover available converters:

```bash
pyimgano-datasets list
pyimgano-datasets list --json
```

---

## `pyimgano-manifest` (converter CLI)

Common flags:
- `--dataset NAME` converter name (`custom`, `mvtec_ad2`, `real_iad`, `rad`, …)
- `--root DIR` dataset root directory
- `--category NAME` category label to stamp into the manifest (required for some converters)
- `--out PATH` output JSONL path
- `--absolute-paths` write absolute paths (portable across working dirs)
- `--include-masks` include `mask_path` when ground-truth masks exist

---

## Supported converters

### `custom`
Built-in layout:

```
root/
  train/normal/*
  test/normal/*
  test/anomaly/*
  ground_truth/anomaly/*_mask.png   (optional)
```

Example:

```bash
pyimgano-manifest \
  --dataset custom \
  --root /path/to/custom_dataset \
  --category demo \
  --out /tmp/custom_manifest.jsonl \
  --include-masks
```

### `mvtec_ad2`
MVTec AD 2 category converter (paths-first).

Example:

```bash
pyimgano-manifest \
  --dataset mvtec_ad2 \
  --root /path/to/mvtec_ad2_root \
  --category bottle \
  --out /tmp/mvtec_ad2_bottle.jsonl \
  --include-masks
```

### `real_iad`
Best-effort converter for Real-IAD-like layouts (study-friendly). Layouts can vary;
this converter supports a small set of common “custom-like” and “mvtec-like” patterns.

Example:

```bash
pyimgano-manifest \
  --dataset real_iad \
  --root /path/to/real_iad_root \
  --category real_iad \
  --out /tmp/real_iad.jsonl \
  --include-masks
```

### `rad`
Best-effort converter for RAD-like layouts. Emits multi-view metadata when encoded in
directory names (e.g. `view_0/`, `cam_1/`, `cond_day/`).

Example:

```bash
pyimgano-manifest \
  --dataset rad \
  --root /path/to/rad_root \
  --category rad \
  --out /tmp/rad.jsonl \
  --include-masks
```

---

## Validation

Validate an existing manifest:

```bash
pyimgano-manifest --validate --manifest /path/to/manifest.jsonl --root /path/to/root_fallback
```

