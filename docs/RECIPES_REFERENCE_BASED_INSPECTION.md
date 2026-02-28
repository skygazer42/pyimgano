# Recipes: Reference-Based Inspection (Golden Template)

Many industrial inspection lines have an explicit **golden reference** image for
each part (or each view). Instead of learning a template from training data, you
can compare each query image directly to its reference.

This is common in settings with:
- controlled parts / fixed viewpoints
- frequent product changeovers
- limited anomaly samples
- alignment stress (lighting, small shifts, slight warps)

---

## Key idea

`query image` + `reference image` → **pixel anomaly map** → defects export

In `pyimgano`, reference-based detectors typically:
- require a `reference_dir` containing images
- match each query image to a reference image by **basename** (filename)

Example:

```
reference_dir/
  IMG_0001.png
  IMG_0002.png

query_dir/
  IMG_0001.png
  IMG_0002.png
```

---

## Recipe 1: Patch distance map (torchvision features)

Model: `vision_ref_patch_distance_map`

This compares query vs reference feature maps and produces an anomaly map from
per-patch distances.

CLI (inference):

```bash
pyimgano-infer \
  --model vision_ref_patch_distance_map \
  --reference-dir /path/to/reference_dir \
  --model-kwargs '{"backbone":"resnet18","pretrained":false,"node":"layer4","metric":"l2","image_size":224,"device":"cpu"}' \
  --input /path/to/query_dir \
  --include-maps --defects --defects-preset industrial-defects-fp40 \
  --save-jsonl ./out/results.jsonl \
  --save-maps ./out/maps
```

Notes:
- Keep `pretrained=false` to avoid implicit downloads.
- For high-resolution parts, enable tiling via model kwargs:
  - `tile_size`, `tile_stride`, `tile_map_reduce`

---

## Practical tips

- If filenames are not unique in `reference_dir`, reference lookup is ambiguous.
  Rename or reorganize the reference set to make basenames unique.
- If you suspect misalignment, start with robust postprocess + defects preset:
  smoothing + hysteresis + shape filters reduce false positives.

