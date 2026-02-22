# Robustness Benchmark (Clean + Industrial Drift Corruptions)

This guide describes the **deploy-style robustness benchmark** introduced in `pyimgano`.

It evaluates an anomaly detector on:
- a **clean** test set, and
- the same test set under a set of **deterministic corruptions** (industrial drift),

while keeping a **single fixed pixel threshold** for the entire run (calibrated once from
normal training images).

---

## What it measures

### Image-level metrics
You always get standard image-level metrics (AUROC/AP/F1/etc.) from `evaluate_detector(...)`.

### Pixel SegF1 + background FPR (VAND-style)
If you enable pixel SegF1 (`--pixel-segf1`, default in the robustness CLI):

- A single pixel threshold `t` is calibrated from normal pixels using a quantile:
  - `t = quantile(scores_normal_pixels, q)`
- The same `t` is used for:
  - clean evaluation
  - every corruption × severity
- Reported pixel metrics include:
  - `pixel_segf1` (pixel-level Segmentation F1)
  - `bg_fpr` (background false-positive rate)

This matches a production constraint: **you don't get to retune the threshold per condition**.

### Latency
The report includes a best-effort `latency_ms_per_image` per condition (measured around
`decision_function()` + optional anomaly map extraction).

---

## Quickstart: CLI (recommended)

Run a robustness benchmark on MVTec AD:

```bash
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --preset industrial-balanced \
  --device cuda \
  --pixel-normal-quantile 0.999 \
  --pixel-calibration-fraction 0.2 \
  --corruptions lighting,jpeg,blur,glare,geo_jitter \
  --severities 1 2 3 4 5 \
  --output runs/robust_mvtec_bottle_patchcore.json
```

For a quick smoke run:

```bash
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --device cpu \
  --no-pretrained \
  --corruptions lighting \
  --severities 1 \
  --limit-train 32 \
  --limit-test 64
```

Notes:
- Datasets without segmentation masks (e.g. some exports of BTAD) should run with:
  - `--no-pixel-segf1`
- Some detectors only support **path inputs**; the robustness CLI currently feeds **numpy RGB**
  images to enable deterministic corruption application. Use vision models that accept
  `RGB/u8/HWC` numpy images (e.g. PatchCore / AnomalyDINO / SuperAD-style patch-kNN models).

---

## Corruptions included

The default corruption set is:

- `lighting`: exposure / contrast / gamma + mild per-channel gain drift
- `jpeg`: JPEG encode/decode artifacts (blocking/ringing)
- `blur`: Gaussian blur
- `glare`: synthetic specular/glare blobs
- `geo_jitter`: small affine warp (image + mask are warped consistently)

Each corruption is deterministic for a given `--seed`, name, and severity.

---

## Output schema

The CLI prints a JSON object (or saves via `--output`) with a structure like:

```json
{
  "dataset": "...",
  "category": "...",
  "model": "...",
  "robustness": {
    "pixel_threshold_strategy": "normal_pixel_quantile",
    "pixel_normal_quantile": 0.999,
    "clean": {
      "latency_ms_per_image": 12.3,
      "results": { "...": "evaluate_detector() payload" }
    },
    "corruptions": {
      "lighting": {
        "severity_1": { "...": "same schema as clean" },
        "severity_2": { "...": "..." }
      }
    }
  }
}
```

---

## Tuning tips

- `--pixel-normal-quantile` controls the tradeoff:
  - higher quantile → fewer background false positives (lower `bg_fpr`), but may miss small defects
  - lower quantile → more sensitive, but more false positives
- If your production normals are noisy, consider evaluating with `vision_softpatch` and the
  `industrial-balanced` preset first.

