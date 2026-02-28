# Industrial MVP Loop (Synthesize → Infer → Defects)

This page describes a stable, dependency-light “industrial MVP loop” that works well for
real factory workflows:

1) start from a folder of normal images
2) (optionally) synthesize anomalies + masks and produce a JSONL manifest
3) run inference with a pixel-map baseline
4) export defects artifacts (binary mask + connected-component regions + overlays)

Key ideas:
- **Manifests** (`.jsonl`) are the interchange format across synthesis, evaluation, and export.
- **Pixel anomaly maps** are first-class outputs and can be turned into “defects” for downstream systems.
- Everything is **safe-by-default**: no implicit weight downloads; optional backends stay optional.

---

## 1) Synthesize a tiny dataset (optional but recommended)

Input: a folder of normal images (one product, one station).

Run:

```bash
pyimgano-synthesize \
  --in-dir /path/to/normals \
  --out-root /tmp/pyimgano_synth \
  --category demo \
  --preset tape \
  --blend alpha \
  --alpha 0.9 \
  --seed 0 \
  --n-train 20 \
  --n-test-normal 10 \
  --n-test-anomaly 10 \
  --manifest /tmp/pyimgano_synth/manifest.jsonl \
  --absolute-paths
```

Output:
- `/tmp/pyimgano_synth/train/normal/*.png`
- `/tmp/pyimgano_synth/test/{normal,anomaly}/*.png`
- `/tmp/pyimgano_synth/ground_truth/anomaly/*_mask.png` (when available for the chosen preset/blend)
- `/tmp/pyimgano_synth/manifest.jsonl`

---

## 2) Infer with a pixel-map baseline and export defects

Use a template-style SSIM pixel-map model as a stable baseline and turn its maps into defects.

Run:

```bash
pyimgano-infer \
  --model ssim_template_map \
  --train-dir /tmp/pyimgano_synth/train/normal \
  --input /tmp/pyimgano_synth/test \
  --defects \
  --save-jsonl /tmp/pyimgano_synth/out.jsonl \
  --save-masks /tmp/pyimgano_synth/masks \
  --save-overlays /tmp/pyimgano_synth/overlays
```

Notes:
- `--defects` implies anomaly maps internally (you do not need to add `--include-maps`).
- Pixel thresholding defaults to a **normal-pixel quantile** calibrated from `--train-dir`.
- The JSONL records include defects provenance so downstream users can track how the mask was produced.

---

## 3) What to look at

- The JSONL file: `out.jsonl`
  - image-level score/label (when thresholding is available)
  - `defects.mask.path` (saved binary mask artifact)
  - `defects.regions` (connected components + stats)
- Overlays: `overlays/*.png` for quick FP debugging and qualitative triage.

