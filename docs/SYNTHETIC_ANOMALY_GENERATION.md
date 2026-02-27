# Synthetic Anomaly Generation (Industrial Cookbook)

This page documents the **built-in synthetic anomaly pipeline** in `pyimgano`.

Why synthesize anomalies?
- Bootstrapping when real defect samples are scarce
- Stress-testing model robustness (domain shifts, textures, lighting)
- Generating pixel masks for quick segmentation evaluation smoke tests

Important notes:
- Synthetic defects are not a substitute for real data.
- Prefer small, controlled synthesis that matches your production failure modes.

---

## 1) Python API: `AnomalySynthesizer`

```python
import numpy as np
from pyimgano.synthesis import AnomalySynthesizer, SynthSpec

# image_u8: uint8 numpy array (H,W,3) in BGR/RGB, or (H,W) grayscale
image_u8 = np.zeros((256, 256, 3), dtype=np.uint8)

syn = AnomalySynthesizer(
    SynthSpec(
        preset="scratch",   # scratch|stain|pit|glare|rust|oil|crack
        blend="alpha",      # alpha|poisson
        alpha=0.9,
        probability=1.0,
    )
)

res = syn(image_u8, seed=42)
out_img = res.image_u8
mask = res.mask_u8        # (H,W) uint8 {0,255}
label = res.label         # 0/1
meta = res.meta
```

## 2) Presets

Current built-in presets:
- `scratch`: thin dark streaks (common for surface damage)
- `stain`: organic blotches (Perlin-style masks)
- `pit`: small dark pits (point-like defects)
- `glare`: bright specular blobs / exposure flare
- `rust`: corrosion / rust-like organic blobs with speckle texture
- `oil`: darker organic oil stains
- `crack`: thin fracture-like lines

Get preset names:

```python
from pyimgano.synthesis import get_preset_names
print(get_preset_names())
```

---

## 3) CLI: `pyimgano-synthesize`

Generate a **tiny dataset** in the built-in `custom` layout, including a JSONL manifest:

```bash
pyimgano-synthesize \
  --in-dir /path/to/normal_images \
  --out-root ./out_synth_dataset \
  --category synthetic_demo \
  --preset scratch \
  --blend alpha \
  --alpha 0.9 \
  --n-train 50 \
  --n-test-normal 20 \
  --n-test-anomaly 20 \
  --seed 0
```

### Preview Mode (Grid)

To preview a preset without writing a dataset:

```bash
pyimgano-synthesize \
  --in-dir /path/to/normal_images \
  --out-root ./out_synth_dataset \
  --preset rust \
  --preview \
  --preview-out ./out_synth_dataset/preview.png \
  --preview-n 16 \
  --preview-cols 4 \
  --seed 0
```

### From-Manifest Mode (Augment Existing Dataset)

To synthesize anomalies from a manifest (source normals) and write a new
`custom`-layout dataset under `--out-root`:

```bash
pyimgano-synthesize \
  --from-manifest ./manifest.jsonl \
  --from-manifest-category bottle \
  --from-manifest-split train \
  --from-manifest-n 200 \
  --out-root ./out_augmented \
  --manifest ./out_augmented/manifest.jsonl \
  --preset scratch \
  --seed 0
```

Output layout:

```text
out_synth_dataset/
  train/normal/*.png|*.jpg|...
  test/normal/*.png|*.jpg|...
  test/anomaly/*.png
  ground_truth/anomaly/*_mask.png
  manifest.jsonl
```

You can then use `pyimgano-manifest` or the manifest loader:

```python
from pyimgano.datasets.manifest import load_manifest_benchmark_split

split = load_manifest_benchmark_split(
    manifest_path="./out_synth_dataset/manifest.jsonl",
    root_fallback="./out_synth_dataset",
    category="synthetic_demo",
)
```

---

## 4) Building Masks With Perlin + Poisson Blend

Internally, the synthesis pipeline relies on:
- Perlin/fBm noise (mask generation)
- CutPaste variants (patch-based anomalies)
- Alpha blending + Poisson blending (OpenCV `seamlessClone`)

If you need a custom recipe, build your own preset and use `AnomalySynthesizer(preset_fn=...)`.
