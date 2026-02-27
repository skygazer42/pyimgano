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
        preset="scratch",   # see preset list below
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
- `brush`: thick paint/ink brush strokes
- `spatter`: small droplets / spatter (many tiny blobs)
- `tape`: rectangular tape / patch regions (often brighter or stained)
- `marker`: broad marker stroke (strong tint)
- `burn`: heat/burn marks (dark organic regions)
- `bubble`: circular bubble-like defects (bright/dark)
- `fiber`: thin fibers/hairs (curved scratches)
- `wrinkle`: crease/wrinkle waves (wavy stroke)
- `texture`: self-crop texture injection via organic mask
- `edge_wear`: defects concentrated near image borders (edge band)

Get preset names:

```python
from pyimgano.synthesis import get_preset_names
print(get_preset_names())
```

---

## 3) Preset Mixtures (Industrial Sampling)

In many factories, defects are *multi-modal*. Instead of a single defect type,
sample from a **mixture**.

Python:

```python
import numpy as np
from pyimgano.synthesis import AnomalySynthesizer, SynthSpec, make_preset_mixture

img = np.zeros((256, 256, 3), dtype=np.uint8)

mix_fn = make_preset_mixture(["scratch", "stain", "tape", "edge_wear"])
syn = AnomalySynthesizer(
    SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=0.9),
    preset_fn=mix_fn,
)

res = syn(img, seed=0)
print(res.meta.get("preset"), res.meta.get("preset_mixture"))
```

CLI:

```bash
pyimgano-synthesize \
  --in-dir /path/to/normal_images \
  --out-root ./out_synth_dataset \
  --category synthetic_demo \
  --presets scratch stain tape edge_wear \
  --blend alpha \
  --alpha 0.9 \
  --n-train 50 \
  --n-test-normal 20 \
  --n-test-anomaly 20 \
  --seed 0
```

The output manifest will attach `meta` to anomaly samples including:
- `meta.preset`: the chosen preset for that sample
- `meta.preset_mixture`: the mixture list

---

## 4) ROI Mask Constraint (Keep Defects Inside Region)

Many inspection systems have a “valid ROI” (ignore background/fixtures).

Python:

```python
import numpy as np
from pyimgano.synthesis import AnomalySynthesizer, SynthSpec

img = np.zeros((128, 128, 3), dtype=np.uint8)
roi = np.zeros((128, 128), dtype=np.uint8)
roi[16:112, 16:112] = 255

syn = AnomalySynthesizer(SynthSpec(preset="scratch", probability=1.0))
res = syn(img, seed=0, roi_mask=roi)
```

CLI:

```bash
pyimgano-synthesize \
  --in-dir /path/to/normal_images \
  --out-root ./out_synth_dataset \
  --preset scratch \
  --roi-mask /path/to/roi_mask.png \
  --seed 0
```

If the ROI is empty or too restrictive, synthesis may fall back to a normal
sample (label `0` and empty mask).

---

## 5) Dataset Wrapper: `SyntheticAnomalyDataset`

If you're using a torch-style `DataLoader`, you can inject synthetic anomalies
on-the-fly (deterministic per index).

```python
from pyimgano.datasets.synthetic import SyntheticAnomalyDataset
from pyimgano.synthesis import AnomalySynthesizer, SynthSpec, make_preset_mixture

mix_fn = make_preset_mixture(["scratch", "stain", "tape"])
syn = AnomalySynthesizer(
    SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=0.9),
    preset_fn=mix_fn,
)

ds = SyntheticAnomalyDataset(
    image_paths=train_paths,
    synthesizer=syn,
    p_anomaly=0.5,
    seed=0,
)

item = ds[0]
print(item.label, item.mask_u8.shape, item.meta.get("preset"))
```

Notes:
- When an anomaly is injected, `item.meta` includes `preset` and `blend`.
- For ROI restrictions, pass `roi_mask=(H,W) uint8` when constructing the dataset.

---

## 6) CLI: `pyimgano-synthesize`

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

## 7) Building Masks With Perlin + Poisson Blend

Internally, the synthesis pipeline relies on:
- Perlin/fBm noise (mask generation)
- CutPaste variants (patch-based anomalies)
- Alpha blending + Poisson blending (OpenCV `seamlessClone`)

If you need a custom recipe, build your own preset and use `AnomalySynthesizer(preset_fn=...)`.
