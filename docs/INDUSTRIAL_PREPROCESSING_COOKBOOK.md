# Industrial Preprocessing Cookbook

This page collects practical preprocessing “recipes” for industrial visual anomaly detection.
All examples are **lightweight** and intended to run in production pipelines.

## 1. Shading / Illumination Correction

Use this when you have slow-varying lighting changes or vignetting.

```python
from pyimgano.preprocessing.industrial_presets import shading_correction

out = shading_correction(
    image_u8,
    radius=50,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
)
```

Tips:

- Increase `radius` to remove slower-varying background.
- CLAHE increases local contrast; tune `clip_limit` to avoid over-amplifying noise.

## 2. Defect Amplification (Small Bright Defects)

This highlights small bright defects and sharp boundaries.

```python
from pyimgano.preprocessing.industrial_presets import defect_amplification

amp = defect_amplification(
    image_u8,
    tophat_ksize=(15, 15),
    edge_method="sobel",  # or "canny"
    tophat_weight=1.0,
    edge_weight=0.5,
)
```

## 3. Guided Filter (Edge-Preserving Denoise)

```python
from pyimgano.preprocessing.guided_filter import guided_filter

den = guided_filter(image_u8, radius=8, eps=1e-3)
```

## 4. Anisotropic Diffusion (Perona–Malik)

```python
from pyimgano.preprocessing.anisotropic_diffusion import anisotropic_diffusion

den = anisotropic_diffusion(image_u8, niter=10, kappa=50.0, gamma=0.1, option=1)
```

## 5. Local Contrast Normalization (LCN)

Useful as a “last-mile” normalization before feature extraction.

```python
from pyimgano.preprocessing import ImageEnhancer

enh = ImageEnhancer()
lcn = enh.local_contrast_normalization(image_u8, ksize=15, clip=3.0)
```

## 6. JPEG Artifact Robustness

```python
from pyimgano.preprocessing import ImageEnhancer

enh = ImageEnhancer()
clean = enh.jpeg_robust_preprocess(image_u8, strength=0.7)
```

## 7. High-Resolution Images: Tile + Blend

When operations are too slow on the full frame, run them per-tile and blend seams.

```python
from pyimgano.preprocessing.tiling import tile_apply
from pyimgano.preprocessing.guided_filter import guided_filter

out = tile_apply(
    image_u8,
    lambda tile: guided_filter(tile, radius=8, eps=1e-3),
    tile_size=512,
    overlap=64,
    blend="hann",
)
```

## Notes

- Most helpers assume `uint8` images. Keep your preprocessing explicit and deterministic.
- Prefer a small number of strong steps over stacking many weak transforms.

