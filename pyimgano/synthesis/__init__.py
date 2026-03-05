"""Industrial synthetic anomaly generation utilities.

This package provides **lightweight** utilities to generate synthetic anomalies
and corresponding pixel masks for industrial anomaly detection workflows.

Design goals:
- Deterministic given a seed (for reproducible experiments)
- Dependency-stable (uses NumPy + OpenCV already in core deps)
- Works on `uint8` images (OpenCV-friendly)
"""

from __future__ import annotations

from .perlin import fractal_perlin_noise_2d, perlin_noise_2d
from .presets import get_preset_names, make_preset, make_preset_mixture
from .sources import TextureSourceBank
from .synthesizer import AnomalySynthesizer, SynthResult, SynthSpec

__all__ = [
    "AnomalySynthesizer",
    "SynthResult",
    "SynthSpec",
    "perlin_noise_2d",
    "fractal_perlin_noise_2d",
    "get_preset_names",
    "make_preset",
    "make_preset_mixture",
    "TextureSourceBank",
]
