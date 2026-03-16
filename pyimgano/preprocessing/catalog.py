"""Named preprocessing schemes for discovery-oriented CLI and config workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

_ILLUMINATION = "illumination"
_CONTRAST = "contrast"
_DEPLOYABLE = "deployable"
_INDUSTRIAL = "industrial"
_PYTHON_API = "python-api"
_PREPROCESSING_ILLUMINATION_CONTRAST = "preprocessing.illumination_contrast"

_ENABLED = "enabled"
_WHITE_BALANCE = "white_balance"
_GRAY_WORLD = "gray_world"
_MAX_RGB = "max_rgb"
_HOMOMORPHIC = "homomorphic"
_HOMOMORPHIC_CUTOFF = "homomorphic_cutoff"
_HOMOMORPHIC_GAMMA_LOW = "homomorphic_gamma_low"
_HOMOMORPHIC_GAMMA_HIGH = "homomorphic_gamma_high"
_HOMOMORPHIC_C = "homomorphic_c"
_HOMOMORPHIC_PER_CHANNEL = "homomorphic_per_channel"
_CLAHE = "clahe"
_CLAHE_TILE_GRID_SIZE = "clahe_tile_grid_size"
_CLAHE_CLIP_LIMIT = "clahe_clip_limit"
_GAMMA = "gamma"
_CONTRAST_STRETCH = "contrast_stretch"
_CONTRAST_LOWER_PERCENTILE = "contrast_lower_percentile"
_CONTRAST_UPPER_PERCENTILE = "contrast_upper_percentile"


@dataclass(frozen=True)
class PreprocessingScheme:
    name: str
    description: str
    deployable: bool
    tags: tuple[str, ...] = ()
    config_key: str | None = None
    payload: Mapping[str, Any] | None = None
    entrypoint: str | None = None


_SCHEMES: dict[str, PreprocessingScheme] = {
    "illumination-contrast-balanced": PreprocessingScheme(
        name="illumination-contrast-balanced",
        description=(
            "Balanced industrial illumination normalization with gray-world white balance, "
            "mild homomorphic filtering, CLAHE, and percentile contrast stretching."
        ),
        deployable=True,
        tags=(_ILLUMINATION, _CONTRAST, _DEPLOYABLE, _INDUSTRIAL),
        config_key=_PREPROCESSING_ILLUMINATION_CONTRAST,
        payload={
            _ENABLED: True,
            _WHITE_BALANCE: _GRAY_WORLD,
            _HOMOMORPHIC: True,
            _HOMOMORPHIC_CUTOFF: 0.25,
            _HOMOMORPHIC_GAMMA_LOW: 0.8,
            _HOMOMORPHIC_GAMMA_HIGH: 1.2,
            _CLAHE: True,
            _CLAHE_TILE_GRID_SIZE: [8, 8],
            _CLAHE_CLIP_LIMIT: 2.0,
            _GAMMA: None,
            _CONTRAST_LOWER_PERCENTILE: 1.0,
            _CONTRAST_UPPER_PERCENTILE: 99.0,
        },
    ),
    "illumination-contrast-lowlight": PreprocessingScheme(
        name="illumination-contrast-lowlight",
        description=(
            "Stronger low-light recovery for dim production lines with gray-world white balance, "
            "homomorphic filtering, CLAHE, and a gentle gamma lift."
        ),
        deployable=True,
        tags=(_ILLUMINATION, _CONTRAST, _DEPLOYABLE, "lowlight", _INDUSTRIAL),
        config_key=_PREPROCESSING_ILLUMINATION_CONTRAST,
        payload={
            _ENABLED: True,
            _WHITE_BALANCE: _GRAY_WORLD,
            _HOMOMORPHIC: True,
            _HOMOMORPHIC_CUTOFF: 0.2,
            _HOMOMORPHIC_GAMMA_LOW: 0.7,
            _HOMOMORPHIC_GAMMA_HIGH: 1.35,
            _CLAHE: True,
            _CLAHE_TILE_GRID_SIZE: [8, 8],
            _CLAHE_CLIP_LIMIT: 2.5,
            _GAMMA: 1.08,
            _CONTRAST_LOWER_PERCENTILE: 0.5,
            _CONTRAST_UPPER_PERCENTILE: 99.5,
        },
    ),
    "illumination-contrast-texture-boost": PreprocessingScheme(
        name="illumination-contrast-texture-boost",
        description=(
            "Texture-biased normalization for subtle surface defects with max-RGB white balance, "
            "CLAHE, and tighter percentile contrast stretching."
        ),
        deployable=True,
        tags=(_ILLUMINATION, _CONTRAST, _DEPLOYABLE, "texture", _INDUSTRIAL),
        config_key=_PREPROCESSING_ILLUMINATION_CONTRAST,
        payload={
            _ENABLED: True,
            _WHITE_BALANCE: _MAX_RGB,
            _HOMOMORPHIC: False,
            _HOMOMORPHIC_CUTOFF: 0.25,
            _HOMOMORPHIC_GAMMA_LOW: 0.8,
            _HOMOMORPHIC_GAMMA_HIGH: 1.2,
            _CLAHE: True,
            _CLAHE_TILE_GRID_SIZE: [8, 8],
            _CLAHE_CLIP_LIMIT: 3.0,
            _GAMMA: None,
            _CONTRAST_LOWER_PERCENTILE: 2.0,
            _CONTRAST_UPPER_PERCENTILE: 98.0,
        },
    ),
    "illumination-contrast-aggressive": PreprocessingScheme(
        name="illumination-contrast-aggressive",
        description=(
            "Aggressive illumination/contrast normalization for difficult production data: "
            "gray-world white balance + homomorphic filtering + strong CLAHE + contrast stretching."
        ),
        deployable=True,
        tags=(_ILLUMINATION, _CONTRAST, _DEPLOYABLE, _INDUSTRIAL, "aggressive"),
        config_key=_PREPROCESSING_ILLUMINATION_CONTRAST,
        payload={
            _ENABLED: True,
            _WHITE_BALANCE: _GRAY_WORLD,
            _HOMOMORPHIC: True,
            _HOMOMORPHIC_CUTOFF: 0.18,
            _HOMOMORPHIC_GAMMA_LOW: 0.65,
            _HOMOMORPHIC_GAMMA_HIGH: 1.45,
            _HOMOMORPHIC_C: 1.0,
            _HOMOMORPHIC_PER_CHANNEL: True,
            _CLAHE: True,
            _CLAHE_TILE_GRID_SIZE: [8, 8],
            _CLAHE_CLIP_LIMIT: 3.5,
            _GAMMA: 1.05,
            _CONTRAST_STRETCH: True,
            _CONTRAST_LOWER_PERCENTILE: 0.5,
            _CONTRAST_UPPER_PERCENTILE: 99.5,
        },
    ),
    "illumination-contrast-no-homomorphic": PreprocessingScheme(
        name="illumination-contrast-no-homomorphic",
        description=(
            "Frequency-domain-free industrial preset: gray-world white balance + CLAHE + "
            "mild contrast stretching (no homomorphic filter)."
        ),
        deployable=True,
        tags=(_ILLUMINATION, _CONTRAST, _DEPLOYABLE, _INDUSTRIAL, "no-homomorphic"),
        config_key=_PREPROCESSING_ILLUMINATION_CONTRAST,
        payload={
            _ENABLED: True,
            _WHITE_BALANCE: _GRAY_WORLD,
            _HOMOMORPHIC: False,
            _HOMOMORPHIC_CUTOFF: 0.25,
            _HOMOMORPHIC_GAMMA_LOW: 0.8,
            _HOMOMORPHIC_GAMMA_HIGH: 1.2,
            _HOMOMORPHIC_C: 1.0,
            _HOMOMORPHIC_PER_CHANNEL: True,
            _CLAHE: True,
            _CLAHE_TILE_GRID_SIZE: [8, 8],
            _CLAHE_CLIP_LIMIT: 2.2,
            _GAMMA: None,
            _CONTRAST_STRETCH: True,
            _CONTRAST_LOWER_PERCENTILE: 1.0,
            _CONTRAST_UPPER_PERCENTILE: 99.0,
        },
    ),
    "illumination-contrast-color-stable": PreprocessingScheme(
        name="illumination-contrast-color-stable",
        description=(
            "Color-stable preset for color-sensitive inspection: max-RGB white balance + "
            "gentle CLAHE + mild gamma lift (avoids heavy global contrast stretching)."
        ),
        deployable=True,
        tags=(_ILLUMINATION, _CONTRAST, _DEPLOYABLE, _INDUSTRIAL, "color-stable"),
        config_key=_PREPROCESSING_ILLUMINATION_CONTRAST,
        payload={
            _ENABLED: True,
            _WHITE_BALANCE: _MAX_RGB,
            _HOMOMORPHIC: False,
            _HOMOMORPHIC_CUTOFF: 0.25,
            _HOMOMORPHIC_GAMMA_LOW: 0.8,
            _HOMOMORPHIC_GAMMA_HIGH: 1.2,
            _HOMOMORPHIC_C: 1.0,
            _HOMOMORPHIC_PER_CHANNEL: True,
            _CLAHE: True,
            _CLAHE_TILE_GRID_SIZE: [8, 8],
            _CLAHE_CLIP_LIMIT: 1.8,
            _GAMMA: 1.03,
            _CONTRAST_STRETCH: False,
            _CONTRAST_LOWER_PERCENTILE: 2.0,
            _CONTRAST_UPPER_PERCENTILE: 98.0,
        },
    ),
    "retinex-msrcr-lite": PreprocessingScheme(
        name="retinex-msrcr-lite",
        description=(
            "Retinex-based illumination flattening for Python API workflows using "
            "`pyimgano.preprocessing.msrcr_lite`."
        ),
        deployable=False,
        tags=("retinex", _ILLUMINATION, _PYTHON_API),
        entrypoint="pyimgano.preprocessing.msrcr_lite",
    ),
    "rolling-ball-flatfield": PreprocessingScheme(
        name="rolling-ball-flatfield",
        description=(
            "Background/flat-field correction for uneven lighting using the rolling-ball background "
            "estimate and subtraction helpers."
        ),
        deployable=False,
        tags=("background", "flatfield", _PYTHON_API),
        entrypoint="pyimgano.preprocessing.subtract_background_rolling_ball",
    ),
    "guided-filter-denoise": PreprocessingScheme(
        name="guided-filter-denoise",
        description=(
            "Edge-preserving denoising using guided filtering for Python API image cleanup before "
            "feature extraction or scoring."
        ),
        deployable=False,
        tags=("denoise", "edge-preserving", _PYTHON_API),
        entrypoint="pyimgano.preprocessing.guided_filter",
    ),
    "defect-amplify-local-contrast": PreprocessingScheme(
        name="defect-amplify-local-contrast",
        description=(
            "Defect amplification helper that boosts local contrast for weak texture anomalies in "
            "manual preprocessing pipelines."
        ),
        deployable=False,
        tags=("defect-boost", _CONTRAST, _PYTHON_API),
        entrypoint="pyimgano.preprocessing.defect_amplification",
    ),
    "anisotropic-diffusion-denoise": PreprocessingScheme(
        name="anisotropic-diffusion-denoise",
        description=(
            "Edge-preserving denoising using Perona–Malik anisotropic diffusion (Python API helper)."
        ),
        deployable=False,
        tags=("denoise", "edge-preserving", _PYTHON_API, _INDUSTRIAL),
        entrypoint="pyimgano.preprocessing.anisotropic_diffusion",
    ),
    "shading-correction-rolling-ball": PreprocessingScheme(
        name="shading-correction-rolling-ball",
        description=(
            "Shading correction preset for uneven illumination: rolling-ball background subtraction + CLAHE."
        ),
        deployable=False,
        tags=("background", "flatfield", _ILLUMINATION, _PYTHON_API, _INDUSTRIAL),
        entrypoint="pyimgano.preprocessing.shading_correction",
    ),
}


def list_preprocessing_schemes(*, deployable_only: bool = False) -> list[PreprocessingScheme]:
    """Return named preprocessing schemes, optionally limited to deployable ones."""

    items = sorted(_SCHEMES.values(), key=lambda item: item.name)
    if not deployable_only:
        return items
    return [item for item in items if item.deployable]


def resolve_preprocessing_scheme(name: str) -> Optional[PreprocessingScheme]:
    """Return a preprocessing scheme by name if it exists."""

    return _SCHEMES.get(str(name).strip(), None)


__all__ = [
    "PreprocessingScheme",
    "list_preprocessing_schemes",
    "resolve_preprocessing_scheme",
]
