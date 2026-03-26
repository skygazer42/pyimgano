"""Named preprocessing schemes for discovery-oriented CLI and config workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

ILLUMINATION_CONTRAST_KEY = "preprocessing.illumination_contrast"


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
        tags=("illumination", "contrast", "deployable", "industrial"),
        config_key=ILLUMINATION_CONTRAST_KEY,
        payload={
            "enabled": True,
            "white_balance": "gray_world",
            "homomorphic": True,
            "homomorphic_cutoff": 0.25,
            "homomorphic_gamma_low": 0.8,
            "homomorphic_gamma_high": 1.2,
            "clahe": True,
            "clahe_tile_grid_size": [8, 8],
            "clahe_clip_limit": 2.0,
            "gamma": None,
            "contrast_lower_percentile": 1.0,
            "contrast_upper_percentile": 99.0,
        },
    ),
    "illumination-contrast-lowlight": PreprocessingScheme(
        name="illumination-contrast-lowlight",
        description=(
            "Stronger low-light recovery for dim production lines with gray-world white balance, "
            "homomorphic filtering, CLAHE, and a gentle gamma lift."
        ),
        deployable=True,
        tags=("illumination", "contrast", "deployable", "lowlight", "industrial"),
        config_key=ILLUMINATION_CONTRAST_KEY,
        payload={
            "enabled": True,
            "white_balance": "gray_world",
            "homomorphic": True,
            "homomorphic_cutoff": 0.2,
            "homomorphic_gamma_low": 0.7,
            "homomorphic_gamma_high": 1.35,
            "clahe": True,
            "clahe_tile_grid_size": [8, 8],
            "clahe_clip_limit": 2.5,
            "gamma": 1.08,
            "contrast_lower_percentile": 0.5,
            "contrast_upper_percentile": 99.5,
        },
    ),
    "illumination-contrast-texture-boost": PreprocessingScheme(
        name="illumination-contrast-texture-boost",
        description=(
            "Texture-biased normalization for subtle surface defects with max-RGB white balance, "
            "CLAHE, and tighter percentile contrast stretching."
        ),
        deployable=True,
        tags=("illumination", "contrast", "deployable", "texture", "industrial"),
        config_key=ILLUMINATION_CONTRAST_KEY,
        payload={
            "enabled": True,
            "white_balance": "max_rgb",
            "homomorphic": False,
            "homomorphic_cutoff": 0.25,
            "homomorphic_gamma_low": 0.8,
            "homomorphic_gamma_high": 1.2,
            "clahe": True,
            "clahe_tile_grid_size": [8, 8],
            "clahe_clip_limit": 3.0,
            "gamma": None,
            "contrast_lower_percentile": 2.0,
            "contrast_upper_percentile": 98.0,
        },
    ),
    "illumination-contrast-aggressive": PreprocessingScheme(
        name="illumination-contrast-aggressive",
        description=(
            "Aggressive illumination/contrast normalization for difficult production data: "
            "gray-world white balance + homomorphic filtering + strong CLAHE + contrast stretching."
        ),
        deployable=True,
        tags=("illumination", "contrast", "deployable", "industrial", "aggressive"),
        config_key=ILLUMINATION_CONTRAST_KEY,
        payload={
            "enabled": True,
            "white_balance": "gray_world",
            "homomorphic": True,
            "homomorphic_cutoff": 0.18,
            "homomorphic_gamma_low": 0.65,
            "homomorphic_gamma_high": 1.45,
            "homomorphic_c": 1.0,
            "homomorphic_per_channel": True,
            "clahe": True,
            "clahe_tile_grid_size": [8, 8],
            "clahe_clip_limit": 3.5,
            "gamma": 1.05,
            "contrast_stretch": True,
            "contrast_lower_percentile": 0.5,
            "contrast_upper_percentile": 99.5,
        },
    ),
    "illumination-contrast-no-homomorphic": PreprocessingScheme(
        name="illumination-contrast-no-homomorphic",
        description=(
            "Frequency-domain-free industrial preset: gray-world white balance + CLAHE + "
            "mild contrast stretching (no homomorphic filter)."
        ),
        deployable=True,
        tags=("illumination", "contrast", "deployable", "industrial", "no-homomorphic"),
        config_key=ILLUMINATION_CONTRAST_KEY,
        payload={
            "enabled": True,
            "white_balance": "gray_world",
            "homomorphic": False,
            "homomorphic_cutoff": 0.25,
            "homomorphic_gamma_low": 0.8,
            "homomorphic_gamma_high": 1.2,
            "homomorphic_c": 1.0,
            "homomorphic_per_channel": True,
            "clahe": True,
            "clahe_tile_grid_size": [8, 8],
            "clahe_clip_limit": 2.2,
            "gamma": None,
            "contrast_stretch": True,
            "contrast_lower_percentile": 1.0,
            "contrast_upper_percentile": 99.0,
        },
    ),
    "illumination-contrast-color-stable": PreprocessingScheme(
        name="illumination-contrast-color-stable",
        description=(
            "Color-stable preset for color-sensitive inspection: max-RGB white balance + "
            "gentle CLAHE + mild gamma lift (avoids heavy global contrast stretching)."
        ),
        deployable=True,
        tags=("illumination", "contrast", "deployable", "industrial", "color-stable"),
        config_key=ILLUMINATION_CONTRAST_KEY,
        payload={
            "enabled": True,
            "white_balance": "max_rgb",
            "homomorphic": False,
            "homomorphic_cutoff": 0.25,
            "homomorphic_gamma_low": 0.8,
            "homomorphic_gamma_high": 1.2,
            "homomorphic_c": 1.0,
            "homomorphic_per_channel": True,
            "clahe": True,
            "clahe_tile_grid_size": [8, 8],
            "clahe_clip_limit": 1.8,
            "gamma": 1.03,
            "contrast_stretch": False,
            "contrast_lower_percentile": 2.0,
            "contrast_upper_percentile": 98.0,
        },
    ),
    "retinex-msrcr-lite": PreprocessingScheme(
        name="retinex-msrcr-lite",
        description=(
            "Retinex-based illumination flattening for Python API workflows using "
            "`pyimgano.preprocessing.msrcr_lite`."
        ),
        deployable=False,
        tags=("retinex", "illumination", "python-api"),
        entrypoint="pyimgano.preprocessing.msrcr_lite",
    ),
    "rolling-ball-flatfield": PreprocessingScheme(
        name="rolling-ball-flatfield",
        description=(
            "Background/flat-field correction for uneven lighting using the rolling-ball background "
            "estimate and subtraction helpers."
        ),
        deployable=False,
        tags=("background", "flatfield", "python-api"),
        entrypoint="pyimgano.preprocessing.subtract_background_rolling_ball",
    ),
    "guided-filter-denoise": PreprocessingScheme(
        name="guided-filter-denoise",
        description=(
            "Edge-preserving denoising using guided filtering for Python API image cleanup before "
            "feature extraction or scoring."
        ),
        deployable=False,
        tags=("denoise", "edge-preserving", "python-api"),
        entrypoint="pyimgano.preprocessing.guided_filter",
    ),
    "defect-amplify-local-contrast": PreprocessingScheme(
        name="defect-amplify-local-contrast",
        description=(
            "Defect amplification helper that boosts local contrast for weak texture anomalies in "
            "manual preprocessing pipelines."
        ),
        deployable=False,
        tags=("defect-boost", "contrast", "python-api"),
        entrypoint="pyimgano.preprocessing.defect_amplification",
    ),
    "anisotropic-diffusion-denoise": PreprocessingScheme(
        name="anisotropic-diffusion-denoise",
        description=(
            "Edge-preserving denoising using Perona–Malik anisotropic diffusion (Python API helper)."
        ),
        deployable=False,
        tags=("denoise", "edge-preserving", "python-api", "industrial"),
        entrypoint="pyimgano.preprocessing.anisotropic_diffusion",
    ),
    "shading-correction-rolling-ball": PreprocessingScheme(
        name="shading-correction-rolling-ball",
        description=(
            "Shading correction preset for uneven illumination: rolling-ball background subtraction + CLAHE."
        ),
        deployable=False,
        tags=("background", "flatfield", "illumination", "python-api", "industrial"),
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
