from __future__ import annotations

from typing import Any, Mapping

from pyimgano.preprocessing.industrial_presets import IlluminationContrastKnobs
from pyimgano.workbench.config_parse_primitives import (
    _optional_float,
    _parse_int_pair,
    _require_mapping,
)
from pyimgano.workbench.config_types import PreprocessingConfig


def _parse_preprocessing_config(top: Mapping[str, Any]) -> PreprocessingConfig:
    preprocessing_raw = top.get("preprocessing", None)
    if preprocessing_raw is None:
        return PreprocessingConfig()

    p_map = _require_mapping(preprocessing_raw, name="preprocessing")
    ic_raw = p_map.get("illumination_contrast", None)
    if ic_raw is None:
        illumination_contrast = None
    else:
        ic_map = _require_mapping(ic_raw, name="preprocessing.illumination_contrast")

        wb_raw = str(ic_map.get("white_balance", "none")).strip().lower()
        if wb_raw in ("", "none"):
            white_balance = "none"
        elif wb_raw in ("gray_world", "gray-world", "grayworld"):
            white_balance = "gray_world"
        elif wb_raw in ("max_rgb", "max-rgb", "maxrgb"):
            white_balance = "max_rgb"
        else:
            raise ValueError(
                "preprocessing.illumination_contrast.white_balance must be one of: "
                "none|gray_world|max_rgb"
            )

        cutoff = _optional_float(
            ic_map.get("homomorphic_cutoff", 0.5),
            name="preprocessing.illumination_contrast.homomorphic_cutoff",
        )
        cutoff_v = float(cutoff if cutoff is not None else 0.5)
        if not (0.0 < cutoff_v <= 1.0):
            raise ValueError(
                "preprocessing.illumination_contrast.homomorphic_cutoff must be in (0,1]"
            )

        gamma_low = _optional_float(
            ic_map.get("homomorphic_gamma_low", 0.7),
            name="preprocessing.illumination_contrast.homomorphic_gamma_low",
        )
        gamma_high = _optional_float(
            ic_map.get("homomorphic_gamma_high", 1.5),
            name="preprocessing.illumination_contrast.homomorphic_gamma_high",
        )
        c = _optional_float(
            ic_map.get("homomorphic_c", 1.0),
            name="preprocessing.illumination_contrast.homomorphic_c",
        )

        clahe_tile_grid_size = _parse_int_pair(
            ic_map.get("clahe_tile_grid_size", None),
            name="preprocessing.illumination_contrast.clahe_tile_grid_size",
            default=(8, 8),
        )
        clahe_clip_limit = _optional_float(
            ic_map.get("clahe_clip_limit", 2.0),
            name="preprocessing.illumination_contrast.clahe_clip_limit",
        )
        clahe_clip_limit_v = float(clahe_clip_limit if clahe_clip_limit is not None else 2.0)
        if clahe_clip_limit_v <= 0.0:
            raise ValueError("preprocessing.illumination_contrast.clahe_clip_limit must be > 0")

        gamma = _optional_float(
            ic_map.get("gamma", None),
            name="preprocessing.illumination_contrast.gamma",
        )
        if gamma is not None and float(gamma) <= 0.0:
            raise ValueError("preprocessing.illumination_contrast.gamma must be > 0 or null")

        lp = _optional_float(
            ic_map.get("contrast_lower_percentile", 2.0),
            name="preprocessing.illumination_contrast.contrast_lower_percentile",
        )
        up = _optional_float(
            ic_map.get("contrast_upper_percentile", 98.0),
            name="preprocessing.illumination_contrast.contrast_upper_percentile",
        )
        lp_v = float(lp if lp is not None else 2.0)
        up_v = float(up if up is not None else 98.0)
        if not (0.0 <= lp_v <= 100.0 and 0.0 <= up_v <= 100.0 and lp_v < up_v):
            raise ValueError(
                "preprocessing.illumination_contrast contrast percentiles must satisfy "
                "0<=lower<upper<=100"
            )

        illumination_contrast = IlluminationContrastKnobs(
            white_balance=white_balance,
            homomorphic=bool(ic_map.get("homomorphic", False)),
            homomorphic_cutoff=cutoff_v,
            homomorphic_gamma_low=float(gamma_low if gamma_low is not None else 0.7),
            homomorphic_gamma_high=float(gamma_high if gamma_high is not None else 1.5),
            homomorphic_c=float(c if c is not None else 1.0),
            homomorphic_per_channel=bool(ic_map.get("homomorphic_per_channel", True)),
            clahe=bool(ic_map.get("clahe", False)),
            clahe_clip_limit=clahe_clip_limit_v,
            clahe_tile_grid_size=clahe_tile_grid_size,
            gamma=(float(gamma) if gamma is not None else None),
            contrast_stretch=bool(ic_map.get("contrast_stretch", False)),
            contrast_lower_percentile=lp_v,
            contrast_upper_percentile=up_v,
        )

    return PreprocessingConfig(illumination_contrast=illumination_contrast)


__all__ = ["_parse_preprocessing_config"]
