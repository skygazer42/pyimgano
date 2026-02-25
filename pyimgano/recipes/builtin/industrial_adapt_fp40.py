from __future__ import annotations

from dataclasses import replace
from typing import Any

from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.config import (
    DefectsConfig,
    HysteresisConfig,
    MapSmoothingConfig,
    MergeNearbyConfig,
    ShapeFiltersConfig,
    WorkbenchConfig,
)
from pyimgano.workbench.runner import run_workbench


@register_recipe(
    "industrial-adapt-fp40",
    tags=("builtin", "adaptation", "defects"),
    metadata={
        "description": (
            "Industrial adaptation recipe with FP40 defaults for inference export "
            "(defects ROI/border/smoothing/hysteresis/shape filters)."
        ),
    },
)
def industrial_adapt_fp40(config: WorkbenchConfig) -> dict[str, Any]:
    """Run the industrial workbench with FP-reduction defaults.

    Notes
    -----
    - This recipe is intended for the deploy flow: `pyimgano-train --export-infer-config`
      and then `pyimgano-infer --infer-config ... --defects`.
    - Workbench training/eval does not run defects extraction, but the defects block
      is persisted in the run config and exported into `infer_config.json`.
    """

    recipe_name = "industrial-adapt-fp40"

    # Ensure maps are available downstream (infer-config can auto-enable maps).
    adaptation = config.adaptation
    if not bool(adaptation.save_maps):
        adaptation = replace(adaptation, save_maps=True)

    defects = config.defects
    # Apply "FP40" defaults. Users can still use the base `industrial-adapt` recipe
    # for full manual control.
    defects = DefectsConfig(
        enabled=True,
        pixel_threshold=defects.pixel_threshold,
        pixel_threshold_strategy=str(defects.pixel_threshold_strategy),
        pixel_normal_quantile=float(defects.pixel_normal_quantile),
        mask_format=str(defects.mask_format),
        roi_xyxy_norm=defects.roi_xyxy_norm if defects.roi_xyxy_norm is not None else (0.1, 0.1, 0.9, 0.9),
        border_ignore_px=int(defects.border_ignore_px) if int(defects.border_ignore_px) > 0 else 2,
        map_smoothing=MapSmoothingConfig(
            method=str(defects.map_smoothing.method) if str(defects.map_smoothing.method) != "none" else "median",
            ksize=int(defects.map_smoothing.ksize) if int(defects.map_smoothing.ksize) > 0 else 3,
            sigma=float(defects.map_smoothing.sigma),
        ),
        hysteresis=HysteresisConfig(
            enabled=True if not bool(defects.hysteresis.enabled) else bool(defects.hysteresis.enabled),
            low=defects.hysteresis.low,
            high=defects.hysteresis.high,
        ),
        shape_filters=ShapeFiltersConfig(
            min_fill_ratio=defects.shape_filters.min_fill_ratio
            if defects.shape_filters.min_fill_ratio is not None
            else 0.15,
            max_aspect_ratio=defects.shape_filters.max_aspect_ratio
            if defects.shape_filters.max_aspect_ratio is not None
            else 6.0,
            min_solidity=defects.shape_filters.min_solidity
            if defects.shape_filters.min_solidity is not None
            else 0.8,
        ),
        merge_nearby=MergeNearbyConfig(
            enabled=True if not bool(defects.merge_nearby.enabled) else bool(defects.merge_nearby.enabled),
            max_gap_px=int(defects.merge_nearby.max_gap_px) if int(defects.merge_nearby.max_gap_px) > 0 else 1,
        ),
        min_area=int(defects.min_area) if int(defects.min_area) > 0 else 16,
        min_score_max=defects.min_score_max if defects.min_score_max is not None else 0.6,
        min_score_mean=defects.min_score_mean,
        open_ksize=int(defects.open_ksize),
        close_ksize=int(defects.close_ksize),
        fill_holes=bool(defects.fill_holes),
        max_regions=defects.max_regions if defects.max_regions is not None else 20,
        max_regions_sort_by=str(defects.max_regions_sort_by),
    )

    cfg = replace(config, recipe=recipe_name, adaptation=adaptation, defects=defects)
    return run_workbench(config=cfg, recipe_name=recipe_name)

