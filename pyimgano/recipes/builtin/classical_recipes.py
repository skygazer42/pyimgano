from __future__ import annotations

from dataclasses import replace
from typing import Any

from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


@register_recipe(
    "classical-hog-ecod",
    tags=("builtin", "classical"),
    metadata={"description": "Classical baseline: HOG features + ECOD"},
)
def classical_hog_ecod(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-hog-ecod"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault("feature_extractor", {"name": "hog", "kwargs": {"resize_hw": [128, 128]}})
    cfg = replace(config, model=replace(config.model, name="vision_ecod", model_kwargs=model_kwargs))
    return run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-lbp-loop",
    tags=("builtin", "classical"),
    metadata={"description": "Classical baseline: LBP features + LoOP"},
)
def classical_lbp_loop(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-lbp-loop"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        "feature_extractor",
        {"name": "lbp", "kwargs": {"n_points": 8, "radius": 1.0, "method": "uniform"}},
    )
    model_kwargs.setdefault("n_neighbors", 15)
    cfg = replace(config, model=replace(config.model, name="vision_loop", model_kwargs=model_kwargs))
    return run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-colorhist-mahalanobis",
    tags=("builtin", "classical"),
    metadata={"description": "Classical baseline: HSV color hist + Mahalanobis"},
)
def classical_colorhist_mahalanobis(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-colorhist-mahalanobis"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        "feature_extractor",
        {"name": "color_hist", "kwargs": {"colorspace": "hsv", "bins": [16, 16, 16]}},
    )
    model_kwargs.setdefault("reg", 1e-6)
    cfg = replace(
        config,
        model=replace(config.model, name="vision_mahalanobis", model_kwargs=model_kwargs),
    )
    return run_workbench(config=cfg, recipe_name=recipe_name)

