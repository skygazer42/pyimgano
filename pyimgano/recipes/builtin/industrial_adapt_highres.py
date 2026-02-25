from __future__ import annotations

from dataclasses import replace
from typing import Any

from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.adaptation import MapPostprocessConfig, TilingConfig
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


@register_recipe(
    "industrial-adapt-highres",
    tags=("builtin", "adaptation", "tiling"),
    metadata={
        "description": (
            "Industrial adaptation recipe with high-resolution tiling defaults "
            "(tile_size/stride + seam-reducing map blending)."
        ),
    },
)
def industrial_adapt_highres(config: WorkbenchConfig) -> dict[str, Any]:
    """Run the industrial workbench with practical high-res defaults.

    This recipe is a convenience preset for inspection images where a single
    224/256 resize is not sufficient.
    """

    recipe_name = "industrial-adapt-highres"

    tiling = config.adaptation.tiling
    if tiling.tile_size is None:
        tiling = TilingConfig(
            tile_size=512,
            stride=384,
            score_reduce="topk_mean",
            score_topk=0.1,
            map_reduce="hann",
        )

    post = config.adaptation.postprocess
    if post is None:
        post = MapPostprocessConfig(
            normalize=True,
            normalize_method="percentile",
            percentile_range=(1.0, 99.0),
            gaussian_sigma=1.0,
            morph_open_ksize=0,
            morph_close_ksize=0,
            component_threshold=None,
            min_component_area=0,
        )

    adaptation = replace(config.adaptation, tiling=tiling, postprocess=post, save_maps=True)
    cfg = replace(config, recipe=recipe_name, adaptation=adaptation)
    return run_workbench(config=cfg, recipe_name=recipe_name)

