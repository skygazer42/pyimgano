from __future__ import annotations

from dataclasses import replace
from typing import Any

import pyimgano.services.workbench_service as workbench_service
from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.config import ModelConfig, WorkbenchConfig


@register_recipe(
    "industrial-embedding-core-fast",
    tags=("builtin", "embeddings", "pipeline"),
    metadata={
        "description": (
            "Industrial baseline recipe: deep embeddings + classical core detector "
            "(safe-by-default; no implicit weight downloads)."
        ),
    },
)
def industrial_embedding_core_fast(config: WorkbenchConfig) -> dict[str, Any]:
    """Workbench recipe that defaults to the stable 'embeddings + core' route.

    Design goals:
    - make embedding+core a 1-line recipe choice
    - avoid implicit weight downloads by default
    - preserve explicit user config overrides in `model.model_kwargs`
    """

    recipe_name = "industrial-embedding-core-fast"

    model = config.model
    mk = dict(model.model_kwargs or {})

    # Respect explicit user overrides; only fill missing keys.
    if "embedding_extractor" not in mk:
        mk["embedding_extractor"] = "torchvision_backbone"
    embed_name = str(mk.get("embedding_extractor")).strip()

    if embed_name == "torchvision_backbone" and "embedding_kwargs" not in mk:
        mk["embedding_kwargs"] = {
            "backbone": "resnet18",
            "pretrained": False,
            "pool": "avg",
            "device": str(model.device),
        }
    mk.setdefault("core_detector", "core_ecod")
    mk.setdefault("core_kwargs", {})

    new_model = ModelConfig(
        name="vision_embedding_core",
        device=str(model.device),
        preset=None,
        pretrained=False,
        contamination=float(model.contamination),
        model_kwargs=mk,
        checkpoint_path=model.checkpoint_path,
    )

    cfg = replace(config, recipe=recipe_name, model=new_model)
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)
