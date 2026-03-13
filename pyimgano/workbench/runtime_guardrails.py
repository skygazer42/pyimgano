from __future__ import annotations

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.model_compatibility import (
    collect_workbench_pixel_map_requirements,
    load_workbench_model_capabilities,
)


def _require_pixel_map_model_for_workbench_features(*, config: WorkbenchConfig) -> None:
    needs = collect_workbench_pixel_map_requirements(config=config)
    if not needs:
        return

    caps = load_workbench_model_capabilities(model_name=str(config.model.name))
    if caps.supports_pixel_map:
        return

    raise ValueError(
        "These workbench options require a model that supports pixel maps: "
        f"{', '.join(needs)}. "
        f"model={config.model.name!r} supports_pixel_map={bool(caps.supports_pixel_map)!r}. "
        "Choose a model with tag 'pixel_map' or disable the pixel-map-dependent options."
    )


def validate_workbench_runtime_guardrails(*, config: WorkbenchConfig) -> None:
    if bool(config.adaptation.save_maps) and not bool(config.output.save_run):
        raise ValueError("adaptation.save_maps requires output.save_run=true.")
    if bool(config.training.enabled) and not bool(config.output.save_run):
        raise ValueError("training.enabled requires output.save_run=true.")
    _require_pixel_map_model_for_workbench_features(config=config)


__all__ = ["validate_workbench_runtime_guardrails"]
