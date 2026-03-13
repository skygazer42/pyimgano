from __future__ import annotations

from typing import Any, Callable

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.model_compatibility import (
    collect_workbench_pixel_map_requirements,
    load_workbench_model_capabilities,
)


def run_workbench_model_compat_preflight(
    *,
    config: WorkbenchConfig,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> None:
    try:
        illumination_contrast = config.preprocessing.illumination_contrast
    except Exception:  # noqa: BLE001 - best-effort config probing
        illumination_contrast = None

    pixel_map_requirements = collect_workbench_pixel_map_requirements(config=config)
    save_maps_enabled = "adaptation.save_maps" in pixel_map_requirements
    postprocess_enabled = "adaptation.postprocess" in pixel_map_requirements
    defects_enabled = "defects.enabled" in pixel_map_requirements

    if illumination_contrast is None and not pixel_map_requirements:
        return

    model_name = str(config.model.name)

    try:
        import pyimgano.models  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - best-effort optional import
        issues.append(
            issue_builder(
                "MODEL_REGISTRY_IMPORT_FAILED",
                "warning",
                "Unable to import pyimgano.models to validate model/config compatibility.",
                context={"model": model_name, "error": str(exc)},
            )
        )
        return

    try:
        caps = load_workbench_model_capabilities(model_name=model_name)
    except Exception as exc:  # noqa: BLE001 - registry boundary
        issues.append(
            issue_builder(
                "MODEL_REGISTRY_LOOKUP_FAILED",
                "warning",
                "Unable to look up model in registry to validate model/config compatibility.",
                context={"model": model_name, "error": str(exc)},
            )
        )
        return

    supported_input_modes = tuple(caps.input_modes)
    if illumination_contrast is not None and "numpy" not in supported_input_modes:
        issues.append(
            issue_builder(
                "PREPROCESSING_REQUIRES_NUMPY_MODEL",
                "error",
                "preprocessing.illumination_contrast requires a model that supports numpy inputs.",
                context={
                    "model": model_name,
                    "supported_input_modes": supported_input_modes,
                    "hint": "Choose a model with tag 'numpy' or disable preprocessing.",
                },
            )
        )

    if caps.supports_pixel_map:
        return

    pixel_map_context = {
        "model": model_name,
        "supports_pixel_map": bool(caps.supports_pixel_map),
        "hint": "Choose a model with tag 'pixel_map' or disable the pixel-map-dependent option.",
    }
    if save_maps_enabled:
        issues.append(
            issue_builder(
                "SAVE_MAPS_REQUIRES_PIXEL_MAP_MODEL",
                "error",
                "adaptation.save_maps requires a model that supports pixel maps.",
                context=pixel_map_context,
            )
        )
    if postprocess_enabled:
        issues.append(
            issue_builder(
                "POSTPROCESS_REQUIRES_PIXEL_MAP_MODEL",
                "error",
                "adaptation.postprocess requires a model that supports pixel maps.",
                context=pixel_map_context,
            )
        )
    if defects_enabled:
        issues.append(
            issue_builder(
                "DEFECTS_REQUIRES_PIXEL_MAP_MODEL",
                "error",
                "defects.enabled requires a model that supports pixel maps.",
                context=pixel_map_context,
            )
        )


__all__ = ["run_workbench_model_compat_preflight"]
