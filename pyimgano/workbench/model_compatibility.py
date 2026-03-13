from __future__ import annotations

from dataclasses import dataclass

from pyimgano.workbench.config import WorkbenchConfig


@dataclass(frozen=True)
class WorkbenchModelCapabilitySummary:
    model_name: str
    input_modes: tuple[str, ...]
    supports_pixel_map: bool


def collect_workbench_pixel_map_requirements(*, config: WorkbenchConfig) -> tuple[str, ...]:
    try:
        adaptation = config.adaptation
    except Exception:  # noqa: BLE001 - best-effort config probing
        adaptation = None
    try:
        defects = config.defects
    except Exception:  # noqa: BLE001 - best-effort config probing
        defects = None

    requirements: list[str] = []
    if bool(getattr(adaptation, "save_maps", False)):
        requirements.append("adaptation.save_maps")
    if getattr(adaptation, "postprocess", None) is not None:
        requirements.append("adaptation.postprocess")
    if bool(getattr(defects, "enabled", False)):
        requirements.append("defects.enabled")
    return tuple(requirements)


def load_workbench_model_capabilities(*, model_name: str) -> WorkbenchModelCapabilitySummary:
    from pyimgano.models.capabilities import compute_model_capabilities
    from pyimgano.models.registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY.info(str(model_name))
    caps = compute_model_capabilities(entry)
    return WorkbenchModelCapabilitySummary(
        model_name=str(model_name),
        input_modes=tuple(str(mode) for mode in caps.input_modes),
        supports_pixel_map=bool(caps.supports_pixel_map),
    )


__all__ = [
    "WorkbenchModelCapabilitySummary",
    "collect_workbench_pixel_map_requirements",
    "load_workbench_model_capabilities",
]
