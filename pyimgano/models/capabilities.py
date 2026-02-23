from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from pyimgano.models.introspection import get_constructor_signature_info


class _ModelEntryLike(Protocol):
    name: str
    constructor: Any
    tags: Sequence[str]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ModelCapabilities:
    """Computed capabilities for a registry model.

    This intentionally stays lightweight (constructor-introspection only) so it
    can be used for CLI discovery without instantiating detectors.
    """

    input_modes: tuple[str, ...]
    supports_pixel_map: bool
    supports_checkpoint: bool
    requires_checkpoint: bool
    supports_save_load: bool


def _constructor_supports_numpy(constructor: Any) -> bool:
    if not isinstance(constructor, type):
        return False

    try:
        from pyimgano.models.baseCv import BaseVisionDeepDetector
    except Exception:
        return False

    try:
        return issubclass(constructor, BaseVisionDeepDetector)
    except TypeError:
        return False


def _constructor_supports_pixel_map(constructor: Any) -> bool:
    return bool(
        hasattr(constructor, "predict_anomaly_map") or hasattr(constructor, "get_anomaly_map")
    )


def compute_model_capabilities(entry: _ModelEntryLike) -> ModelCapabilities:
    tags = set(str(t) for t in entry.tags)

    supports_numpy = "numpy" in tags or _constructor_supports_numpy(entry.constructor)
    input_modes = ["paths"]
    if supports_numpy:
        input_modes.append("numpy")

    supports_pixel_map = ("pixel_map" in tags) or _constructor_supports_pixel_map(
        entry.constructor
    )

    requires_checkpoint = bool(entry.metadata.get("requires_checkpoint", False))
    _signature, accepted_kwargs, accepts_var_kwargs = get_constructor_signature_info(
        entry.constructor
    )
    supports_checkpoint = bool(
        requires_checkpoint or accepts_var_kwargs or ("checkpoint_path" in accepted_kwargs)
    )

    supports_save_load = bool("classical" in tags and not requires_checkpoint)

    return ModelCapabilities(
        input_modes=tuple(input_modes),
        supports_pixel_map=bool(supports_pixel_map),
        supports_checkpoint=bool(supports_checkpoint),
        requires_checkpoint=bool(requires_checkpoint),
        supports_save_load=bool(supports_save_load),
    )

