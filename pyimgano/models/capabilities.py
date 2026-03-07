from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from pyimgano.models.introspection import get_constructor_signature_info


class _ModelEntryLike(Protocol):
    # Read-only structural protocol: capabilities logic only reads attributes.
    @property
    def name(self) -> str: ...

    @property
    def constructor(self) -> Any: ...

    @property
    def tags(self) -> Sequence[str]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...


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
        from pyimgano.models.baseml import BaseVisionDetector
    except Exception:
        return False

    try:
        return issubclass(constructor, (BaseVisionDetector, BaseVisionDeepDetector))
    except TypeError:
        return False


def _constructor_supports_pixel_map(constructor: Any) -> bool:
    return bool(
        hasattr(constructor, "predict_anomaly_map") or hasattr(constructor, "get_anomaly_map")
    )


def compute_model_capabilities(entry: _ModelEntryLike) -> ModelCapabilities:
    tags = set(str(t) for t in entry.tags)

    is_core = bool(entry.name.startswith("core_") or ("core" in tags))

    supports_numpy_images = "numpy" in tags or _constructor_supports_numpy(entry.constructor)

    # Input-mode semantics:
    # - `paths`: list[str|Path] pointing to images on disk (vision wrappers)
    # - `numpy`: list[np.ndarray] images already decoded in memory (vision wrappers)
    # - `features`: 2D feature matrix (N,D) for classical/tabular style cores
    if is_core:
        input_modes = ["features"]
    else:
        input_modes = ["paths"]
        if supports_numpy_images:
            input_modes.append("numpy")

    supports_pixel_map = ("pixel_map" in tags) or _constructor_supports_pixel_map(entry.constructor)

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
