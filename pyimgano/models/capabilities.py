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
    supports_confidence: bool


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


def _constructor_supports_confidence(constructor: Any) -> bool:
    if hasattr(constructor, "predict_confidence"):
        return True

    if not isinstance(constructor, type):
        return False

    try:
        from pyimgano.models.base_detector import BaseDetector
    except Exception:
        return False

    try:
        return issubclass(constructor, BaseDetector)
    except TypeError:
        return False


def _deployment_training_regime(
    *,
    tags: set[str],
    family: Sequence[str],
    supervision: str | None,
    requires_checkpoint: bool,
) -> str:
    if requires_checkpoint:
        return "checkpoint-wrapper"
    if supervision is not None and str(supervision).strip():
        return str(supervision).strip()
    if "template" in {str(item) for item in family}:
        return "reference-fit"
    if "classical" in tags:
        return "one-class-fit"
    return "normal-only-fit"


def _runtime_cost_hint(
    *,
    tags: set[str],
    family: Sequence[str],
    has_memory_bank: bool,
) -> str:
    family_set = {str(item) for item in family}
    if (
        "clip" in tags
        or "filopp" in family_set
        or "aaclip" in family_set
        or "adaclip" in family_set
    ):
        return "high"
    if has_memory_bank or "patchcore" in family_set:
        return "high"
    if "deep" in tags or "embeddings" in tags:
        return "medium"
    return "low"


def _memory_cost_hint(
    *,
    tags: set[str],
    family: Sequence[str],
    has_memory_bank: bool,
) -> str:
    family_set = {str(item) for item in family}
    if has_memory_bank or "memory_bank" in family_set or "patchcore" in family_set:
        return "high"
    if "deep" in tags or "embeddings" in tags:
        return "medium"
    return "low"


def compute_model_deployment_profile(entry: _ModelEntryLike) -> dict[str, Any]:
    from pyimgano.models.metadata_contract import resolve_metadata_contract_payload

    _signature, accepted_kwargs, _accepts_var_kwargs = get_constructor_signature_info(
        entry.constructor
    )
    tags = {str(tag).strip().lower() for tag in entry.tags}
    tuning_knobs = [
        name
        for name in (
            "knn_backend",
            "n_neighbors",
            "k_neighbors",
            "coreset_sampling_ratio",
            "coreset_ratio",
            "memory_bank_dtype",
            "feature_projection_dim",
            "memory_size",
        )
        if name in accepted_kwargs
    ]
    has_memory_bank = bool(
        any(
            knob in accepted_kwargs
            for knob in (
                "knn_backend",
                "coreset_sampling_ratio",
                "coreset_ratio",
                "memory_bank_dtype",
                "memory_size",
            )
        )
    )

    default_backend = None
    if "knn_backend" in accepted_kwargs:
        try:
            from pyimgano.services.model_options import _default_knn_backend

            default_backend = str(_default_knn_backend())
        except Exception:
            default_backend = None

    contract = resolve_metadata_contract_payload(entry)
    requires_checkpoint = bool(contract.get("requires_checkpoint"))
    family = [str(item) for item in contract.get("family", []) if str(item).strip()]
    supervision = contract.get("supervision")
    supports_pixel_map = bool(contract.get("supports_pixel_map"))
    training_regime = _deployment_training_regime(
        tags=tags,
        family=family,
        supervision=(str(supervision) if supervision is not None else None),
        requires_checkpoint=bool(requires_checkpoint),
    )
    runtime_cost = _runtime_cost_hint(tags=tags, family=family, has_memory_bank=has_memory_bank)
    memory_cost = _memory_cost_hint(tags=tags, family=family, has_memory_bank=has_memory_bank)

    artifact_requirements: list[str] = []
    if requires_checkpoint:
        artifact_requirements.append("checkpoint")

    return {
        "family": family,
        "training_regime": training_regime,
        "industrial_fit": {
            "pixel_localization": bool(supports_pixel_map),
            "reference_inspection": bool("template" in set(family)),
            "few_shot_adaptation": bool(str(supervision).strip() in {"few-shot", "zero-shot"}),
            "checkpoint_dependent": bool(requires_checkpoint),
            "memory_bank_retrieval": bool(has_memory_bank or "memory_bank" in set(family)),
        },
        "runtime_cost_hint": runtime_cost,
        "memory_cost_hint": memory_cost,
        "export_support": {
            "checkpoint": bool(
                requires_checkpoint
                or "checkpoint_path" in accepted_kwargs
                or bool(contract.get("weights_source"))
            ),
            "save_load": bool("classical" in tags and not requires_checkpoint),
            "onnx": bool("onnx" in tags or contract.get("weights_source") == "local-exported-onnx"),
            "torchscript": bool(
                "torchscript" in tags
                or contract.get("weights_source") == "local-exported-torchscript"
            ),
        },
        "artifact_requirements": artifact_requirements,
        "memory_bank": {
            "enabled": bool(has_memory_bank),
            "backend_param": ("knn_backend" if "knn_backend" in accepted_kwargs else None),
            "default_backend": default_backend,
            "tuning_knobs": list(tuning_knobs),
        },
    }


def compute_model_capabilities(entry: _ModelEntryLike) -> ModelCapabilities:
    tags = {str(t) for t in entry.tags}

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
    supports_confidence = _constructor_supports_confidence(entry.constructor)

    return ModelCapabilities(
        input_modes=tuple(input_modes),
        supports_pixel_map=bool(supports_pixel_map),
        supports_checkpoint=bool(supports_checkpoint),
        requires_checkpoint=bool(requires_checkpoint),
        supports_save_load=bool(supports_save_load),
        supports_confidence=bool(supports_confidence),
    )
