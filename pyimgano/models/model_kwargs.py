from __future__ import annotations

from typing import Any

from pyimgano.models.introspection import get_constructor_signature_info
from pyimgano.models.registry import MODEL_REGISTRY, materialize_model_constructor


def merge_checkpoint_path(
    model_kwargs: dict[str, Any],
    *,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return dict(model_kwargs)

    out = dict(model_kwargs)
    existing = out.get("checkpoint_path", None)
    if existing is not None and str(existing) != str(checkpoint_path):
        raise ValueError(
            "checkpoint_path conflict: "
            f"--checkpoint-path={checkpoint_path!r} but model-kwargs checkpoint_path={existing!r}"
        )

    out["checkpoint_path"] = str(checkpoint_path)
    return out


def get_model_signature_info(model_name: str) -> tuple[set[str], bool]:
    """Return (accepted_kwarg_names, accepts_var_kwargs) for a registered model."""

    MODEL_REGISTRY.info(model_name)
    constructor = materialize_model_constructor(model_name)
    _signature, accepted, accepts_var_kwargs = get_constructor_signature_info(constructor)
    return accepted, accepts_var_kwargs


def validate_user_model_kwargs(model_name: str, user_kwargs: dict[str, Any]) -> None:
    accepted, accepts_var_kwargs = get_model_signature_info(model_name)
    if accepts_var_kwargs:
        return

    unknown = sorted(set(user_kwargs) - accepted)
    if unknown:
        allowed = ", ".join(sorted(accepted)) or "<none>"
        raise TypeError(
            f"Model {model_name!r} does not accept model-kwargs: {unknown}. "
            f"Allowed keys: {allowed}"
        )


def build_model_kwargs(
    model_name: str,
    *,
    user_kwargs: dict[str, Any],
    preset_kwargs: dict[str, Any] | None = None,
    auto_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Combine preset, user, and auto kwargs with stable precedence."""

    validate_user_model_kwargs(model_name, user_kwargs)
    accepted, accepts_var_kwargs = get_model_signature_info(model_name)

    out: dict[str, Any] = {}
    if preset_kwargs:
        for key, value in preset_kwargs.items():
            if accepts_var_kwargs or key in accepted:
                out[key] = value
    out.update(user_kwargs)
    auto_passthrough = {"contamination", "random_state", "random_seed"}
    for key, value in auto_kwargs.items():
        if key in out:
            continue
        if key in accepted or (accepts_var_kwargs and key in auto_passthrough):
            out[key] = value

    if "feature_extractor" in out:
        from pyimgano.features.registry import resolve_feature_extractor

        out["feature_extractor"] = resolve_feature_extractor(out["feature_extractor"])
    return out


__all__ = [
    "build_model_kwargs",
    "get_model_signature_info",
    "merge_checkpoint_path",
    "validate_user_model_kwargs",
]
