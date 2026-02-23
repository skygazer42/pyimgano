from __future__ import annotations

import inspect
import json
from typing import Any

from pyimgano.models.registry import MODEL_REGISTRY


def parse_model_kwargs(text: str | None) -> dict[str, Any]:
    if text is None:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--model-kwargs must be valid JSON. Original error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("--model-kwargs must be a JSON object (e.g. '{\"k\": 1}').")

    return dict(parsed)


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


def _get_model_signature_info(model_name: str) -> tuple[set[str], bool]:
    """Return (accepted_kwarg_names, accepts_var_kwargs) for a registered model."""

    entry = MODEL_REGISTRY.info(model_name)
    constructor = entry.constructor
    sig = inspect.signature(constructor)

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return accepted, accepts_var_kwargs


def validate_user_model_kwargs(model_name: str, user_kwargs: dict[str, Any]) -> None:
    accepted, accepts_var_kwargs = _get_model_signature_info(model_name)
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
    """Combine preset, user, and auto kwargs with stable precedence.

    Precedence (highest wins):
    - user kwargs (explicit --model-kwargs)
    - preset kwargs (--preset)
    - auto kwargs (device/contamination/pretrained inferred from CLI flags)

    Preset/auto kwargs are filtered to only include keys accepted by the target
    model constructor (unless it accepts **kwargs).
    """

    validate_user_model_kwargs(model_name, user_kwargs)
    accepted, accepts_var_kwargs = _get_model_signature_info(model_name)

    out: dict[str, Any] = {}
    if preset_kwargs:
        for key, value in preset_kwargs.items():
            if accepts_var_kwargs or key in accepted:
                out[key] = value
    out.update(user_kwargs)
    for key, value in auto_kwargs.items():
        if key in out:
            continue
        if accepts_var_kwargs or key in accepted:
            out[key] = value
    return out

