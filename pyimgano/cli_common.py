from __future__ import annotations

import json
from typing import Any

import pyimgano.models.model_kwargs as model_kwargs_support


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
    return model_kwargs_support.merge_checkpoint_path(
        model_kwargs,
        checkpoint_path=checkpoint_path,
    )


def _get_model_signature_info(model_name: str) -> tuple[set[str], bool]:
    return model_kwargs_support.get_model_signature_info(model_name)


def validate_user_model_kwargs(model_name: str, user_kwargs: dict[str, Any]) -> None:
    model_kwargs_support.validate_user_model_kwargs(model_name, user_kwargs)


def build_model_kwargs(
    model_name: str,
    *,
    user_kwargs: dict[str, Any],
    preset_kwargs: dict[str, Any] | None = None,
    auto_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return model_kwargs_support.build_model_kwargs(
        model_name,
        user_kwargs=user_kwargs,
        preset_kwargs=preset_kwargs,
        auto_kwargs=auto_kwargs,
    )
