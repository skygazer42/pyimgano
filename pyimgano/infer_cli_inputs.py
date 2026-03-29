from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_json_mapping_arg(text: str, *, arg_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON. Original error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object (e.g. '{{\"k\": 1}}').")

    return dict(parsed)


def parse_csv_ints_arg(text: str, *, arg_name: str) -> list[int]:
    raw = [token.strip() for token in str(text).split(",")]
    values: list[int] = []
    for item in raw:
        if not item:
            continue
        try:
            values.append(int(item))
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(
                f"{arg_name} must be a comma-separated list of ints, got {text!r}"
            ) from exc
    return values


def parse_csv_strs_arg(text: str, *, arg_name: str) -> list[str]:
    del arg_name
    raw = [token.strip() for token in str(text).split(",")]
    return [token for token in raw if token]


def collect_image_paths(raw: str | Path) -> list[str]:
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path}")
        return [str(path)]

    output: list[str] = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            output.append(str(child))
    return output


__all__ = [
    "IMAGE_SUFFIXES",
    "collect_image_paths",
    "parse_csv_ints_arg",
    "parse_csv_strs_arg",
    "parse_json_mapping_arg",
]
