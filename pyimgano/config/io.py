from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file into a Python dict.

    Supported formats:
    - JSON (.json) always
    - YAML (.yml/.yaml) only when PyYAML is installed
    """

    config_path = Path(path)
    suffix = str(config_path.suffix).lower()

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001 - dependency boundary
            raise ImportError(
                "YAML config files require PyYAML.\n"
                "Install it via:\n"
                "  pip install 'PyYAML'"
            ) from exc

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config extension: {suffix!r} for {str(config_path)!r}. "
            "Supported: .json, .yml, .yaml."
        )

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            "Config must be an object/dict at the top level, "
            f"got {type(data).__name__} from {str(config_path)!r}."
        )

    return dict(data)

