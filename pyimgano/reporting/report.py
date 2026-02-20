from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_run_report(path: str | Path, results: dict) -> None:
    """Save a run result dict as JSON (converting numpy types)."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _to_jsonable(results)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

