from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_component(text: str) -> str:
    value = str(text).strip()
    value = _SAFE_CHARS_RE.sub("_", value)
    return value.strip("._-") or "map"


def save_anomaly_map_npy(
    out_dir: str | Path,
    *,
    index: int,
    input_path: str | Path,
    anomaly_map: np.ndarray,
) -> Path:
    base = Path(out_dir)
    maps_dir = base / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    stem = _sanitize_component(Path(str(input_path)).stem)
    out_path = maps_dir / f"{int(index):06d}_{stem}.npy"
    np.save(out_path, np.asarray(anomaly_map, dtype=np.float32))
    return out_path

