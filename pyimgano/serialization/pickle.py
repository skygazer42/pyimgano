from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def is_pickle_safe_detector(detector: Any) -> bool:
    """Return True if `detector` is in the supported pickle-safe set.

    Notes
    -----
    - This is a *best-effort* guardrail to keep users from accidentally
      serializing deep models (GPU tensors, large weights).
    - Pickle is not a secure format. Never load a pickle file from an
      untrusted source.
    """

    try:
        from pyimgano.models.baseml import BaseVisionDetector
    except Exception:
        return False

    return isinstance(detector, BaseVisionDetector)


def save_detector(path: str | Path, detector: Any) -> None:
    """Serialize a detector to disk via pickle (restricted to classical detectors)."""

    if not is_pickle_safe_detector(detector):
        raise TypeError(
            "Unsupported detector for pickle serialization. "
            "Only classical vision detectors (BaseVisionDetector wrappers) are supported."
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        pickle.dump(detector, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301


def load_detector(path: str | Path) -> Any:
    """Load a detector from disk via pickle (restricted to classical detectors)."""

    in_path = Path(path)
    with in_path.open("rb") as f:
        detector = pickle.load(f)  # nosec B301 - controlled helper; see module docstring.

    if not is_pickle_safe_detector(detector):
        raise TypeError(
            "Loaded detector is not in the supported pickle-safe set. "
            "Refuse to return it."
        )

    return detector
