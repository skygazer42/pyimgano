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
            "Unsupported detector for pickle serialization. Only classical vision detectors "
            "(BaseVisionDetector wrappers) are supported."
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        pickle.dump(  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
            detector, f, protocol=pickle.HIGHEST_PROTOCOL
        )  # Intentional legacy artifact format; loading requires explicit trust.


def load_detector(path: str | Path, *, trusted: bool = False) -> Any:
    """Load a trusted detector pickle.

    Pickle can execute arbitrary code before the detector type can be checked.
    Callers must set ``trusted=True`` only for artifacts whose origin and
    integrity they have independently verified.
    """

    if not trusted:
        raise ValueError(
            "Refusing to load executable pickle without trusted=True. "
            "Only load detector artifacts from a verified source."
        )

    in_path = Path(path)
    with in_path.open("rb") as f:
        detector = pickle.load(f)  # nosemgrep  # nosec B301 - trusted gate above

    if not is_pickle_safe_detector(detector):
        raise TypeError(
            "Loaded detector is not in the supported pickle-safe set. Refuse to return it."
        )

    return detector
