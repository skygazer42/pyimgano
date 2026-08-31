from __future__ import annotations

"""Safe serialization for detectors whose fitted state is structured data."""

from pathlib import Path
from typing import Any, Mapping

from pyimgano.serialization.safe_checkpoint import (
    SafeCheckpointError,
    load_safe_checkpoint,
    save_safe_checkpoint,
)

DETECTOR_STATE_FORMAT = "pyimgano.detector-state"
DETECTOR_STATE_VERSION = 1
DETECTOR_STATE_COMPLETENESS = "unknown"


def _detector_type_name(detector: Any) -> str:
    detector_type = type(detector)
    return f"{detector_type.__module__}.{detector_type.__qualname__}"


def save_safe_detector_state(detector: Any, path: str | Path) -> Path:
    """Save legacy structured detector state without executable pickle data.

    This compatibility format is always marked ``completeness=unknown``.  Generic
    ``__dict__`` replay cannot certify that every fitted field is present and must
    never be promoted to a trained-export checkpoint merely because it loads.
    New portable artifacts use registered codecs from
    :mod:`pyimgano.exporting.state_codec` instead.
    """

    state = getattr(detector, "__dict__", None)
    if not isinstance(state, Mapping):
        raise SafeCheckpointError("Detector does not expose a mapping state dictionary.")
    return save_safe_checkpoint(
        {
            "format": DETECTOR_STATE_FORMAT,
            "version": DETECTOR_STATE_VERSION,
            "completeness": DETECTOR_STATE_COMPLETENESS,
            "detector_type": _detector_type_name(detector),
            "state": dict(state),
        },
        path,
    )


def load_safe_detector_state(detector: Any, path: str | Path) -> None:
    """Restore a safe detector-state archive into an exact detector type."""

    payload = load_safe_checkpoint(path)
    if payload.get("format") != DETECTOR_STATE_FORMAT:
        raise SafeCheckpointError("Checkpoint is not a detector-state archive.")
    if int(payload.get("version", -1)) != DETECTOR_STATE_VERSION:
        raise SafeCheckpointError("Unsupported detector-state archive version.")

    expected_type = _detector_type_name(detector)
    actual_type = str(payload.get("detector_type", ""))
    if actual_type != expected_type:
        raise SafeCheckpointError(
            "Safe detector-state type does not match the constructed detector. "
            f"Loaded={actual_type!r}, expected={expected_type!r}"
        )

    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise SafeCheckpointError("Safe detector-state payload is missing a mapping state.")

    target_state = getattr(detector, "__dict__", None)
    if not isinstance(target_state, dict):
        raise SafeCheckpointError("Detector state cannot be restored in place.")
    target_state.clear()
    target_state.update(dict(state))


def inspect_safe_detector_state(path: str | Path) -> dict[str, Any]:
    """Return non-executable metadata without upgrading legacy completeness."""

    payload = load_safe_checkpoint(path)
    if payload.get("format") != DETECTOR_STATE_FORMAT:
        raise SafeCheckpointError("Checkpoint is not a detector-state archive.")
    if int(payload.get("version", -1)) != DETECTOR_STATE_VERSION:
        raise SafeCheckpointError("Unsupported detector-state archive version.")
    return {
        "format": DETECTOR_STATE_FORMAT,
        "version": DETECTOR_STATE_VERSION,
        "completeness": DETECTOR_STATE_COMPLETENESS,
        "detector_type": str(payload.get("detector_type", "")),
        "serialization": "safe-data",
    }


__all__ = [
    "DETECTOR_STATE_FORMAT",
    "DETECTOR_STATE_COMPLETENESS",
    "DETECTOR_STATE_VERSION",
    "inspect_safe_detector_state",
    "load_safe_detector_state",
    "save_safe_detector_state",
]
