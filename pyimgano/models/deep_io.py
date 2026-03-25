from __future__ import annotations

"""Best-effort save/load for deep detectors.

This is intentionally conservative:
- only saves torch `state_dict` + small JSON-friendly metadata
- does not bundle weights into the package (user provides checkpoint paths)
"""

from pathlib import Path
from typing import Any


def _require_torch():
    from pyimgano.utils.optional_deps import require

    return require("torch", extra="torch", purpose="deep model IO")


def save_deep_detector(
    detector: Any, path: str | Path, *, meta: dict[str, Any] | None = None
) -> Path:
    """Save `detector.model.state_dict()` to disk."""

    torch = _require_torch()

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    model = getattr(detector, "model", None)
    if model is None:
        raise ValueError(
            "Detector has no `.model` attribute set; fit/build the model before saving."
        )

    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("Detector `.model` does not expose state_dict(); cannot save.")

    payload = {
        "state_dict": state_dict(),
        "meta": {
            "detector_class": f"{detector.__class__.__module__}.{detector.__class__.__name__}",
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            **(dict(meta or {})),
        },
    }
    torch.save(payload, str(p))
    return p


def safe_torch_load(path: str | Path, *, map_location: str | None = "cpu") -> Any:
    """Load a torch checkpoint in weights-only mode."""

    torch = _require_torch()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    try:
        return torch.load(str(p), map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe checkpoint loading requires a torch build that supports "
            "`torch.load(..., weights_only=True)`."
        ) from exc


def load_deep_detector(detector: Any, path: str | Path, *, map_location: str | None = "cpu") -> Any:
    """Load a saved `state_dict` into `detector.model` (best-effort)."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    payload = safe_torch_load(p, map_location=map_location)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Invalid checkpoint payload; expected dict with key 'state_dict'.")

    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint payload: state_dict must be a dict.")

    model = getattr(detector, "model", None)
    if model is None:
        build = getattr(detector, "build_model", None)
        if not callable(build):
            raise ValueError(
                "Detector has no `.model` and no build_model() method; cannot load checkpoint."
            )
        detector.model = build()
        model = detector.model

    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError("Detector `.model` does not expose load_state_dict(); cannot load.")

    load_state_dict(state_dict, strict=False)
    return detector


__all__ = ["save_deep_detector", "safe_torch_load", "load_deep_detector"]
