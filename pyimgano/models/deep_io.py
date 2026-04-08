from __future__ import annotations

"""Best-effort save/load for deep detectors.

This is intentionally conservative:
- only saves torch `state_dict` + small JSON-friendly metadata
- does not bundle weights into the package (user provides checkpoint paths)
"""

from contextlib import nullcontext
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


def export_module_state_dict(module: Any) -> dict[str, object]:
    """Return a CPU-friendly copy of a torch module state dict."""

    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("Module does not expose state_dict().")

    normalized: dict[str, object] = {}
    for key, value in dict(state_dict()).items():
        detach = getattr(value, "detach", None)
        cpu = getattr(value, "cpu", None)
        if callable(detach) and callable(cpu):
            try:
                normalized[str(key)] = detach().cpu()
                continue
            except Exception:
                pass
        normalized[str(key)] = value
    return normalized


def safe_torch_load(path: str | Path, *, map_location: str | None = "cpu") -> Any:
    """Load a torch checkpoint in weights-only mode."""

    torch = _require_torch()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    try:
        try:
            import numpy as np

            safe_globals = [
                np._core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                type(np.dtype(np.float16)),
                type(np.dtype(np.float32)),
                type(np.dtype(np.float64)),
                type(np.dtype(np.int32)),
                type(np.dtype(np.int64)),
                type(np.dtype(np.uint8)),
                type(np.dtype(np.bool_)),
            ]
        except Exception:
            safe_globals = []

        safe_globals_ctx = getattr(torch.serialization, "safe_globals", None)
        ctx = safe_globals_ctx(safe_globals) if callable(safe_globals_ctx) else nullcontext()
        with ctx:
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


__all__ = [
    "export_module_state_dict",
    "save_deep_detector",
    "safe_torch_load",
    "load_deep_detector",
]
