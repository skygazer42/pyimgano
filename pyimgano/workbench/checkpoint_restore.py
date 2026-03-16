from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _call_with_path_or_str(load_fn, path: Path) -> None:  # noqa: ANN001 - helper
    try:
        load_fn(str(path))
    except TypeError:
        load_fn(path)


def _try_detector_method(detector: Any, method_name: str, path: Path) -> bool:
    load_fn = getattr(detector, method_name, None)
    if not callable(load_fn):
        return False
    _call_with_path_or_str(load_fn, path)
    return True


def _ensure_detector_model(detector: Any) -> Any | None:
    model = getattr(detector, "model", None)
    if model is not None:
        return model

    build_fn = getattr(detector, "build_model", None)
    if not callable(build_fn):
        return None

    try:
        model = build_fn()
    except Exception as exc:
        raise ValueError(f"Failed to build detector model before checkpoint load: {exc}") from exc

    device = getattr(detector, "device", None)
    move_to = getattr(model, "to", None)
    if device is not None and callable(move_to):
        try:
            model = move_to(device)
        except Exception:
            pass

    try:
        setattr(detector, "model", model)
    except Exception:
        pass

    return model


def _normalize_torch_state_dict(state: object) -> object:
    if not isinstance(state, Mapping):
        return state

    if "model_state_dict" in state and isinstance(state["model_state_dict"], Mapping):
        return state["model_state_dict"]
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        return state["state_dict"]
    return state


def _try_load_state_dict(model: Any, path: Path) -> bool:
    if not callable(getattr(model, "load_state_dict", None)):
        return False

    from pyimgano.utils.optional_deps import require

    torch = require(
        "torch",
        extra="torch",
        purpose="load checkpoints via detector.model.load_state_dict",
    )

    state = torch.load(path, map_location="cpu")
    state = _normalize_torch_state_dict(state)
    try:
        model.load_state_dict(state)
    except Exception as exc:
        raise ValueError(f"Failed to load checkpoint into detector.model: {exc}") from exc
    return True


def _try_restore_serialized_detector(detector: Any, path: Path) -> bool:
    try:
        from pyimgano.models.serialization import load_model
    except Exception:
        return False

    try:
        loaded = load_model(path)
    except Exception:
        return False
    if loaded is None:
        return False

    from pyimgano.inference.runtime_wrappers import unwrap_runtime_detector

    loaded = unwrap_runtime_detector(loaded)
    restore_target = unwrap_runtime_detector(detector)

    if type(loaded) is not type(restore_target):
        raise ValueError(
            "Loaded serialized detector type does not match the constructed detector. "
            f"Loaded={type(loaded).__name__}, expected={type(restore_target).__name__}"
        )

    try:
        restore_target.__dict__.clear()
        restore_target.__dict__.update(loaded.__dict__)
    except Exception as exc:
        raise ValueError(f"Failed to restore serialized detector state: {exc}") from exc
    return True


def load_checkpoint_into_detector(detector: Any, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if _try_detector_method(detector, "load_checkpoint", path):
        return

    if _try_detector_method(detector, "load", path):
        return

    model = _ensure_detector_model(detector)
    if model is not None and _try_load_state_dict(model, path):
        return

    if _try_restore_serialized_detector(detector, path):
        return

    raise NotImplementedError(
        "Unable to load checkpoint into detector. Expected one of:\n"
        "- `detector.load_checkpoint(path)`\n"
        "- `detector.load(path)`\n"
        "- `detector.build_model()` + `detector.model.load_state_dict(...)`\n"
        "- `detector.model.load_state_dict(...)` (torch)\n"
        "- a joblib-serialized detector object matching the requested model\n"
    )


__all__ = ["load_checkpoint_into_detector"]
