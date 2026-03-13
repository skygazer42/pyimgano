from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def load_checkpoint_into_detector(detector: Any, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    from pyimgano.inference.runtime_wrappers import unwrap_runtime_detector

    load_fn = getattr(detector, "load_checkpoint", None)
    if callable(load_fn):
        try:
            load_fn(str(path))
        except TypeError:
            load_fn(path)
        return

    load_fn = getattr(detector, "load", None)
    if callable(load_fn):
        try:
            load_fn(str(path))
        except TypeError:
            load_fn(path)
        return

    model = getattr(detector, "model", None)
    if model is None:
        build_fn = getattr(detector, "build_model", None)
        if callable(build_fn):
            try:
                model = build_fn()
            except Exception as exc:
                raise ValueError(
                    f"Failed to build detector model before checkpoint load: {exc}"
                ) from exc

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

    if model is not None and callable(getattr(model, "load_state_dict", None)):
        from pyimgano.utils.optional_deps import require

        torch = require(
            "torch",
            extra="torch",
            purpose="load checkpoints via detector.model.load_state_dict",
        )

        state = torch.load(path, map_location="cpu")
        if isinstance(state, Mapping):
            if "model_state_dict" in state and isinstance(state["model_state_dict"], Mapping):
                state = state["model_state_dict"]
            elif "state_dict" in state and isinstance(state["state_dict"], Mapping):
                state = state["state_dict"]

        try:
            model.load_state_dict(state)
        except Exception as exc:
            raise ValueError(f"Failed to load checkpoint into detector.model: {exc}") from exc
        return

    try:
        from pyimgano.models.serialization import load_model

        loaded = load_model(path)
    except Exception:
        loaded = None

    if loaded is not None:
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
