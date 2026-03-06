from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(detector: Any, path: str | Path) -> Path:
    """Best-effort checkpoint saving for recipe/workbench runs.

    Priority:
    1) `detector.save_checkpoint(path)` when present.
    2) `detector.model.state_dict()` saved via `torch.save(...)` when present.
    3) serialize the detector object itself via joblib when possible.
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from pyimgano.inference.runtime_wrappers import unwrap_runtime_detector

    checkpoint_target = unwrap_runtime_detector(detector)

    save_fn = getattr(checkpoint_target, "save_checkpoint", None)
    if callable(save_fn):
        try:
            save_fn(str(out_path))
        except TypeError:
            save_fn(out_path)
        return out_path

    model = getattr(checkpoint_target, "model", None)
    if model is not None and callable(getattr(model, "state_dict", None)):
        from pyimgano.utils.optional_deps import require

        torch = require(
            "torch",
            extra="torch",
            purpose="save checkpoints via detector.model.state_dict",
        )

        state = model.state_dict()
        torch.save(state, out_path)
        return out_path

    try:
        from pyimgano.models.serialization import save_model

        return save_model(checkpoint_target, out_path)
    except Exception as exc:
        raise NotImplementedError(
            "Detector does not support checkpoint saving. Expected one of:\n"
            "- `detector.save_checkpoint(path)`\n"
            "- `detector.model` with a torch-style `state_dict()`\n"
            "- a joblib-serializable detector object\n"
        ) from exc
