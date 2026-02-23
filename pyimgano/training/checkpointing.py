from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(detector: Any, path: str | Path) -> Path:
    """Best-effort checkpoint saving for recipe/workbench runs.

    Priority:
    1) `detector.save_checkpoint(path)` when present.
    2) `detector.model.state_dict()` saved via `torch.save(...)` when present.
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_fn = getattr(detector, "save_checkpoint", None)
    if callable(save_fn):
        try:
            save_fn(str(out_path))
        except TypeError:
            save_fn(out_path)
        return out_path

    model = getattr(detector, "model", None)
    if model is not None and callable(getattr(model, "state_dict", None)):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - dependency boundary
            raise ImportError(
                "torch is required to save checkpoints via `detector.model.state_dict()`.\n"
                "Install it via:\n"
                "  pip install torch"
            ) from exc

        state = model.state_dict()
        torch.save(state, out_path)
        return out_path

    raise NotImplementedError(
        "Detector does not support checkpoint saving. Expected one of:\n"
        "- `detector.save_checkpoint(path)`\n"
        "- `detector.model` with a torch-style `state_dict()`\n"
    )

