from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def resolve_torch_device(device: str | None) -> "object":
    """Resolve a torch device from a string (best-effort).

    Returns a `torch.device` instance, but is annotated as object to avoid
    hard mypy dependency on torch types.
    """

    from pyimgano.utils.optional_deps import require

    torch = require("torch", extra="torch", purpose="resolve_torch_device")

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev = str(device).strip().lower()
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(dev)


@contextmanager
def torch_inference(model=None) -> Iterator[None]:  # noqa: ANN001 - torch is optional in type system
    """Context manager for safe torch inference (eval + inference_mode).

    - Sets `model.eval()` if a model is provided.
    - Uses `torch.inference_mode()` (stronger than `no_grad()`).
    - Restores training mode afterwards.
    """

    from pyimgano.utils.optional_deps import require

    torch = require("torch", extra="torch", purpose="torch_inference")

    was_training = None
    if model is not None:
        was_training = bool(getattr(model, "training", False))
        try:
            model.eval()
        except Exception:
            was_training = None

    with torch.inference_mode():
        yield

    if model is not None and was_training is True:
        try:
            model.train()
        except Exception:
            pass
