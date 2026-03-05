from __future__ import annotations

"""Torch AMP (mixed precision) helpers.

Goals:
- opt-in only (disabled by default)
- safe across CPU/GPU
- avoid import-time dependency explosions (torch is optional in pyimgano core)
"""

from contextlib import nullcontext
from typing import Any, ContextManager, Literal

_AmpDType = Literal["float16", "bfloat16"]


def _as_device_type(device: Any) -> str:
    # Accept torch.device, strings, or None.
    if device is None:
        return "cpu"
    s = str(device).lower()
    if "cuda" in s:
        return "cuda"
    if "mps" in s:
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: _AmpDType, *, device_type: str) -> Any:
    import torch

    key = str(dtype).lower().strip()
    if key == "float16":
        # CPU float16 autocast is not universally supported; bfloat16 is safer.
        if device_type == "cpu":
            return torch.bfloat16
        return torch.float16
    if key == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown AMP dtype: {dtype!r}")


def amp_autocast(
    *,
    enabled: bool,
    device: Any,
    dtype: _AmpDType = "float16",
) -> ContextManager[None]:
    """Return a context manager enabling AMP autocast when supported."""

    if not bool(enabled):
        return nullcontext()

    try:
        import torch
    except Exception:  # pragma: no cover - depends on optional torch
        return nullcontext()

    device_type = _as_device_type(device)
    dt = _resolve_dtype(dtype, device_type=device_type)

    # torch.autocast preferred (works across device types on newer torch).
    autocast = getattr(torch, "autocast", None)
    if callable(autocast):
        try:
            return autocast(device_type=device_type, dtype=dt, enabled=True)
        except TypeError:
            # Older torch signature differences.
            return autocast(device_type=device_type, dtype=dt)

    # Fallback for very old versions: cuda-only autocast.
    cuda_amp = getattr(torch.cuda, "amp", None)
    if cuda_amp is not None:
        ac = getattr(cuda_amp, "autocast", None)
        if callable(ac) and device_type == "cuda":
            return ac(dtype=dt, enabled=True)

    return nullcontext()


__all__ = ["amp_autocast"]
