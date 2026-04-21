from __future__ import annotations

"""TorchScript helpers that suppress known upstream deprecation noise.

These wrappers keep current TorchScript-based workflows stable while avoiding
test and CLI noise from deprecation warnings emitted by recent torch versions.
"""

import warnings
from pathlib import Path
from typing import Any


def trace_module(model: Any, example_inputs: Any):
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit")
        return torch.jit.trace(model, example_inputs)


def freeze_module(module: Any):
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit")
        return torch.jit.freeze(module)


def load_module(path: str | Path, *, map_location: Any):
    import torch

    # TorchScript artifacts here are trusted local model files created by the
    # library's own export flows; bandit flags torch.jit.load generically.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit")
        return torch.jit.load(str(path), map_location=map_location)  # nosec B614


__all__ = [
    "freeze_module",
    "load_module",
    "trace_module",
]
