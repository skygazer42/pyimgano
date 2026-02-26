"""Torch activation helpers.

We intentionally keep this small and dependency-free (beyond PyTorch) so our
deep detectors can share a consistent, user-friendly activation API.
"""

from __future__ import annotations

import torch.nn as nn


def get_activation_by_name(name: str) -> nn.Module:
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("Unknown activation: empty name")

    mapping: dict[str, nn.Module] = {
        "identity": nn.Identity(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "prelu": nn.PReLU(),
        "gelu": nn.GELU(),
    }

    act = mapping.get(key)
    if act is None:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown activation: {name!r}. Supported: {supported}")

    return act

