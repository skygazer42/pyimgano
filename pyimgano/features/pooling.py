from __future__ import annotations

"""Pooling helpers for embedding extractors.

This module provides small, dependency-stable pooling utilities used by
torch/torchvision feature extractors.

We keep it torch-only and import torch lazily to avoid heavy imports at
`import pyimgano`.
"""


def gem_pool2d(x, *, p: float = 3.0, eps: float = 1e-6):  # noqa: ANN001, ANN201 - torch boundary
    """Generalized Mean (GeM) pooling over spatial dimensions.

    Parameters
    ----------
    x:
        Torch tensor of shape (N, C, H, W).
    p:
        GeM exponent. Typical values are in [1, 6]. `p=1` approximates average
        pooling; larger values approach max pooling.
    eps:
        Clamp epsilon to keep gradients and numeric stability.
    """

    from pyimgano.utils.optional_deps import require

    torch = require("torch", extra="torch", purpose="gem_pool2d")

    p_f = float(p)
    if p_f <= 0.0:
        raise ValueError(f"p must be > 0, got {p}")
    eps_f = float(eps)
    if eps_f <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    xt = torch.as_tensor(x)
    if xt.ndim != 4:
        raise ValueError(f"Expected x shape (N,C,H,W), got {tuple(xt.shape)}")

    # Clamp to keep the power stable for negative activations.
    xt = xt.clamp(min=eps_f)
    pooled = xt.pow(p_f).mean(dim=(-2, -1)).pow(1.0 / p_f)
    return pooled


__all__ = ["gem_pool2d"]
