from __future__ import annotations

"""Torchvision model loading helpers (safe defaults).

Why this exists
---------------
Torchvision has multiple APIs across versions:
- Older: `models.resnet18(pretrained=True)`
- Newer: `models.get_model(name, weights=WeightsEnum.DEFAULT)`

Industrial constraints:
- **No implicit network downloads by default**
- Keep `import pyimgano` / registry discovery lightweight

This module centralizes "safe" loading so our feature extractors and models
behave consistently across environments.
"""

from typing import Any


def load_torchvision_model(name: str, *, pretrained: bool):
    """Load a torchvision model with a best-effort transform.

    Notes
    -----
    - By default, callers should pass `pretrained=False` to avoid implicit
      downloads. If `pretrained=True`, torchvision may download weights via
      `torch.hub` when weights are not already cached.
    """

    from pyimgano.utils.optional_deps import require

    models = require("torchvision.models", extra="torch", purpose="torchvision model loading")

    model_name = str(name).strip()

    if hasattr(models, "get_model") and hasattr(models, "get_model_weights"):
        weights = None
        weight_transform = None
        if pretrained:
            weights_enum = models.get_model_weights(model_name)
            weights = weights_enum.DEFAULT
            weight_transform = weights.transforms()
        model = models.get_model(model_name, weights=weights)
        return model, weight_transform

    # Fallback for older torchvision.
    ctor = getattr(models, model_name)
    model = ctor(pretrained=bool(pretrained))
    return model, None


def strip_classification_head(model: Any):
    """Replace common classification heads with identity to expose embeddings."""

    from pyimgano.utils.optional_deps import require

    nn = require("torch.nn", extra="torch", purpose="torchvision backbone head stripping")

    if hasattr(model, "fc"):
        model.fc = nn.Identity()  # type: ignore[attr-defined]
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()  # type: ignore[attr-defined]
    elif hasattr(model, "head"):
        model.head = nn.Identity()  # type: ignore[attr-defined]
    elif hasattr(model, "heads"):
        # Torchvision VisionTransformer uses `.heads`.
        model.heads = nn.Identity()  # type: ignore[attr-defined]
    return model


def load_torchvision_backbone(name: str, *, pretrained: bool):
    """Load a torchvision backbone with classifier head removed."""

    model, transform = load_torchvision_model(str(name), pretrained=bool(pretrained))
    model = strip_classification_head(model)
    return model, transform
