from __future__ import annotations

"""OpenCLIP backend model skeletons.

This module intentionally has **no hard dependency** on `open_clip_torch`. The
OpenCLIP Python module (`open_clip`) is only imported at runtime when a model is
constructed, via `pyimgano.utils.optional_deps.require`.

The goal is to make `import pyimgano.models` safe even when optional deps are
not installed, while still registering model names so they can be discovered.
"""

from typing import Any, Optional

from pyimgano.utils.optional_deps import require

from .registry import register_model


def _require_open_clip(open_clip_module=None):
    if open_clip_module is not None:
        return open_clip_module
    return require("open_clip", extra="clip", purpose="OpenCLIP detectors")


@register_model(
    "vision_openclip_promptscore",
    tags=("vision", "openclip", "backend"),
    metadata={
        "description": "OpenCLIP prompt scoring detector (requires pyimgano[clip])",
        "backend": "openclip",
    },
)
class VisionOpenCLIPPromptScore:
    """Skeleton for an OpenCLIP prompt-score based vision detector.

    Parameters
    ----------
    open_clip_module : optional
        Dependency injection hook. When provided, avoids importing `open_clip`.
        Intended for unit tests and advanced callers.
    """

    def __init__(
        self,
        *,
        open_clip_module=None,
        **kwargs: Any,
    ) -> None:
        # Keep the runtime dependency check lazy so importing this module never
        # requires `open_clip` to be installed.
        self._open_clip = _require_open_clip(open_clip_module)
        self._kwargs = dict(kwargs)

    def fit(self, X, y=None):  # pragma: no cover - skeleton API
        raise NotImplementedError("OpenCLIP backend skeleton: fit() not implemented yet.")

    def decision_function(self, X):  # pragma: no cover - skeleton API
        raise NotImplementedError(
            "OpenCLIP backend skeleton: decision_function() not implemented yet."
        )


@register_model(
    "vision_openclip_patchknn",
    tags=("vision", "openclip", "backend", "knn"),
    metadata={
        "description": "OpenCLIP patch embedding + kNN detector (requires pyimgano[clip])",
        "backend": "openclip",
    },
)
class VisionOpenCLIPPatchKNN:
    """Skeleton for an OpenCLIP patch embedding + kNN detector.

    Parameters
    ----------
    open_clip_module : optional
        Dependency injection hook. When provided, avoids importing `open_clip`.
    """

    def __init__(
        self,
        *,
        open_clip_module=None,
        knn_index: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self._open_clip = _require_open_clip(open_clip_module)
        self._knn_index = knn_index
        self._kwargs = dict(kwargs)

    def fit(self, X, y=None):  # pragma: no cover - skeleton API
        raise NotImplementedError("OpenCLIP backend skeleton: fit() not implemented yet.")

    def decision_function(self, X):  # pragma: no cover - skeleton API
        raise NotImplementedError(
            "OpenCLIP backend skeleton: decision_function() not implemented yet."
        )

