# -*- coding: utf-8 -*-
"""PyOD AnoGAN vision wrapper (optional pandas dependency).

PyOD's `AnoGAN` implementation imports `pandas` and `matplotlib` at runtime.
To keep `import pyimgano.models` lightweight, we import the PyOD module lazily
when constructing the detector.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from pyimgano.utils.optional_deps import optional_import

from .baseml import BaseVisionDetector
from .registry import register_model


def _load_pyod_anogan():
    module, error = optional_import("pyod.models.anogan")
    if module is None:
        raise ImportError(
            "PyOD AnoGAN backend is unavailable.\n"
            "It requires extra third-party deps (not installed by default), commonly:\n"
            "  pip install pandas matplotlib\n"
            "Original error: "
            f"{error}"
        ) from error

    cls = getattr(module, "AnoGAN", None)
    if cls is None:
        raise ImportError("pyod.models.anogan does not export AnoGAN")
    return cls


@register_model(
    "vision_anogan",
    tags=("vision", "deep", "gan", "anogan", "pyod"),
    metadata={"description": "PyOD AnoGAN wrapper (feature-based; requires pandas/matplotlib)"},
)
class VisionAnoGAN(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's `AnoGAN` detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        activation_hidden: str = "tanh",
        dropout_rate: float = 0.2,
        latent_dim_G: int = 2,
        G_layers: Optional[Sequence[int]] = None,
        D_layers: Optional[Sequence[int]] = None,
        verbose: int = 0,
        index_D_layer_for_recon_error: int = 1,
        epochs: int = 500,
        preprocessing: bool = False,
        learning_rate: float = 1e-3,
        learning_rate_query: float = 1e-2,
        epochs_query: int = 20,
        batch_size: int = 32,
        output_activation: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._detector_kwargs = {
            "activation_hidden": str(activation_hidden),
            "dropout_rate": float(dropout_rate),
            "latent_dim_G": int(latent_dim_G),
            "G_layers": list(G_layers) if G_layers is not None else [20, 10, 3, 10, 20],
            "verbose": int(verbose),
            "D_layers": list(D_layers) if D_layers is not None else [20, 10, 5],
            "index_D_layer_for_recon_error": int(index_D_layer_for_recon_error),
            "epochs": int(epochs),
            "preprocessing": bool(preprocessing),
            "learning_rate": float(learning_rate),
            "learning_rate_query": float(learning_rate_query),
            "epochs_query": int(epochs_query),
            "batch_size": int(batch_size),
            "output_activation": output_activation,
            "device": device,
        }

        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        AnoGAN = _load_pyod_anogan()
        return AnoGAN(contamination=self.contamination, **self._detector_kwargs)

