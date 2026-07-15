"""DFM Gaussian deep-feature model for one-class image anomaly detection."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torchvision import transforms

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."


@register_model(
    "vision_dfm",
    tags=("vision", "deep", "dfm", "fast", "gaussian", "pca"),
    metadata={
        "description": "DFM PCA and full Gaussian-likelihood one-class adaptation",
        "paper": "Probabilistic Modeling of Deep Features for Out-of-Distribution and Adversarial Detection",
        "paper_url": "https://arxiv.org/abs/1909.11786",
        "year": 2019,
        "supervision": "one-class",
        "implementation_status": "paper-gaussian-one-class-adaptation",
        "paper_fidelity": "paper-adaptation",
        "speed": "very-fast",
        "training": "feature-statistics-only",
    },
)
class VisionDFM(BaseVisionDeepDetector):
    """Paper-aligned DFM Gaussian branch adapted to normal-only image data.

    The paper evaluates each selected network layer independently. This detector
    therefore accepts exactly one layer, average-pools it by four, retains 99.5%
    PCA variance, and fits the paper's separate-covariance Gaussian. With one
    normal class, that distribution becomes a single full Gaussian in PCA space.

    ``pretrained=False`` remains the offline-safe repository default. Enable
    ImageNet weights for a meaningful transferred feature representation.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        layers: list[str] | None = None,
        pretrained: bool = False,
        device: str | None = "cpu",
        *,
        layer: str | None = None,
        pooling_kernel_size: int = 4,
        pca_variance: float = 0.995,
        **kwargs: object,
    ) -> None:
        if backbone not in {"resnet18", "resnet50"}:
            raise ValueError("DFM supports 'resnet18' and 'resnet50'.")
        if layers is not None and layer is not None:
            raise ValueError("Pass either layer or legacy layers, not both.")
        if layers is not None:
            if len(layers) != 1:
                raise ValueError("DFM scores one paper layer at a time; layers must contain one item.")
            layer = str(layers[0])
        selected_layer = "layer4" if layer is None else str(layer)
        if pooling_kernel_size <= 0:
            raise ValueError("pooling_kernel_size must be positive.")
        if not 0.0 < pca_variance <= 1.0:
            raise ValueError("pca_variance must be in (0, 1].")

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        super().__init__(
            device=device,
            train_transform=transform,
            eval_transform=transform,
            **kwargs,
        )

        self.backbone_name = str(backbone)
        self.layer = selected_layer
        self.layers = [selected_layer]
        self.pretrained = bool(pretrained)
        self.pooling_kernel_size = int(pooling_kernel_size)
        self.pca_variance = float(pca_variance)

        self.feature_map: torch.Tensor | None = None
        self.pca_mean_: NDArray[np.float64] | None = None
        self.pca_components_: NDArray[np.float64] | None = None
        self.gaussian_mean_: NDArray[np.float64] | None = None
        self.gaussian_variance_: NDArray[np.float64] | None = None
        self._build_model()

    def _build_model(self) -> None:
        self.model, _ = load_torchvision_model(
            self.backbone_name,
            pretrained=self.pretrained,
        )
        if not hasattr(self.model, self.layer):
            raise ValueError(f"Backbone {self.backbone_name!r} has no layer {self.layer!r}.")
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)

        def capture_feature(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            self.feature_map = output.detach()

        getattr(self.model, self.layer).register_forward_hook(capture_feature)

    def _extract_features(self, image_path: str) -> NDArray[np.float64]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.eval_transform(image).unsqueeze(0).to(self.device)

        self.feature_map = None
        with torch.no_grad():
            self.model(image_tensor)
        feature = self.feature_map
        if feature is None or feature.ndim != 4:
            raise RuntimeError(f"DFM layer {self.layer!r} did not return a 4-D feature map.")
        if min(feature.shape[-2:]) < self.pooling_kernel_size:
            raise ValueError(
                f"DFM pooling kernel {self.pooling_kernel_size} exceeds feature map "
                f"size {tuple(feature.shape[-2:])}."
            )
        if self.pooling_kernel_size > 1:
            feature = F.avg_pool2d(feature, kernel_size=self.pooling_kernel_size)
        return feature.flatten(start_dim=1)[0].cpu().numpy().astype(np.float64, copy=False)

    def _fit_density(self, features: NDArray[np.float64]) -> None:
        self.pca_mean_ = features.mean(axis=0)
        centered = features - self.pca_mean_
        _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
        explained = np.square(singular_values)
        total_variance = float(explained.sum())
        if total_variance <= np.finfo(np.float64).eps:
            raise ValueError("DFM requires at least two distinct normal feature vectors.")
        retained = int(
            np.searchsorted(
                np.cumsum(explained) / total_variance,
                self.pca_variance,
                side="left",
            )
            + 1
        )
        self.pca_components_ = components[:retained]
        reduced = centered @ self.pca_components_.T
        self.gaussian_mean_ = reduced.mean(axis=0)
        # PCA diagonalizes the sample covariance, so these are the exact
        # eigenvalues of the paper's full Gaussian in the retained subspace.
        variance = np.mean((reduced - self.gaussian_mean_) ** 2, axis=0)
        self.gaussian_variance_ = np.maximum(variance, np.finfo(np.float64).eps)

    def _check_fitted(self) -> None:
        if any(
            getattr(self, name, None) is None
            for name in (
                "pca_mean_",
                "pca_components_",
                "gaussian_mean_",
                "gaussian_variance_",
            )
        ):
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

    def _score_features(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        self._check_fitted()
        assert self.pca_mean_ is not None
        assert self.pca_components_ is not None
        assert self.gaussian_mean_ is not None
        assert self.gaussian_variance_ is not None
        reduced = (features - self.pca_mean_) @ self.pca_components_.T
        centered = reduced - self.gaussian_mean_
        mahalanobis = np.sum(np.square(centered) / self.gaussian_variance_, axis=1)
        log_determinant = float(np.log(self.gaussian_variance_).sum())
        dimensions = int(self.gaussian_variance_.size)
        return 0.5 * (mahalanobis + log_determinant + dimensions * math.log(2.0 * math.pi))

    def fit(
        self,
        x: object = MISSING,
        y: NDArray | None = None,
        **kwargs: object,
    ) -> "VisionDFM":
        del y
        paths = list(cast(Iterable[str], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if len(paths) < 2:
            raise ValueError("DFM requires at least two normal training images.")

        features = np.vstack([self._extract_features(path) for path in paths])
        self._fit_density(features)
        self.decision_scores_ = self._score_features(features)
        self._process_decision_scores()
        self.is_fitted_ = True
        return self

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        self._check_fitted()
        scores = self.decision_function(
            cast(Iterable[str], resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        )
        return (scores >= self.threshold_).astype(int)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: int | None = None,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        self._check_fitted()
        paths = list(
            cast(
                Iterable[str],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        if not paths:
            return np.empty(0, dtype=np.float64)
        features = np.vstack([self._extract_features(path) for path in paths])
        return self._score_features(features).astype(np.float64, copy=False)
