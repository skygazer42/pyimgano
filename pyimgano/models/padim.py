"""
PaDiM: Patch Distribution Modeling for industrial anomaly detection.

PaDiM models per-location (patch) feature distributions from a pretrained
backbone and scores anomalies via Mahalanobis distance.

Notes for this implementation:
- Fits on `list[str]` image paths (unified pyimgano interface).
- Supports pixel-level anomaly maps via `get_anomaly_map()` and
  `predict_anomaly_map()`.
- Keeps the historical registry name `padim` for compatibility and adds the
  canonical name `vision_padim` to match docs/CLI expectations.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.random_projection import GaussianRandomProjection
from torchvision import models, transforms

from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)

ImageInput = Union[str, np.ndarray]


def _build_torchvision_backbone(name: str, *, pretrained: bool) -> torch.nn.Module:
    if name == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            return models.resnet18(weights=weights)
        except Exception:  # pragma: no cover - fallback for older torchvision
            return models.resnet18(pretrained=pretrained)
    if name == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            return models.resnet50(weights=weights)
        except Exception:  # pragma: no cover - fallback for older torchvision
            return models.resnet50(pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {name!r}. Choose from: resnet18, resnet50")


@register_model(
    "vision_padim",
    tags=("vision", "deep", "patch", "distribution", "numpy", "pixel_map"),
    metadata={
        "description": "PaDiM - patch distribution modeling (ECCV 2020-style)",
        "paper": "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization",
        "year": 2020,
    },
)
@register_model(
    "padim",
    tags=("vision", "deep", "patch", "distribution", "numpy", "pixel_map"),
    metadata={
        "description": "PaDiM (legacy alias) - patch distribution modeling",
        "year": 2020,
    },
)
class VisionPaDiM(BaseVisionDeepDetector):
    """PaDiM-style anomaly detector (feature distribution per patch location)."""

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = "resnet18",
        d_reduced: int = 128,
        image_size: int = 224,
        pretrained: bool = True,
        device: str = "cpu",
        projection_fit_samples: int = 10,
        covariance_eps: float = 0.01,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(contamination=contamination, **kwargs)

        if d_reduced < 1:
            raise ValueError(f"d_reduced must be >= 1, got {d_reduced}")
        if image_size < 32:
            raise ValueError(f"image_size must be >= 32, got {image_size}")
        if projection_fit_samples < 1:
            raise ValueError(f"projection_fit_samples must be >= 1, got {projection_fit_samples}")
        if covariance_eps <= 0:
            raise ValueError(f"covariance_eps must be > 0, got {covariance_eps}")

        self.backbone_name = str(backbone)
        self.d_reduced = int(d_reduced)
        self.image_size = int(image_size)
        self.pretrained = bool(pretrained)
        self.device = str(device)
        self.projection_fit_samples = int(projection_fit_samples)
        self.covariance_eps = float(covariance_eps)
        self.random_state = int(random_state)

        self.model = _build_torchvision_backbone(self.backbone_name, pretrained=self.pretrained)
        self.model.eval()
        self.model.to(self.device)

        self.feature_maps: dict[str, torch.Tensor] = {}
        self._register_hooks()

        self.random_projection = GaussianRandomProjection(
            n_components=self.d_reduced,
            random_state=self.random_state,
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.means: Optional[NDArray] = None
        self.inv_covs: Optional[NDArray] = None
        self.patch_shape: Optional[Tuple[int, int]] = None

    def _register_hooks(self) -> None:
        def get_activation(name: str):
            def hook(_module, _input, output):
                self.feature_maps[name] = output.detach()

            return hook

        for layer in ("layer2", "layer3"):
            if not hasattr(self.model, layer):
                raise ValueError(f"Backbone {self.backbone_name!r} has no layer {layer!r}")
            getattr(self.model, layer).register_forward_hook(get_activation(layer))

    def _load_image_rgb(self, image_path: ImageInput) -> NDArray:
        if isinstance(image_path, np.ndarray):
            if image_path.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB image, got dtype={image_path.dtype}")
            if image_path.ndim != 3 or image_path.shape[2] != 3:
                raise ValueError(f"Expected shape (H,W,3), got {image_path.shape}")
            return np.ascontiguousarray(image_path)

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _extract_patch_features(self, image_path: ImageInput) -> NDArray:
        img = self._load_image_rgb(image_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model(img_tensor)

        layer2_feat = self.feature_maps["layer2"]  # (1, C2, H2, W2)
        layer3_feat = self.feature_maps["layer3"]  # (1, C3, H3, W3)

        layer3_feat = F.interpolate(
            layer3_feat,
            size=layer2_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        features = torch.cat([layer2_feat, layer3_feat], dim=1)  # (1, C, H, W)
        _b, c, h, w = features.shape
        self.patch_shape = (int(h), int(w))

        # (H*W, C)
        return features.permute(0, 2, 3, 1).reshape(h * w, c).cpu().numpy()

    def _describe_input(self, item: ImageInput, idx: int) -> str:
        if isinstance(item, str):
            return item
        return f"<ndarray:{idx} shape={tuple(item.shape)} dtype={item.dtype}>"

    def fit(self, X: Iterable[ImageInput], y: Optional[NDArray] = None) -> "VisionPaDiM":
        X_list = list(X)
        if not X_list:
            raise ValueError("Training set cannot be empty")

        # Fit random projection on a small subset for speed/stability.
        proj_fit = []
        for image in X_list[: min(self.projection_fit_samples, len(X_list))]:
            proj_fit.append(self._extract_patch_features(image))
        proj_fit_mat = np.vstack(proj_fit)
        self.random_projection.fit(proj_fit_mat)

        reduced_features: list[NDArray] = []
        for idx, image in enumerate(X_list):
            try:
                feat = self._extract_patch_features(image)
                reduced = self.random_projection.transform(feat)
                reduced_features.append(reduced)
            except Exception as exc:
                logger.warning(
                    "Failed to process %s: %s",
                    self._describe_input(image, idx),
                    exc,
                )

        if not reduced_features:
            raise ValueError("Failed to extract features from any training image")

        all_features = np.stack(reduced_features, axis=0)  # (N, P, D)
        means = np.mean(all_features, axis=0).astype(np.float32, copy=False)  # (P, D)

        n_images, n_patches, d = all_features.shape
        if d != self.d_reduced:
            raise RuntimeError(f"Expected reduced dim={self.d_reduced}, got {d}")

        inv_covs = np.empty((n_patches, d, d), dtype=np.float32)
        eye = np.eye(d, dtype=np.float32)

        # Per-location covariance. For small N, fall back to diagonal eps.
        for i in range(n_patches):
            patch_feats = all_features[:, i, :]
            if n_images < 2:
                cov = eye * self.covariance_eps
            else:
                cov = np.cov(patch_feats, rowvar=False).astype(np.float32, copy=False)
                cov = cov + eye * self.covariance_eps
            inv_covs[i] = np.linalg.inv(cov).astype(np.float32, copy=False)

        self.means = means
        self.inv_covs = inv_covs

        # Calibrate threshold for `predict()`.
        self.decision_scores_ = self.decision_function(X_list)
        self._process_decision_scores()
        return self

    def _check_fitted(self) -> None:
        if self.means is None or self.inv_covs is None or self.patch_shape is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def _compute_patch_distances(self, image_path: ImageInput) -> NDArray:
        self._check_fitted()
        feat = self._extract_patch_features(image_path)
        reduced = self.random_projection.transform(feat).astype(np.float32, copy=False)  # (P, D)

        diff = reduced - self.means  # (P, D)
        q = np.einsum("pd,pde,pe->p", diff, self.inv_covs, diff)
        q = np.maximum(q, 0.0)
        return np.sqrt(q, dtype=np.float32)

    def decision_function(self, X: Iterable[ImageInput]) -> NDArray:
        self._check_fitted()
        X_list = list(X)
        scores = np.zeros(len(X_list), dtype=np.float32)

        for i, image in enumerate(X_list):
            try:
                distances = self._compute_patch_distances(image)
                scores[i] = float(np.max(distances))
            except Exception as exc:
                logger.warning("Failed to score %s: %s", self._describe_input(image, i), exc)
                scores[i] = 0.0
        return scores

    def predict(self, X: Iterable[ImageInput]) -> NDArray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)

    def get_anomaly_map(self, image_path: ImageInput) -> NDArray:
        distances = self._compute_patch_distances(image_path)
        h, w = self.patch_shape or (0, 0)
        if h * w != distances.shape[0]:
            raise RuntimeError(
                f"Patch shape mismatch: {self.patch_shape} -> {h*w} != {distances.shape[0]}"
            )

        low_res = distances.reshape(h, w)
        return cv2.resize(
            low_res,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32, copy=False)

    def predict_anomaly_map(self, X: Iterable[ImageInput]) -> NDArray:
        maps = [self.get_anomaly_map(item) for item in X]
        return np.stack(maps, axis=0)
