"""
SPADE: Sub-Image Anomaly Detection with Deep Pyramid Correspondences (ECCV 2020).

SPADE uses deep pyramid feature extraction followed by k-NN matching for
sub-image anomaly detection with strong pixel-level localization.

This implementation:
- Fits on `list[str]` image paths (unified pyimgano interface).
- Exposes pixel maps via `get_anomaly_map()` / `predict_anomaly_map()`.
- Keeps the historical registry name `spade` and adds the canonical name
  `vision_spade` to match docs/CLI expectations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


def _build_resnet_backbone(name: str, *, pretrained: bool) -> nn.Module:
    if name in {"wide_resnet50", "resnet50", "resnet18"}:
        model, _ = load_torchvision_model(name, pretrained=bool(pretrained))
        return model
    raise ValueError(f"Unknown backbone: {name!r}")


class DeepPyramidExtractor(nn.Module):
    """Deep pyramid feature extractor for SPADE."""

    def __init__(self, backbone: str = "wide_resnet50", *, pretrained: bool = False) -> None:
        super().__init__()

        resnet = _build_resnet_backbone(backbone, pretrained=pretrained)

        # Extract features at different scales
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Low-level
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # Mid-level
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # High-level

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3


@register_model(
    "vision_spade",
    tags=("vision", "deep", "spade", "knn", "numpy", "pixel_map"),
    metadata={
        "description": "SPADE - Deep pyramid k-NN localization (ECCV 2020)",
        "paper": "Sub-Image Anomaly Detection with Deep Pyramid Correspondences",
        "year": 2020,
    },
)
@register_model(
    "spade",
    tags=("vision", "deep", "spade", "knn", "numpy", "pixel_map"),
    metadata={
        "description": "SPADE (legacy alias) - Deep pyramid k-NN localization",
        "year": 2020,
    },
)
class VisionSPADEDetector(BaseVisionDeepDetector):
    """SPADE anomaly detector (path-based interface)."""

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = "wide_resnet50",
        pretrained: bool = False,
        image_size: int = 256,
        k_neighbors: int = 50,
        feature_levels: Sequence[str] = ("layer1", "layer2", "layer3"),
        align_features: bool = True,
        gaussian_sigma: float = 4.0,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(contamination=contamination, **kwargs)

        if image_size < 32:
            raise ValueError(f"image_size must be >= 32, got {image_size}")
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")

        self.backbone_name = str(backbone)
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)
        self.k_neighbors = int(k_neighbors)
        self.feature_levels = tuple(feature_levels)
        self.align_features = bool(align_features)
        self.gaussian_sigma = float(gaussian_sigma)

        if device is not None:
            device_str = device
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        self.feature_extractor = DeepPyramidExtractor(
            self.backbone_name,
            pretrained=self.pretrained,
        ).to(self.device)
        self.feature_extractor.eval()

        self.memory_bank: Optional[dict[str, NDArray]] = None
        self.kd_trees: Optional[dict[str, cKDTree]] = None

    def _load_image(self, image_path: str) -> NDArray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return (img.astype(np.float32) / 255.0).astype(np.float32, copy=False)

    def _iter_images(self, x: Union[Iterable[str], NDArray]) -> Iterable[NDArray]:
        if isinstance(x, str):
            raise TypeError("Expected an iterable of paths, got a single string path.")

        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                yield self._maybe_resize_image(x)
                return
            if x.ndim != 4:
                raise ValueError(f"Expected x shape (N,H,W,C) or (H,W,C); got {x.shape}")
            for img in x:
                yield self._maybe_resize_image(img)
            return

        for item in x:
            if isinstance(item, str):
                yield self._load_image(item)
            elif isinstance(item, np.ndarray):
                yield self._maybe_resize_image(item)
            else:
                raise TypeError(
                    "Expected x to be an iterable of str paths or numpy images, "
                    f"got element type {type(item)}"
                )

    def _maybe_resize_image(self, image: NDArray) -> NDArray:
        img = image
        if img.ndim == 2:
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return img

    def _preprocess(self, image: NDArray) -> torch.Tensor:
        # image: (H,W,C) in [0,1] float32
        img = self._maybe_resize_image(image)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img_t - mean) / std

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionSPADEDetector":
        legacy_kwargs: dict[str, object] = {}
        if "X" in kwargs:
            legacy_kwargs["X"] = kwargs.pop("X")
        del y, kwargs
        images = list(
            self._iter_images(
                cast(
                    Union[Iterable[str], NDArray],
                    resolve_legacy_x_keyword(x, legacy_kwargs, method_name="fit"),
                )
            )
        )
        if not images:
            raise ValueError("Training set cannot be empty")

        logger.info("Building SPADE memory bank on %d images", len(images))

        memory_bank: dict[str, list[NDArray]] = {level: [] for level in self.feature_levels}

        with torch.no_grad():
            for i, img in enumerate(images):
                if (i + 1) % 50 == 0:
                    logger.info("  Processing image %d/%d", i + 1, len(images))

                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)
                f1, f2, f3 = self.feature_extractor(img_tensor)
                feature_dict = {"layer1": f1, "layer2": f2, "layer3": f3}

                for level in self.feature_levels:
                    feat = feature_dict[level]
                    _, c, _, _ = feat.shape
                    feat = feat.permute(0, 2, 3, 1).reshape(-1, c)
                    memory_bank[level].append(feat.cpu().numpy())

        self.memory_bank = {level: np.vstack(chunks) for level, chunks in memory_bank.items()}
        self.kd_trees = {level: cKDTree(self.memory_bank[level]) for level in self.feature_levels}

        # Calibrate threshold for `predict()`.
        self.decision_scores_ = self.decision_function(images)
        self._process_decision_scores()
        return self

    def _check_fitted(self) -> None:
        if self.memory_bank is None or self.kd_trees is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        # SPADE scores one image at a time. Keep `batch_size` for interface
        # compatibility with BaseDeepLearningDetector.
        if batch_size is not None:
            batch_size_int = int(batch_size)
            if batch_size_int <= 0:
                raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")

        self._check_fitted()

        scores: list[float] = []
        for img in self._iter_images(
            cast(
                Union[Iterable[str], NDArray],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        ):
            anomaly_map = self._compute_anomaly_map(img)
            scores.append(float(anomaly_map.max()))

        return np.asarray(scores, dtype=np.float32)

    def predict_proba(self, x: object = MISSING, **kwargs: object) -> NDArray:
        # For SPADE, "probability" is an anomaly score; keep for compatibility.
        legacy_kwargs: dict[str, object] = {}
        if "X" in kwargs:
            legacy_kwargs["X"] = kwargs.pop("X")
        del kwargs
        return self.decision_function(
            cast(
                Union[Iterable[str], NDArray],
                resolve_legacy_x_keyword(x, legacy_kwargs, method_name="predict_proba"),
            )
        )

    def get_anomaly_map(self, image_path: str) -> NDArray:
        self._check_fitted()
        img = self._load_image(image_path)
        return self._compute_anomaly_map(img)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        self._check_fitted()
        maps = [
            self._compute_anomaly_map(img)
            for img in self._iter_images(
                cast(
                    Union[Iterable[str], NDArray],
                    resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
                )
            )
        ]
        return np.stack(maps, axis=0)

    def _compute_anomaly_map(self, image: NDArray) -> NDArray:
        self._check_fitted()

        with torch.no_grad():
            img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
            f1, f2, f3 = self.feature_extractor(img_tensor)
            feature_dict = {"layer1": f1, "layer2": f2, "layer3": f3}

            anomaly_maps: list[NDArray] = []
            for level in self.feature_levels:
                feat = feature_dict[level]
                _b, c, h, w = feat.shape

                feat = feat.permute(0, 2, 3, 1).reshape(h * w, c).cpu().numpy()
                if self.align_features:
                    feat = self._align_features(feat)

                distances, _ = self.kd_trees[level].query(feat, k=self.k_neighbors, workers=-1)
                distances_arr = np.asarray(distances, dtype=np.float32)
                if distances_arr.ndim == 1:
                    distances_arr = distances_arr.reshape(-1, 1)
                anomaly_scores = distances_arr.mean(axis=1).astype(np.float32, copy=False)
                anomaly_map = anomaly_scores.reshape(h, w)

                anomaly_map_t = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0)
                anomaly_map_t = F.interpolate(
                    anomaly_map_t,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_maps.append(anomaly_map_t.squeeze().cpu().numpy())

            final_map = np.mean(anomaly_maps, axis=0)
            if self.gaussian_sigma > 0:
                final_map = gaussian_filter(final_map, sigma=self.gaussian_sigma)

        return final_map.astype(np.float32, copy=False)

    def _align_features(self, features: NDArray) -> NDArray:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norms + 1e-8)

    def save_checkpoint(self, path: str | Path) -> Path:
        self._check_fitted()

        from pyimgano.models.serialization import save_model

        feature_state = self.feature_extractor.state_dict()
        normalized_feature_state = {}
        for key, value in dict(feature_state).items():
            detach = getattr(value, "detach", None)
            cpu = getattr(value, "cpu", None)
            if callable(detach) and callable(cpu):
                try:
                    normalized_feature_state[str(key)] = detach().cpu()
                    continue
                except Exception:
                    pass
            normalized_feature_state[str(key)] = value

        payload = {
            "schema_version": 1,
            "detector": "vision_spade",
            "config": {
                "contamination": float(self.contamination),
                "backbone": str(self.backbone_name),
                "pretrained": bool(self.pretrained),
                "image_size": int(self.image_size),
                "k_neighbors": int(self.k_neighbors),
                "feature_levels": list(self.feature_levels),
                "align_features": bool(self.align_features),
                "gaussian_sigma": float(self.gaussian_sigma),
                "device": str(self.device),
            },
            "state": {
                "feature_extractor_state_dict": normalized_feature_state,
                "memory_bank": {
                    str(level): np.asarray(bank, dtype=np.float32)
                    for level, bank in (self.memory_bank or {}).items()
                },
                "decision_scores_": (
                    np.asarray(self.decision_scores_, dtype=np.float64)
                    if getattr(self, "decision_scores_", None) is not None
                    else None
                ),
                "threshold_": (
                    float(self.threshold_)
                    if getattr(self, "threshold_", None) is not None
                    else None
                ),
                "labels_": (
                    np.asarray(self.labels_, dtype=np.int64)
                    if getattr(self, "labels_", None) is not None
                    else None
                ),
            },
        }
        return save_model(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        from pyimgano.models.serialization import load_model

        payload = load_model(path)
        if not isinstance(payload, dict):
            raise ValueError("Invalid SPADE checkpoint payload: expected a dict.")
        if str(payload.get("detector", "")) not in {"vision_spade", "spade"}:
            raise ValueError("Invalid SPADE checkpoint payload: detector marker mismatch.")

        config = dict(cast(dict[str, object], payload.get("config", {})))
        self.contamination = float(config.get("contamination", self.contamination))
        self.backbone_name = str(config.get("backbone", self.backbone_name))
        self.pretrained = bool(config.get("pretrained", self.pretrained))
        self.image_size = int(config.get("image_size", self.image_size))
        self.k_neighbors = int(config.get("k_neighbors", self.k_neighbors))
        self.feature_levels = tuple(config.get("feature_levels", self.feature_levels))
        self.align_features = bool(config.get("align_features", self.align_features))
        self.gaussian_sigma = float(config.get("gaussian_sigma", self.gaussian_sigma))

        device_value = config.get("device", None)
        if device_value is not None:
            self.device = torch.device(str(device_value))

        self.feature_extractor = DeepPyramidExtractor(
            self.backbone_name,
            pretrained=self.pretrained,
        ).to(self.device)
        self.feature_extractor.eval()

        state = payload.get("state", None)
        if not isinstance(state, dict):
            raise ValueError("Invalid SPADE checkpoint payload: missing state section.")

        feature_state = state.get("feature_extractor_state_dict", None)
        if isinstance(feature_state, dict):
            self.feature_extractor.load_state_dict(dict(feature_state), strict=False)

        memory_bank_payload = state.get("memory_bank", None)
        if not isinstance(memory_bank_payload, dict):
            raise ValueError("Invalid SPADE checkpoint payload: missing memory bank.")
        self.memory_bank = {
            str(level): np.asarray(bank, dtype=np.float32)
            for level, bank in memory_bank_payload.items()
        }
        self.kd_trees = {level: cKDTree(bank) for level, bank in self.memory_bank.items()}

        decision_scores = state.get("decision_scores_", None)
        self.decision_scores_ = (
            np.asarray(decision_scores, dtype=np.float64) if decision_scores is not None else None
        )
        threshold = state.get("threshold_", None)
        self.threshold_ = float(threshold) if threshold is not None else None
        labels = state.get("labels_", None)
        if labels is not None:
            self.labels_ = np.asarray(labels, dtype=np.int64)
