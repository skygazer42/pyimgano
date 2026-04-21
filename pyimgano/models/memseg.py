from __future__ import annotations

"""
MemSeg: Memory-Guided Semantic Segmentation for Anomaly Detection.

Paper: Memory-guided anomaly detection (various implementations)
Concept: Uses memory banks with attention mechanisms for anomaly segmentation

Key Features:
- Memory-guided attention
- Semantic segmentation approach
- Multi-scale memory banks
- Strong localization
- Efficient inference
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._image_batch import coerce_rgb_image_batch
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .deep_io import export_module_state_dict, safe_torch_load
from .registry import register_model

logger = logging.getLogger(__name__)


class MemoryBank(nn.Module):
    """Memory bank for storing normal feature patterns."""

    def __init__(self, memory_size: int, feature_dim: int):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim

        # Initialize memory bank
        self.register_buffer("memory", torch.randn(memory_size, feature_dim))
        self.memory_filled = 0

    def update(self, features: torch.Tensor):
        """Update memory bank with new features.

        Args:
            features: Feature vectors (N, D).
        """
        batch_size = features.size(0)

        if self.memory_filled < self.memory_size:
            # Fill memory bank
            end_idx = min(self.memory_filled + batch_size, self.memory_size)
            self.memory[self.memory_filled : end_idx] = features[: end_idx - self.memory_filled]
            self.memory_filled = end_idx
        else:
            # Random replacement
            indices = torch.randperm(self.memory_size)[:batch_size]
            self.memory[indices] = features

    def query(self, queries: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query memory bank for k nearest neighbors.

        Args:
            queries: Query features (N, D).
            k: Number of nearest neighbors.

        Returns:
            Tuple of (distances, indices).
        """
        # Normalize features
        queries = F.normalize(queries, dim=1)
        memory = F.normalize(self.memory, dim=1)

        # Compute similarities
        similarities = torch.mm(queries, memory.t())  # (N, M)

        # Get top-k
        values, indices = torch.topk(similarities, k=k, dim=1)

        # Convert similarities to distances
        distances = 1.0 - values

        return distances, indices


class FeatureExtractorWithMemory(nn.Module):
    """Feature extractor with memory-guided attention."""

    def __init__(
        self,
        backbone: str = "resnet18",
        memory_size: int = 1000,
        pretrained: bool = False,
    ):
        super().__init__()

        # Load backbone
        if backbone == "resnet18":
            resnet, _ = load_torchvision_model("resnet18", pretrained=bool(pretrained))
            self.feature_dim = 512
        elif backbone == "resnet34":
            resnet, _ = load_torchvision_model("resnet34", pretrained=bool(pretrained))
            self.feature_dim = 512
        elif backbone == "resnet50":
            resnet, _ = load_torchvision_model("resnet50", pretrained=bool(pretrained))
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Extract intermediate features
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])

        # Memory banks for different scales
        self.memory_banks = nn.ModuleDict(
            {
                "layer2": MemoryBank(memory_size, 128 if backbone.startswith("resnet1") else 512),
                "layer3": MemoryBank(memory_size, 256 if backbone.startswith("resnet1") else 1024),
                "layer4": MemoryBank(memory_size, self.feature_dim),
            }
        )

    def forward(self, x: torch.Tensor, update_memory: bool = False):
        """Forward pass with memory attention.

        Args:
            x: Input tensor (B, 3, H, W).
            update_memory: Whether to update memory banks.

        Returns:
            Multi-scale features and anomaly scores.
        """
        # Extract features
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = {"layer2": x2, "layer3": x3, "layer4": x4}

        if update_memory:
            # Update memory banks during training
            with torch.no_grad():
                for layer_name, feat in features.items():
                    # Global average pooling
                    pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze()
                    if pooled.dim() == 1:
                        pooled = pooled.unsqueeze(0)
                    self.memory_banks[layer_name].update(pooled)

        return features

    def compute_anomaly_scores(self, features: dict, k: int = 3) -> dict:
        """Compute anomaly scores using memory banks.

        Args:
            features: Dictionary of features for each layer.
            k: Number of nearest neighbors.

        Returns:
            Dictionary of anomaly scores.
        """
        anomaly_scores = {}

        for layer_name, feat in features.items():
            b, c, h, w = feat.shape

            # Reshape to (B*H*W, C)
            feat_reshaped = feat.permute(0, 2, 3, 1).reshape(b * h * w, c)

            # Query memory
            distances, _ = self.memory_banks[layer_name].query(feat_reshaped, k=k)

            # Average distance as anomaly score
            scores = distances.mean(dim=1)

            # Reshape back
            scores = scores.reshape(b, h, w)

            anomaly_scores[layer_name] = scores

        return anomaly_scores


class SegmentationHead(nn.Module):
    """Segmentation head for anomaly localization."""

    def __init__(self, feature_dims: List[int], output_dim: int = 1):
        super().__init__()

        # Projection layers for each scale
        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(dim, 64, kernel_size=1) for dim in feature_dims]
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * len(feature_dims), 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]):
        """Forward pass.

        Args:
            features: List of feature tensors.
            target_size: Target output size.

        Returns:
            Segmentation map.
        """
        # Project and upsample all features
        projected = []
        for feat, proj in zip(features, self.proj_layers):
            proj_feat = proj(feat)
            upsampled = F.interpolate(
                proj_feat, size=target_size, mode="bilinear", align_corners=False
            )
            projected.append(upsampled)

        # Concatenate and fuse
        fused = torch.cat(projected, dim=1)
        output = self.fusion(fused)

        return output


@register_model(
    "vision_memseg",
    tags=("vision", "deep", "memseg", "memory", "segmentation", "pixel_map"),
    metadata={
        "description": "MemSeg - memory-guided anomaly segmentation (ICCV 2022-style)",
        "year": 2022,
    },
)
@register_model(
    "memseg",
    tags=("vision", "deep", "memseg", "memory", "segmentation", "pixel_map"),
    metadata={
        "description": "MemSeg (legacy alias) - memory-guided anomaly segmentation",
        "year": 2022,
    },
)
class MemSegDetector(BaseVisionDeepDetector):
    """Memory-guided segmentation anomaly detector.

    Uses memory banks with attention mechanisms for anomaly detection
    and segmentation.

    Args:
        backbone: Feature extraction backbone ("resnet18", "resnet34", "resnet50").
        memory_size: Size of memory banks.
        k_neighbors: Number of nearest neighbors for scoring.
        pretrained: Whether to use pretrained backbone.
        use_segmentation_head: Whether to use segmentation head.
        device: Device to use ("cuda" or "cpu").

    References:
        Various memory-guided anomaly detection approaches.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        memory_size: int = 1000,
        k_neighbors: int = 3,
        pretrained: bool = False,
        use_segmentation_head: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_name = backbone
        self.memory_size = memory_size
        self.k_neighbors = k_neighbors
        self.pretrained = pretrained
        self.use_segmentation_head = use_segmentation_head

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

    def _build_model(self):
        """Build the MemSeg model."""
        # Feature extractor with memory
        self.feature_extractor = FeatureExtractorWithMemory(
            self.backbone_name,
            self.memory_size,
            self.pretrained,
        ).to(self.device)

        # Segmentation head
        if self.use_segmentation_head:
            if self.backbone_name.startswith("resnet1"):
                feature_dims = [128, 256, 512]
            else:
                feature_dims = [512, 1024, 2048]

            self.seg_head = SegmentationHead(feature_dims).to(self.device)
        else:
            self.seg_head = None

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ):
        """Build memory banks from training images.

        Args:
            X: Training images (N, H, W, C).
            y: Not used (unsupervised).
        """
        legacy_kwargs = {}
        if "X" in kwargs:
            legacy_kwargs["X"] = kwargs.pop("X")
        x_array = coerce_rgb_image_batch(
            resolve_legacy_x_keyword(x, legacy_kwargs, method_name="fit")
        )
        del y, kwargs
        logger.info("Building MemSeg memory banks...")

        if x_array.max() > 1.0:
            x_array = x_array.astype(np.float32) / 255.0

        self.feature_extractor.eval()

        with torch.no_grad():
            for i, img in enumerate(x_array):
                if (i + 1) % 50 == 0:
                    logger.info("  Processing image %d/%d", i + 1, len(x_array))

                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

                # Extract features and update memory
                _ = self.feature_extractor(img_tensor, update_memory=True)

        logger.info("Memory banks built successfully!")
        logger.info(
            "  Memory filled: %d/%d",
            self.feature_extractor.memory_banks["layer4"].memory_filled,
            self.memory_size,
        )

    def predict_proba(self, x: object = MISSING, **kwargs: object) -> NDArray:
        """Predict anomaly scores.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            Anomaly scores.
        """
        legacy_kwargs = {}
        if "X" in kwargs:
            legacy_kwargs["X"] = kwargs.pop("X")
        x_array = coerce_rgb_image_batch(
            resolve_legacy_x_keyword(x, legacy_kwargs, method_name="predict_proba")
        )
        del kwargs
        if x_array.max() > 1.0:
            x_array = x_array.astype(np.float32) / 255.0

        self.feature_extractor.eval()
        scores = []

        with torch.no_grad():
            for img in x_array:
                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

                # Extract features
                features = self.feature_extractor(img_tensor, update_memory=False)

                # Compute anomaly scores
                anomaly_scores = self.feature_extractor.compute_anomaly_scores(
                    features, k=self.k_neighbors
                )

                # Aggregate scores
                score = torch.stack([s.max() for s in anomaly_scores.values()]).mean().item()
                scores.append(score)

        return np.array(scores)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: int | None = None,
        **kwargs: object,
    ) -> NDArray:
        del batch_size
        return np.asarray(self.predict_proba(x, **kwargs), dtype=np.float64).reshape(-1)

    def save_checkpoint(self, path: str | Path) -> Path:
        memory_filled = {
            layer: int(bank.memory_filled)
            for layer, bank in self.feature_extractor.memory_banks.items()
        }
        if not any(value > 0 for value in memory_filled.values()):
            raise RuntimeError("Model not fitted. Call fit() first.")

        from pyimgano.utils.optional_deps import require

        torch_runtime = require("torch", extra="torch", purpose="save MemSeg checkpoint")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "detector": "vision_memseg",
            "config": {
                "contamination": float(self.contamination),
                "backbone": str(self.backbone_name),
                "memory_size": int(self.memory_size),
                "k_neighbors": int(self.k_neighbors),
                "pretrained": bool(self.pretrained),
                "use_segmentation_head": bool(self.use_segmentation_head),
                "device": str(self.device),
            },
            "state": {
                "feature_extractor_state_dict": export_module_state_dict(self.feature_extractor),
                "seg_head_state_dict": (
                    export_module_state_dict(self.seg_head) if self.seg_head is not None else None
                ),
                "memory_filled": memory_filled,
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
        torch_runtime.save(payload, out_path)
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Invalid MemSeg checkpoint payload: expected a dict.")
        if str(payload.get("detector", "")) not in {"vision_memseg", "memseg"}:
            raise ValueError("Invalid MemSeg checkpoint payload: detector marker mismatch.")

        config = payload.get("config", None)
        if not isinstance(config, dict):
            raise ValueError("Invalid MemSeg checkpoint payload: missing config.")

        self.contamination = float(config.get("contamination", self.contamination))
        self.backbone_name = str(config.get("backbone", self.backbone_name))
        self.memory_size = int(config.get("memory_size", self.memory_size))
        self.k_neighbors = int(config.get("k_neighbors", self.k_neighbors))
        self.pretrained = bool(config.get("pretrained", self.pretrained))
        self.use_segmentation_head = bool(
            config.get("use_segmentation_head", self.use_segmentation_head)
        )
        self.device = torch.device(str(config.get("device", self.device)))
        self._build_model()

        state = payload.get("state", None)
        if not isinstance(state, dict):
            raise ValueError("Invalid MemSeg checkpoint payload: missing state.")

        feature_state = state.get("feature_extractor_state_dict", None)
        if not isinstance(feature_state, dict):
            raise ValueError("Invalid MemSeg checkpoint payload: missing feature extractor state.")
        self.feature_extractor.load_state_dict(dict(feature_state), strict=False)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        seg_head_state = state.get("seg_head_state_dict", None)
        if self.seg_head is not None and seg_head_state is not None:
            if not isinstance(seg_head_state, dict):
                raise ValueError(
                    "Invalid MemSeg checkpoint payload: invalid segmentation head state."
                )
            self.seg_head.load_state_dict(dict(seg_head_state), strict=False)
            self.seg_head.to(self.device)
            self.seg_head.eval()

        memory_filled = state.get("memory_filled", None)
        if not isinstance(memory_filled, dict):
            raise ValueError("Invalid MemSeg checkpoint payload: missing memory_filled.")
        for layer, value in memory_filled.items():
            if layer not in self.feature_extractor.memory_banks:
                continue
            self.feature_extractor.memory_banks[layer].memory_filled = int(value)

        if state.get("decision_scores_", None) is not None:
            self.decision_scores_ = np.asarray(state["decision_scores_"], dtype=np.float64)
        if state.get("threshold_", None) is not None:
            self.threshold_ = float(state["threshold_"])
        if state.get("labels_", None) is not None:
            self.labels_ = np.asarray(state["labels_"], dtype=np.int64)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> List[NDArray]:
        """Predict pixel-level anomaly maps.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            List of anomaly maps.
        """
        x_array = coerce_rgb_image_batch(
            resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        )
        if x_array.max() > 1.0:
            x_array = x_array.astype(np.float32) / 255.0

        self.feature_extractor.eval()
        if self.seg_head is not None:
            self.seg_head.eval()

        anomaly_maps = []

        with torch.no_grad():
            for img in x_array:
                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

                # Extract features
                features = self.feature_extractor(img_tensor, update_memory=False)

                if self.use_segmentation_head and self.seg_head is not None:
                    # Use segmentation head
                    feat_list = [features["layer2"], features["layer3"], features["layer4"]]
                    seg_map = self.seg_head(feat_list, img.shape[:2])
                    anomaly_map = seg_map.squeeze().cpu().numpy()
                else:
                    # Use memory-based scores
                    anomaly_scores = self.feature_extractor.compute_anomaly_scores(
                        features, k=self.k_neighbors
                    )

                    # Combine multi-scale scores
                    maps = []
                    for score_map in anomaly_scores.values():
                        score_map = score_map.unsqueeze(0).unsqueeze(0)
                        upsampled = F.interpolate(
                            score_map, size=img.shape[:2], mode="bilinear", align_corners=False
                        )
                        maps.append(upsampled.squeeze().cpu().numpy())

                    anomaly_map = np.mean(maps, axis=0)

                anomaly_maps.append(anomaly_map)

        return anomaly_maps

    def _preprocess(self, image: NDArray) -> torch.Tensor:
        """Preprocess image.

        Args:
            image: Input image (H, W, C) in [0, 1].

        Returns:
            Preprocessed tensor (C, H, W).
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis].repeat(3, axis=2)

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image
