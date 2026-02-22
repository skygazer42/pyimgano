"""
DifferNet: A Learnable Difference Anomaly Detector.

Paper: https://arxiv.org/abs/2108.09810
Conference: WACV 2023

DifferNet learns to detect anomalies by modeling the difference between
a test image and its k-nearest neighbors in the feature space.

Key Features:
- Learnable difference module
- Memory bank of normal features
- Efficient k-NN search
- Good localization performance
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class DifferenceModule(nn.Module):
    """Learnable difference module for DifferNet."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)

        self.conv3 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute learnable difference.

        Args:
            x1: First feature map (B, C, H, W).
            x2: Second feature map (B, C, H, W).

        Returns:
            Difference map (B, 1, H, W).
        """
        # Concatenate features
        x = torch.cat([x1, x2], dim=1)

        # Learnable difference
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        return x


class FeatureExtractor(nn.Module):
    """Multi-scale feature extractor."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()

        # Load pretrained backbone
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # 64 channels
            self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # 128 channels
            self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # 256 channels
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # 256 channels
            self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # 512 channels
            self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # 1024 channels
        elif backbone == "wide_resnet50":
            resnet = models.wide_resnet50_2(pretrained=pretrained)
            self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # 256 channels
            self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # 512 channels
            self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # 1024 channels
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Tuple of (layer1, layer2, layer3) features.
        """
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return x1, x2, x3


@register_model(
    "vision_differnet",
    tags=("vision", "deep", "differnet", "knn", "wacv2023", "pixel_map"),
    metadata={
        "description": "DifferNet - learnable difference + kNN anomaly detection (WACV 2023)",
        "paper": "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows",
        "year": 2023,
    },
)
@register_model(
    "differnet",
    tags=("vision", "deep", "differnet", "knn", "wacv2023", "pixel_map"),
    metadata={
        "description": "DifferNet (legacy alias) - learnable difference + kNN anomaly detection",
        "year": 2023,
    },
)
class DifferNetDetector(BaseVisionDeepDetector):
    """DifferNet anomaly detector.

    Learns to detect anomalies by modeling differences between test images
    and their k-nearest neighbors in the feature space.

    Args:
        backbone: Feature extraction backbone ("resnet18", "resnet50", "wide_resnet50").
        pretrained: Whether to use pretrained backbone.
        k_neighbors: Number of nearest neighbors to use.
        feature_layer: Which layer to use for k-NN ("layer1", "layer2", "layer3", "all").
        train_difference: Whether to train the difference module.
        epochs: Number of training epochs for difference module.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        device: Device to use ("cuda" or "cpu").

    References:
        Rudolph et al. "Same Same But DifferNet: Semi-Supervised Defect Detection
        with Normalizing Flows." WACV 2021.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        pretrained: bool = True,
        k_neighbors: int = 5,
        feature_layer: str = "layer3",
        train_difference: bool = True,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.k_neighbors = k_neighbors
        self.feature_layer = feature_layer
        self.train_difference = train_difference
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

        # Memory bank
        self.memory_bank = None
        self.kd_tree = None

    def _build_model(self):
        """Build the DifferNet model."""
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            self.backbone_name,
            self.pretrained,
        ).to(self.device)

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Get feature dimensions
        if self.backbone_name == "resnet18":
            feature_dims = {"layer1": 64, "layer2": 128, "layer3": 256}
        else:  # resnet50, wide_resnet50
            feature_dims = {"layer1": 256, "layer2": 512, "layer3": 1024}

        # Difference modules
        if self.train_difference:
            self.diff_modules = nn.ModuleDict()
            for layer in ["layer1", "layer2", "layer3"]:
                self.diff_modules[layer] = DifferenceModule(
                    feature_dims[layer], out_channels=1
                ).to(self.device)

    def fit(self, X: NDArray, y: Optional[NDArray] = None, **kwargs):
        """Fit the DifferNet detector.

        Args:
            X: Training images (N, H, W, C) or (N, C, H, W).
            y: Not used (unsupervised).
        """
        # Normalize to [0, 1]
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        # Extract features and build memory bank
        self._build_memory_bank(X)

        # Train difference module if requested
        if self.train_difference:
            self._train_difference_module(X)

    def _build_memory_bank(self, X: NDArray):
        """Build memory bank of normal features.

        Args:
            X: Training images.
        """
        print("Building memory bank...")
        self.feature_extractor.eval()

        features_dict = {"layer1": [], "layer2": [], "layer3": []}

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i : i + self.batch_size]
                batch_tensor = self._preprocess(batch).to(self.device)

                # Extract multi-scale features
                f1, f2, f3 = self.feature_extractor(batch_tensor)

                features_dict["layer1"].append(f1.cpu())
                features_dict["layer2"].append(f2.cpu())
                features_dict["layer3"].append(f3.cpu())

        # Concatenate features
        self.memory_bank = {}
        for layer in ["layer1", "layer2", "layer3"]:
            features = torch.cat(features_dict[layer], dim=0)
            # Flatten spatial dimensions
            b, c, h, w = features.shape
            features = features.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)
            features = features.reshape(-1, c)  # (B*H*W, C)
            self.memory_bank[layer] = features.numpy()

        # Build k-D tree for efficient k-NN search
        if self.feature_layer == "all":
            # Concatenate all layers
            all_features = np.hstack(
                [self.memory_bank[l] for l in ["layer1", "layer2", "layer3"]]
            )
            self.kd_tree = cKDTree(all_features)
        else:
            self.kd_tree = cKDTree(self.memory_bank[self.feature_layer])

        print(f"Memory bank built with {len(self.memory_bank['layer3'])} features")

    def _train_difference_module(self, X: NDArray):
        """Train the learnable difference module.

        Args:
            X: Training images.
        """
        print("Training difference module...")

        # Create training data: pairs of (image, nn_image)
        X_tensor = self._preprocess(X).to(self.device)

        # Extract all features
        with torch.no_grad():
            all_features = []
            for i in range(0, len(X), self.batch_size):
                batch = X_tensor[i : i + self.batch_size]
                features = self.feature_extractor(batch)
                all_features.append([f.cpu() for f in features])

        # Training loop for each layer
        for layer_name, diff_module in self.diff_modules.items():
            optimizer = torch.optim.Adam(diff_module.parameters(), lr=self.learning_rate)
            diff_module.train()

            for epoch in range(self.epochs):
                epoch_loss = 0.0

                for i in range(len(X)):
                    # Get feature for current image
                    layer_idx = ["layer1", "layer2", "layer3"].index(layer_name)
                    feat_i = all_features[i // self.batch_size][layer_idx][
                        i % self.batch_size
                    ].to(self.device)

                    # Get random neighbor
                    nn_idx = np.random.randint(0, len(X))
                    feat_nn = all_features[nn_idx // self.batch_size][layer_idx][
                        nn_idx % self.batch_size
                    ].to(self.device)

                    # Compute difference (should be small for normal samples)
                    diff_map = diff_module(
                        feat_i.unsqueeze(0), feat_nn.unsqueeze(0)
                    )

                    # Loss: minimize difference for normal samples
                    loss = diff_map.abs().mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % 5 == 0:
                    print(
                        f"  Layer {layer_name} Epoch [{epoch+1}/{self.epochs}] "
                        f"Loss: {epoch_loss/len(X):.6f}"
                    )

            diff_module.eval()

    def predict_proba(self, X: NDArray, **kwargs) -> NDArray:
        """Predict anomaly scores.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            Anomaly scores.
        """
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        self.feature_extractor.eval()
        scores = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i : i + self.batch_size]
                batch_tensor = self._preprocess(batch).to(self.device)

                # Extract features
                f1, f2, f3 = self.feature_extractor(batch_tensor)
                features_dict = {"layer1": f1, "layer2": f2, "layer3": f3}

                # Compute anomaly scores
                for j in range(len(batch)):
                    if self.train_difference:
                        # Use difference module
                        score = self._score_with_difference(
                            {k: v[j : j + 1] for k, v in features_dict.items()}
                        )
                    else:
                        # Use k-NN distance
                        score = self._score_with_knn(
                            {k: v[j : j + 1] for k, v in features_dict.items()}
                        )

                    scores.append(score)

        return np.array(scores)

    def _score_with_knn(self, features: dict) -> float:
        """Score using k-NN distance.

        Args:
            features: Dictionary of features for each layer.

        Returns:
            Anomaly score.
        """
        # Get features for selected layer
        if self.feature_layer == "all":
            feat = torch.cat(
                [features[l] for l in ["layer1", "layer2", "layer3"]], dim=1
            )
        else:
            feat = features[self.feature_layer]

        # Flatten
        b, c, h, w = feat.shape
        feat = feat.view(c, -1).permute(1, 0).cpu().numpy()  # (H*W, C)

        # Query k-NN
        distances, _ = self.kd_tree.query(feat, k=self.k_neighbors)

        # Average distance to k nearest neighbors
        score = distances.mean()

        return score

    def _score_with_difference(self, features: dict) -> float:
        """Score using learned difference module.

        Args:
            features: Dictionary of features for each layer.

        Returns:
            Anomaly score.
        """
        total_diff = 0.0

        for layer_name in ["layer1", "layer2", "layer3"]:
            feat = features[layer_name]  # (1, C, H, W)

            # Find k-nearest neighbors in memory bank
            b, c, h, w = feat.shape
            feat_flat = feat.view(c, -1).permute(1, 0).cpu().numpy()  # (H*W, C)

            distances, indices = self.kd_tree.query(feat_flat, k=1)
            nn_idx = indices[distances.argmin()]

            # Get nearest neighbor feature from memory bank
            nn_feat = (
                torch.from_numpy(self.memory_bank[layer_name][nn_idx])
                .view(1, c, 1, 1)
                .expand(1, c, h, w)
                .to(self.device)
            )

            # Compute difference
            diff_map = self.diff_modules[layer_name](feat, nn_feat)

            # Average difference
            total_diff += diff_map.abs().mean().item()

        return total_diff / 3.0  # Average across layers

    def _preprocess(self, images: NDArray) -> torch.Tensor:
        """Preprocess images.

        Args:
            images: Input images (N, H, W, C).

        Returns:
            Preprocessed tensor (N, C, H, W).
        """
        # Convert to tensor
        if images.ndim == 3:
            images = images[np.newaxis, ...]

        # (N, H, W, C) -> (N, C, H, W)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images
