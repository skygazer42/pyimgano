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

from typing import Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from scipy.spatial import cKDTree

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._image_batch import coerce_rgb_image_batch
from ._legacy_x import MISSING, resolve_legacy_x_keyword
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

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()

        # Load pretrained backbone
        if backbone == "resnet18":
            resnet, _ = load_torchvision_model("resnet18", pretrained=bool(pretrained))
            self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # 64 channels
            self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # 128 channels
            self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # 256 channels
        elif backbone == "resnet50":
            resnet, _ = load_torchvision_model("resnet50", pretrained=bool(pretrained))
            self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # 256 channels
            self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # 512 channels
            self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # 1024 channels
        elif backbone == "wide_resnet50":
            resnet, _ = load_torchvision_model("wide_resnet50", pretrained=bool(pretrained))
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
        pretrained: bool = False,
        k_neighbors: int = 5,
        feature_layer: str = "layer3",
        train_difference: bool = True,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
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
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

        # Memory bank
        self.memory_bank = None
        self.kd_tree = None  # backward-compatible alias (selected layer only)
        self.kd_trees = None  # per-layer kNN indices for pixel-map scoring
        self.fitted_ = False

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
                self.diff_modules[layer] = DifferenceModule(feature_dims[layer], out_channels=1).to(
                    self.device
                )

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ):
        """Fit the DifferNet detector.

        Args:
            X: Training images (N, H, W, C) or (N, C, H, W).
            y: Not used (unsupervised).
        """
        legacy_kwargs = {}
        if "X" in kwargs:
            legacy_kwargs["X"] = kwargs.pop("X")
        x_array = coerce_rgb_image_batch(
            resolve_legacy_x_keyword(x, legacy_kwargs, method_name="fit")
        )
        del kwargs
        # Normalize to [0, 1]
        if x_array.max() > 1.0:
            x_array = x_array.astype(np.float32) / 255.0

        # Extract features and build memory bank
        self._build_memory_bank(x_array)

        # Train difference module if requested
        if self.train_difference:
            self._train_difference_module(x_array)

        self.decision_scores_ = np.asarray(self.predict_proba(x_array), dtype=np.float64).reshape(
            -1
        )
        self._process_decision_scores()
        self._set_n_classes(y)
        self.fitted_ = True
        return self

    def _build_memory_bank(self, x: NDArray):
        """Build memory bank of normal features.

        Args:
            X: Training images.
        """
        print("Building memory bank...")
        self.feature_extractor.eval()

        features_dict = {"layer1": [], "layer2": [], "layer3": []}

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch = x[i : i + self.batch_size]
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
            b, c, _, _ = features.shape
            features = features.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)
            features = features.reshape(-1, c)  # (B*H*W, C)
            self.memory_bank[layer] = features.numpy()

        # Build k-D trees for efficient k-NN search (per-layer).
        self.kd_trees = {
            layer: cKDTree(self.memory_bank[layer]) for layer in ["layer1", "layer2", "layer3"]
        }
        if self.feature_layer != "all":
            if self.feature_layer not in self.kd_trees:
                raise ValueError(f"Unknown feature_layer: {self.feature_layer!r}")
            # Backward-compatible alias for code paths that use only one layer.
            self.kd_tree = self.kd_trees[self.feature_layer]
        else:
            self.kd_tree = None

        print(f"Memory bank built with {len(self.memory_bank['layer3'])} features")

    def _train_difference_module(self, x: NDArray):
        """Train the learnable difference module.

        Args:
            X: Training images.
        """
        print("Training difference module...")

        # Create training data: pairs of (image, nn_image)
        x_tensor = self._preprocess(x).to(self.device)

        # Extract all features
        with torch.no_grad():
            all_features = []
            for i in range(0, len(x), self.batch_size):
                batch = x_tensor[i : i + self.batch_size]
                features = self.feature_extractor(batch)
                all_features.append([f.cpu() for f in features])

        # Training loop for each layer
        for layer_name, diff_module in self.diff_modules.items():
            optimizer = torch.optim.Adam(
                diff_module.parameters(), lr=self.learning_rate, weight_decay=0.0
            )
            diff_module.train()

            for epoch in range(self.epochs):
                epoch_loss = 0.0

                for i in range(len(x)):
                    # Get feature for current image
                    layer_idx = ["layer1", "layer2", "layer3"].index(layer_name)
                    feat_i = all_features[i // self.batch_size][layer_idx][i % self.batch_size].to(
                        self.device
                    )

                    # Get random neighbor
                    nn_idx = int(self.rng.integers(0, len(x)))
                    feat_nn = all_features[nn_idx // self.batch_size][layer_idx][
                        nn_idx % self.batch_size
                    ].to(self.device)

                    # Compute difference (should be small for normal samples)
                    diff_map = diff_module(feat_i.unsqueeze(0), feat_nn.unsqueeze(0))

                    # Loss: minimize difference for normal samples
                    loss = diff_map.abs().mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()

                if (epoch + 1) % 5 == 0:
                    print(
                        f"  Layer {layer_name} Epoch [{epoch+1}/{self.epochs}] "
                        f"Loss: {epoch_loss/len(x):.6f}"
                    )

            diff_module.eval()

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
            for i in range(0, len(x_array), self.batch_size):
                batch = x_array[i : i + self.batch_size]
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
        if self.kd_trees is None:
            raise RuntimeError("Memory bank not built. Call fit() first.")

        layers = (
            ["layer1", "layer2", "layer3"]
            if self.feature_layer == "all"
            else [str(self.feature_layer)]
        )

        scores: list[float] = []
        for layer in layers:
            if layer not in features:
                raise KeyError(f"Missing features for layer {layer!r}")
            if layer not in self.kd_trees:
                raise KeyError(f"Missing kNN index for layer {layer!r}")

            feat = features[layer]
            _b, c, _, _ = feat.shape
            feat_flat = feat.view(c, -1).permute(1, 0).cpu().numpy()  # (H*W, C)

            distances, _ = self.kd_trees[layer].query(feat_flat, k=int(self.k_neighbors))
            d = np.asarray(distances, dtype=np.float64)
            if d.ndim == 1:
                d = d.reshape(-1, 1)
            scores.append(float(np.mean(d)))

        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _score_with_difference(self, features: dict) -> float:
        """Score using learned difference module.

        Args:
            features: Dictionary of features for each layer.

        Returns:
            Anomaly score.
        """
        if self.kd_trees is None or self.memory_bank is None:
            raise RuntimeError("Memory bank not built. Call fit() first.")

        total_diff = 0.0
        for layer_name in ["layer1", "layer2", "layer3"]:
            if layer_name not in self.kd_trees:
                raise KeyError(f"Missing kNN index for layer {layer_name!r}")
            if layer_name not in self.memory_bank:
                raise KeyError(f"Missing memory bank for layer {layer_name!r}")

            feat = features[layer_name]  # (1, C, H, W)

            # Find one nearest neighbor patch (best-effort) from the memory bank.
            _b, c, h, w = feat.shape
            feat_flat = feat.view(c, -1).permute(1, 0).cpu().numpy()  # (H*W, C)
            distances, indices = self.kd_trees[layer_name].query(feat_flat, k=1)
            d = np.asarray(distances, dtype=np.float64).reshape(-1)
            idx = np.asarray(indices, dtype=np.int64).reshape(-1)
            nn_idx = int(idx[int(np.argmin(d))])

            nn_vec = np.asarray(self.memory_bank[layer_name][nn_idx], dtype=np.float32).reshape(-1)
            if nn_vec.shape[0] != int(c):
                raise RuntimeError(
                    f"Memory bank dim mismatch for layer {layer_name}: expected {c}, got {nn_vec.shape[0]}"
                )

            nn_feat = torch.from_numpy(nn_vec).view(1, c, 1, 1).expand(1, c, h, w).to(self.device)

            diff_map = self.diff_modules[layer_name](feat, nn_feat)
            total_diff += diff_map.abs().mean().item()

        return total_diff / 3.0

    # ------------------------------------------------------------------
    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        """Alias for scoring (BaseDetector semantics: higher => more anomalous)."""
        # DiffNet scores each input independently. Keep `batch_size` for
        # interface compatibility with BaseDeepLearningDetector.
        resolved_x = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        if isinstance(resolved_x, (list, tuple)) and len(resolved_x) == 0:
            if batch_size is not None:
                batch_size_int = int(batch_size)
                if batch_size_int <= 0:
                    raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
            return np.asarray([], dtype=np.float64)
        x_array = coerce_rgb_image_batch(resolved_x)
        if batch_size is not None:
            batch_size_int = int(batch_size)
            if batch_size_int <= 0:
                raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")

        return np.asarray(self.predict_proba(x_array), dtype=np.float64).reshape(-1)

    def _infer_image_hw(self, image: np.ndarray) -> tuple[int, int]:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")
        if arr.shape[-1] in (1, 3):
            return int(arr.shape[0]), int(arr.shape[1])
        if arr.shape[0] in (1, 3):
            return int(arr.shape[1]), int(arr.shape[2])
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    def get_anomaly_map(self, image: np.ndarray) -> NDArray:
        """Return a best-effort pixel anomaly map (H,W) for one image."""

        if self.kd_trees is None or self.memory_bank is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        img = np.asarray(image)
        h_img, w_img = self._infer_image_hw(img)

        x = self._preprocess(img).to(self.device)
        self.feature_extractor.eval()

        maps: list[np.ndarray] = []
        with torch.no_grad():
            f1, f2, f3 = self.feature_extractor(x)
            feats = {"layer1": f1, "layer2": f2, "layer3": f3}

            if self.train_difference:
                for layer_name in ["layer1", "layer2", "layer3"]:
                    feat = feats[layer_name]
                    _b, c, h, w = feat.shape

                    feat_flat = feat.view(c, -1).permute(1, 0).cpu().numpy()  # (H*W,C)
                    distances, indices = self.kd_trees[layer_name].query(feat_flat, k=1)
                    d = np.asarray(distances, dtype=np.float64).reshape(-1)
                    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
                    nn_idx = int(idx[int(np.argmin(d))])

                    nn_vec = np.asarray(
                        self.memory_bank[layer_name][nn_idx], dtype=np.float32
                    ).reshape(-1)
                    nn_feat = (
                        torch.from_numpy(nn_vec).view(1, c, 1, 1).expand(1, c, h, w).to(self.device)
                    )

                    diff_map = self.diff_modules[layer_name](feat, nn_feat).abs()  # (1,1,h,w)
                    up = F.interpolate(
                        diff_map, size=(h_img, w_img), mode="bilinear", align_corners=False
                    )
                    maps.append(up[0, 0].detach().cpu().numpy())
            else:
                layers = (
                    ["layer1", "layer2", "layer3"]
                    if self.feature_layer == "all"
                    else [str(self.feature_layer)]
                )

                for layer_name in layers:
                    feat = feats[layer_name]
                    _b, c, h, w = feat.shape
                    feat_flat = feat.view(c, -1).permute(1, 0).cpu().numpy()  # (H*W,C)

                    distances, _ = self.kd_trees[layer_name].query(
                        feat_flat, k=int(self.k_neighbors)
                    )
                    d = np.asarray(distances, dtype=np.float32)
                    if d.ndim == 1:
                        patch_scores = d
                    else:
                        patch_scores = d.mean(axis=1)

                    patch_map = patch_scores.reshape(int(h), int(w)).astype(np.float32, copy=False)
                    patch_map_t = (
                        torch.from_numpy(patch_map).view(1, 1, int(h), int(w)).to(self.device)
                    )
                    up = F.interpolate(
                        patch_map_t, size=(h_img, w_img), mode="bilinear", align_corners=False
                    )
                    maps.append(up[0, 0].detach().cpu().numpy())

        if not maps:
            return np.zeros((h_img, w_img), dtype=np.float32)
        return np.asarray(np.stack(maps, axis=0).mean(axis=0), dtype=np.float32)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        """Return anomaly maps (N,H,W) for a batch of images."""

        arr = coerce_rgb_image_batch(
            resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        )
        if arr.ndim == 3:
            m = self.get_anomaly_map(arr)
            return np.asarray(m[None, ...], dtype=np.float32)

        if arr.ndim != 4:
            raise ValueError(f"Expected X with shape (N,H,W,C) or (N,C,H,W), got {arr.shape}")

        maps = [self.get_anomaly_map(arr[i]) for i in range(int(arr.shape[0]))]
        return np.asarray(np.stack(maps, axis=0), dtype=np.float32)

    def _preprocess(self, images: NDArray) -> torch.Tensor:
        """Preprocess images.

        Args:
            images: Input images (N, H, W, C).

        Returns:
            Preprocessed tensor (N, C, H, W).
        """
        arr = np.asarray(images)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 4:
            raise ValueError(f"Expected images with shape (N,H,W,C) or (N,C,H,W), got {arr.shape}")

        arr_f = arr.astype(np.float32, copy=False)
        if float(np.max(arr_f)) > 1.5:
            arr_f = arr_f / 255.0

        # (N, H, W, C) -> (N, C, H, W) OR accept (N, C, H, W)
        if arr_f.shape[-1] in (1, 3):
            t = torch.from_numpy(arr_f).permute(0, 3, 1, 2)
        elif arr_f.shape[1] in (1, 3):
            t = torch.from_numpy(arr_f)
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if int(t.shape[1]) == 1:
            t = t.repeat(1, 3, 1, 1)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
        return (t - mean) / std
