"""
BGAD (Background-Guided Anomaly Detection)

Paper: "Background-Guided Anomaly Detection with Feature Normalization"
Conference: CVPR 2023

Key Innovation:
- Uses background features to guide anomaly detection
- Feature normalization with background statistics
- Separates foreground from background for better detection
- Robust to background variations

Implementation includes:
- Background feature extraction and modeling
- Foreground-background separation
- Statistical normalization
- Improved anomaly scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from pyimgano.models.base_dl import BaseVisionDeepDetector


class BackgroundModel:
    """Model for background feature statistics."""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_centers = None
        self.cluster_stds = None

    def fit(self, features: NDArray):
        """Fit background model on normal features.

        Args:
            features: Background features [N, D]
        """
        # Cluster features to find background modes
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = self.kmeans.fit_predict(features)

        # Compute statistics for each cluster
        self.cluster_centers = self.kmeans.cluster_centers_
        self.cluster_stds = np.array([
            features[labels == i].std(axis=0) + 1e-8
            for i in range(self.n_clusters)
        ])

    def normalize(self, features: NDArray) -> NDArray:
        """Normalize features using background statistics.

        Args:
            features: Input features [N, D]

        Returns:
            Normalized features [N, D]
        """
        if self.kmeans is None:
            raise RuntimeError("Background model not fitted")

        # Find nearest cluster for each feature
        labels = self.kmeans.predict(features)

        # Normalize using cluster statistics
        normalized = np.zeros_like(features)
        for i in range(self.n_clusters):
            mask = labels == i
            if mask.any():
                normalized[mask] = (features[mask] - self.cluster_centers[i]) / self.cluster_stds[i]

        return normalized


class FeatureExtractor(nn.Module):
    """Extract features for background-guided detection."""

    def __init__(self, backbone: str = "wide_resnet50", pretrained: bool = True):
        super().__init__()

        if backbone == "wide_resnet50":
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            model = wide_resnet50_2(weights=weights)

            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
            )
            self.output_dim = 512

        elif backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)

            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
            )
            self.output_dim = 128
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Features [B, C, H, W]
        """
        return self.features(x)


class BGADDetector(BaseVisionDeepDetector):
    """BGAD (Background-Guided Anomaly Detection) detector.

    Uses background modeling to normalize features and improve
    anomaly detection, especially robust to background variations.

    Args:
        backbone: Feature extraction backbone
        n_background_clusters: Number of background clusters
        foreground_ratio: Expected ratio of foreground pixels
        gaussian_sigma: Sigma for anomaly map smoothing
        pretrained: Use pre-trained backbone
        device: Device to run on

    Example:
        >>> detector = BGADDetector(
        ...     backbone="wide_resnet50",
        ...     n_background_clusters=5
        ... )
        >>> detector.fit(normal_images)
        >>> scores = detector.predict_proba(test_images)
        >>> anomaly_maps = detector.predict_anomaly_map(test_images)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        n_background_clusters: int = 5,
        foreground_ratio: float = 0.3,
        gaussian_sigma: float = 4.0,
        pretrained: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.backbone_name = backbone
        self.n_background_clusters = n_background_clusters
        self.foreground_ratio = foreground_ratio
        self.gaussian_sigma = gaussian_sigma

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained
        ).to(self.device)

        # Background model
        self.background_model = BackgroundModel(n_clusters=n_background_clusters)

        # Normal feature statistics
        self.normal_features_mean = None
        self.normal_features_std = None

        self.fitted_ = False

    def _extract_features(self, images: NDArray) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Input images [N, H, W, C] or [N, C, H, W]

        Returns:
            Features [N, C, H, W]
        """
        # Convert to tensor
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).float()

        # Ensure [N, C, H, W] format
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)

        images = images.to(self.device)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images / 255.0 - mean) / std

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(images)

        return features

    def _separate_foreground_background(
        self,
        features: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Separate foreground and background features.

        Args:
            features: Flattened features [N, D]

        Returns:
            background_features: Background features
            foreground_features: Foreground features
        """
        # Compute feature norms
        norms = np.linalg.norm(features, axis=1)

        # Background typically has lower feature magnitudes
        threshold = np.percentile(norms, 100 * (1 - self.foreground_ratio))

        background_mask = norms < threshold
        foreground_mask = ~background_mask

        return features[background_mask], features[foreground_mask]

    def fit(self, X: NDArray, y: Optional[NDArray] = None):
        """Fit the detector on normal images.

        Args:
            X: Normal images [N, H, W, C]
            y: Ignored (unsupervised)
        """
        # Extract features
        features = self._extract_features(X)  # [N, C, H, W]

        N, C, H, W = features.shape

        # Flatten spatial dimensions
        features_np = features.cpu().numpy()
        features_flat = features_np.reshape(N, C, -1).transpose(0, 2, 1)  # [N, H*W, C]
        features_flat = features_flat.reshape(-1, C)  # [N*H*W, C]

        # Separate foreground and background
        background_features, _ = self._separate_foreground_background(features_flat)

        # Fit background model on background features
        self.background_model.fit(background_features)

        # Compute normal statistics on all normalized features
        normalized_features = self.background_model.normalize(features_flat)
        self.normal_features_mean = np.mean(normalized_features, axis=0)
        self.normal_features_std = np.std(normalized_features, axis=0) + 1e-8

        self.fitted_ = True
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        """Compute anomaly scores for images.

        Args:
            X: Test images [N, H, W, C]

        Returns:
            Anomaly scores [N]
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract features
        features = self._extract_features(X)  # [N, C, H, W]

        N, C, H, W = features.shape

        # Flatten
        features_np = features.cpu().numpy()
        features_flat = features_np.reshape(N, C, -1).transpose(0, 2, 1)  # [N, H*W, C]

        # Compute anomaly scores for each image
        scores = []
        for i in range(N):
            feat = features_flat[i]  # [H*W, C]

            # Normalize with background model
            normalized = self.background_model.normalize(feat)

            # Z-score with normal statistics
            z_scores = (normalized - self.normal_features_mean) / self.normal_features_std

            # Anomaly score (Mahalanobis-like)
            anomaly_scores = np.linalg.norm(z_scores, axis=1)

            # Image-level score (max)
            image_score = np.max(anomaly_scores)
            scores.append(image_score)

        return np.array(scores)

    def predict_anomaly_map(self, X: NDArray) -> NDArray:
        """Generate pixel-level anomaly maps.

        Args:
            X: Test images [N, H, W, C]

        Returns:
            Anomaly maps [N, H, W]
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get image size
        H_img, W_img = X.shape[1:3] if X.shape[-1] == 3 else X.shape[2:4]

        # Extract features
        features = self._extract_features(X)  # [N, C, H, W]

        N, C, H, W = features.shape

        # Flatten
        features_np = features.cpu().numpy()
        features_flat = features_np.reshape(N, C, -1).transpose(0, 2, 1)  # [N, H*W, C]

        # Compute anomaly maps
        anomaly_maps = []
        for i in range(N):
            feat = features_flat[i]  # [H*W, C]

            # Normalize with background model
            normalized = self.background_model.normalize(feat)

            # Z-score
            z_scores = (normalized - self.normal_features_mean) / self.normal_features_std

            # Anomaly scores
            anomaly_scores = np.linalg.norm(z_scores, axis=1)

            # Reshape to spatial
            anomaly_map = anomaly_scores.reshape(H, W)

            anomaly_maps.append(anomaly_map)

        anomaly_maps = np.array(anomaly_maps)  # [N, H, W]

        # Upsample to image size
        from scipy.ndimage import gaussian_filter, zoom

        upsampled_maps = np.zeros((N, H_img, W_img))
        for i in range(N):
            # Upsample
            zoom_factors = (H_img / H, W_img / W)
            upsampled = zoom(anomaly_maps[i], zoom_factors, order=1)

            # Smooth
            upsampled = gaussian_filter(upsampled, sigma=self.gaussian_sigma)

            upsampled_maps[i] = upsampled

        return upsampled_maps

    def predict(self, X: NDArray, threshold: Optional[float] = None) -> NDArray:
        """Predict anomaly labels.

        Args:
            X: Test images [N, H, W, C]
            threshold: Anomaly threshold (if None, uses 95th percentile)

        Returns:
            Labels [N] (0=normal, 1=anomaly)
        """
        scores = self.predict_proba(X)

        if threshold is None:
            # Auto threshold (could be improved with validation set)
            threshold = np.percentile(scores, 95)

        predictions = (scores > threshold).astype(int)
        return predictions
