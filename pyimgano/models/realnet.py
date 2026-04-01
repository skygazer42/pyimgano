"""
RealNet - Feature Selection Network with Realistic Synthetic Anomaly

Reference:
    "RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection"
    CVPR 2024

Uses realistic synthetic anomalies and feature selection mechanisms to improve
anomaly detection performance and generalization.
"""

from typing import Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, TensorDataset

from pyimgano.models._imagenet_preprocess import preprocess_imagenet_batch
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class FeatureExtractor(nn.Module):
    """Multi-scale feature extractor."""

    def __init__(self, backbone: str = "wide_resnet50"):
        super().__init__()

        if backbone == "wide_resnet50":
            resnet, _ = load_torchvision_model("wide_resnet50", pretrained=True)
        elif backbone == "resnet18":
            resnet, _ = load_torchvision_model("resnet18", pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Freeze
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale features."""
        with torch.no_grad():
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
        return f1, f2, f3


class FeatureSelector(nn.Module):
    """Feature selection module with channel attention."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature selection."""
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa

        return x


class AnomalyGenerator:
    """Generates realistic synthetic anomalies."""

    def __init__(
        self,
        anomaly_types: Optional[list] = None,
        random_state: Optional[int] = None,
    ):
        if anomaly_types is None:
            self.anomaly_types = ["perlin_noise", "cutpaste", "texture", "intensity", "blur"]
        else:
            self.anomaly_types = anomaly_types
        self.rng = np.random.default_rng(random_state)

    def generate_anomaly(
        self, image: torch.Tensor, anomaly_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic anomaly.

        Parameters
        ----------
        image : torch.Tensor
            Input image (C, H, W)
        anomaly_type : str, optional
            Type of anomaly to generate

        Returns
        -------
        anomalous_image : torch.Tensor
            Image with synthetic anomaly
        mask : torch.Tensor
            Binary mask of anomaly region
        """
        rng = self.rng
        if anomaly_type is None:
            anomaly_type = rng.choice(self.anomaly_types)

        c, h, w = image.shape
        anomalous_image = image.clone()
        mask = torch.zeros(1, h, w, device=image.device)

        # Random anomaly region
        h_size = int(rng.integers(h // 8, h // 3))
        w_size = int(rng.integers(w // 8, w // 3))
        y = int(rng.integers(0, h - h_size))
        x = int(rng.integers(0, w - w_size))

        if anomaly_type == "perlin_noise":
            # Perlin-like noise using multiple octaves
            noise = self._generate_perlin_noise((h_size, w_size))
            noise_tensor = torch.from_numpy(noise).float().to(image.device)
            noise_tensor = noise_tensor.unsqueeze(0).expand(c, -1, -1)
            anomalous_image[:, y : y + h_size, x : x + w_size] += noise_tensor * 0.5

        elif anomaly_type == "cutpaste":
            # Cut and paste from different location
            src_y = int(rng.integers(0, h - h_size))
            src_x = int(rng.integers(0, w - w_size))
            patch = image[:, src_y : src_y + h_size, src_x : src_x + w_size].clone()
            # Apply random rotation
            angle = float(rng.uniform(-30, 30))
            patch = self._rotate_patch(patch, angle)
            anomalous_image[:, y : y + h_size, x : x + w_size] = patch

        elif anomaly_type == "texture":
            # Random texture from uniform distribution
            texture = torch.rand(c, h_size, w_size, device=image.device) * 2 - 1
            anomalous_image[:, y : y + h_size, x : x + w_size] += texture * 0.3

        elif anomaly_type == "intensity":
            # Intensity change
            factor = float(rng.uniform(0.3, 2.0))
            anomalous_image[:, y : y + h_size, x : x + w_size] *= factor

        elif anomaly_type == "blur":
            # Gaussian blur
            region = anomalous_image[:, y : y + h_size, x : x + w_size].cpu().numpy()
            for c in range(c):
                region[c] = gaussian_filter(region[c], sigma=3)
            anomalous_image[:, y : y + h_size, x : x + w_size] = torch.from_numpy(region).to(
                image.device
            )

        mask[0, y : y + h_size, x : x + w_size] = 1.0

        return anomalous_image, mask

    def _generate_perlin_noise(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate Perlin-like noise."""
        noise = self.rng.standard_normal(shape)
        for sigma in [1, 2, 4]:
            noise += gaussian_filter(self.rng.standard_normal(shape), sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise * 2 - 1

    def _rotate_patch(self, patch: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate patch by angle."""
        # Simple rotation using affine grid
        theta = torch.tensor(
            [
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            ],
            dtype=torch.float32,
            device=patch.device,
        )

        grid = F.affine_grid(theta.unsqueeze(0), patch.unsqueeze(0).size(), align_corners=False)
        rotated = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False).squeeze(0)
        return rotated


@register_model(
    "vision_realnet",
    tags=("vision", "deep", "realnet", "feature-selection", "cvpr2024", "sota"),
    metadata={
        "description": "RealNet - Feature Selection with Realistic Synthetic Anomaly (CVPR 2024)",
        "paper": "RealNet: A Feature Selection Network with Realistic Synthetic Anomaly",
        "year": 2024,
        "conference": "CVPR",
        "type": "feature-selection",
    },
)
class VisionRealNet(BaseVisionDeepDetector):
    """
    RealNet: Feature Selection Network with Realistic Synthetic Anomaly.

    Uses realistic synthetic anomalies and learnable feature selection to
    improve anomaly detection performance.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    reduction : int, default=16
        Reduction ratio for channel attention
    learning_rate : float, default=1e-4
        Learning rate
    batch_size : int, default=16
        Batch size for training
    epochs : int, default=40
        Number of training epochs
    anomaly_ratio : float, default=0.5
        Ratio of synthetic anomalies in training
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    feature_extractor_ : FeatureExtractor
        Multi-scale feature extractor
    feature_selectors_ : nn.ModuleList
        Feature selection modules for each scale
    anomaly_generator_ : AnomalyGenerator
        Synthetic anomaly generator

    Examples
    --------
    >>> from pyimgano.models import VisionRealNet
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.random((100, 224, 224, 3), dtype=np.float32)
    >>> X_test = rng.random((20, 224, 224, 3), dtype=np.float32)
    >>>
    >>> # Create and train detector
    >>> detector = VisionRealNet(epochs=30)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        reduction: int = 16,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 40,
        anomaly_ratio: float = 0.5,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.reduction = reduction
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.anomaly_ratio = anomaly_ratio
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        self.feature_extractor_ = None
        self.feature_selectors_ = None
        self.anomaly_generator_ = AnomalyGenerator(random_state=random_state)
        self.normal_features_ = []

    def _preprocess(self, x: NDArray) -> torch.Tensor:
        """Preprocess images."""
        return preprocess_imagenet_batch(x)

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionRealNet":
        """
        Fit the RealNet detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionRealNet
            Fitted detector
        """
        del y
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        # Preprocess
        x_tensor = self._preprocess(x_array)

        # Initialize feature extractor
        if self.feature_extractor_ is None:
            self.feature_extractor_ = FeatureExtractor(backbone=self.backbone).to(self.device)

        # Get feature dimensions
        with torch.no_grad():
            f1, f2, f3 = self.feature_extractor_(x_tensor[:1].to(self.device))
            channels = [f1.shape[1], f2.shape[1], f3.shape[1]]

        # Initialize feature selectors
        if self.feature_selectors_ is None:
            self.feature_selectors_ = nn.ModuleList(
                [FeatureSelector(ch, self.reduction) for ch in channels]
            ).to(self.device)

        # Training
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(
            self.feature_selectors_.parameters(), lr=self.learning_rate, weight_decay=0.0
        )
        criterion = nn.BCEWithLogitsLoss()

        self.feature_selectors_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)
                b = len(batch_images)

                # Create mixed batch with synthetic anomalies
                n_anomaly = int(b * self.anomaly_ratio)
                labels = torch.zeros(b, device=self.device)

                if n_anomaly > 0:
                    for i in range(n_anomaly):
                        batch_images[i], _ = self.anomaly_generator_.generate_anomaly(
                            batch_images[i]
                        )
                        labels[i] = 1.0

                # Extract features
                f1, f2, f3 = self.feature_extractor_(batch_images)

                # Apply feature selection
                s1 = self.feature_selectors_[0](f1)
                s2 = self.feature_selectors_[1](f2)
                s3 = self.feature_selectors_[2](f3)

                # Compute anomaly scores
                score1 = s1.mean(dim=[1, 2, 3])
                score2 = s2.mean(dim=[1, 2, 3])
                score3 = s3.mean(dim=[1, 2, 3])
                scores = (score1 + score2 + score3) / 3

                # Loss
                loss = criterion(scores, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # Build normal feature memory
        self._build_normal_memory(x_tensor)

        return self

    def _build_normal_memory(self, x_tensor: torch.Tensor):
        """Build memory of normal features."""
        self.feature_selectors_.eval()
        self.normal_features_ = [[], [], []]

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                f1, f2, f3 = self.feature_extractor_(batch)
                s1 = self.feature_selectors_[0](f1)
                s2 = self.feature_selectors_[1](f2)
                s3 = self.feature_selectors_[2](f3)

                self.normal_features_[0].append(s1.cpu())
                self.normal_features_[1].append(s2.cpu())
                self.normal_features_[2].append(s3.cpu())

        for i in range(3):
            self.normal_features_[i] = torch.cat(self.normal_features_[i], dim=0)

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : NDArray of shape (n_samples,)
            Anomaly scores
        """
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="predict"))

        self.feature_selectors_.eval()

        x_tensor = self._preprocess(x_array)
        scores = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract and select features
                f1, f2, f3 = self.feature_extractor_(batch)
                s1 = self.feature_selectors_[0](f1)
                s2 = self.feature_selectors_[1](f2)
                s3 = self.feature_selectors_[2](f3)

                # Compute distances to normal features
                batch_scores = []
                for s, normal_f in zip([s1, s2, s3], self.normal_features_):
                    # Average distance to normal samples
                    s_flat = s.reshape(s.size(0), -1).cpu()
                    normal_flat = normal_f.reshape(normal_f.size(0), -1)

                    dists = torch.cdist(s_flat, normal_flat[:100])  # Use subset for efficiency
                    min_dists = dists.min(dim=1)[0]
                    batch_scores.append(min_dists)

                # Combine scores
                final_scores = torch.stack(batch_scores).mean(dim=0)
                scores.append(final_scores.numpy())

        return np.concatenate(scores)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        """Alias for predict."""
        x_array = cast(
            NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        )
        if batch_size is None:
            return self.predict(x_array)

        batch_size_int = int(batch_size)
        if batch_size_int <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")

        old_batch_size = self.batch_size
        try:
            self.batch_size = batch_size_int
            return self.predict(x_array)
        finally:
            self.batch_size = old_batch_size
