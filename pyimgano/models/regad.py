"""
RegAD - Registration-based Anomaly Detection

Reference:
    "Registration-based Anomaly Detection for Industrial Quality Inspection"

Uses feature registration and alignment to detect anomalies by measuring
misalignment between test and reference features.
"""

from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from pyimgano.models._imagenet_preprocess import preprocess_imagenet_batch
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class SpatialTransformerNetwork(nn.Module):
    """Spatial Transformer Network for feature registration."""

    def __init__(self, feature_channels: int):
        super().__init__()

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(feature_channels, 64, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Regression for affine transformation parameters
        self.fc_loc = nn.Sequential(nn.Linear(128 * 4 * 4, 256), nn.ReLU(True), nn.Linear(256, 6))

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input features (B, C, H, W)

        Returns
        -------
        transformed : torch.Tensor
            Registered features (B, C, H, W)
        """
        # Localization network
        xs = self.localization(x)
        xs = xs.reshape(-1, 128 * 4 * 4)

        # Compute transformation parameters
        theta = self.fc_loc(xs)
        theta = theta.reshape(-1, 2, 3)

        # Generate sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # Sample input with grid
        x = F.grid_sample(x, grid, align_corners=False)

        return x


class RegistrationNetwork(nn.Module):
    """Network for feature registration and alignment."""

    def __init__(self, backbone: str = "wide_resnet50"):
        super().__init__()

        # Feature extractor
        if backbone == "wide_resnet50":
            resnet, _ = load_torchvision_model("wide_resnet50", pretrained=True)
            self.feature_channels = 1024
        elif backbone == "resnet18":
            resnet, _ = load_torchvision_model("resnet18", pretrained=True)
            self.feature_channels = 256
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        # Spatial transformer for registration
        self.stn = SpatialTransformerNetwork(self.feature_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and register features.

        Returns
        -------
        features : torch.Tensor
            Original features
        registered : torch.Tensor
            Registered features
        """
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(x)

        # Apply registration
        registered = self.stn(features)

        return features, registered


@register_model(
    "vision_regad",
    tags=("vision", "deep", "regad", "registration", "alignment", "sota"),
    metadata={
        "description": "RegAD - Registration-based anomaly detection with STN",
        "paper": "Registration-based Anomaly Detection",
        "year": 2023,
        "type": "registration",
    },
)
class VisionRegAD(BaseVisionDeepDetector):
    """
    RegAD: Registration-based Anomaly Detection.

    Detects anomalies by measuring feature misalignment after attempting
    to register test features with a reference template.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    learning_rate : float, default=1e-4
        Learning rate for training
    batch_size : int, default=16
        Batch size for training
    epochs : int, default=40
        Number of training epochs
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    reg_network_ : RegistrationNetwork
        Registration network
    reference_features_ : torch.Tensor
        Reference feature template

    Examples
    --------
    >>> from pyimgano.models import VisionRegAD
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.random((100, 224, 224, 3)).astype(np.float32)
    >>> X_test = rng.random((20, 224, 224, 3)).astype(np.float32)
    >>>
    >>> # Create and train detector
    >>> detector = VisionRegAD(epochs=20)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 40,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        self.reg_network_ = None
        self.reference_features_ = None

    def _preprocess(self, x: NDArray) -> torch.Tensor:
        """Preprocess images."""
        return preprocess_imagenet_batch(x)

    def _registration_loss(self, registered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute registration loss.

        Parameters
        ----------
        registered : torch.Tensor
            Registered features
        target : torch.Tensor
            Target features

        Returns
        -------
        loss : torch.Tensor
            Registration loss
        """
        # MSE loss for alignment
        mse_loss = F.mse_loss(registered, target)

        # Cosine similarity loss
        registered_flat = registered.reshape(registered.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)

        cos_sim = F.cosine_similarity(registered_flat, target_flat, dim=1)
        cos_loss = (1 - cos_sim).mean()

        # Combined loss
        total_loss = mse_loss + 0.1 * cos_loss

        return total_loss

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionRegAD":
        """
        Fit the RegAD detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionRegAD
            Fitted detector
        """
        del y
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        # Preprocess
        x_tensor = self._preprocess(x_array)

        # Initialize network
        if self.reg_network_ is None:
            self.reg_network_ = RegistrationNetwork(backbone=self.backbone).to(self.device)

        # Build reference template from training data
        with torch.no_grad():
            all_features = []
            for i in range(0, min(len(x_tensor), 100), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)
                features, _ = self.reg_network_(batch)
                all_features.append(features)

            # Average features as reference
            self.reference_features_ = torch.cat(all_features, dim=0).mean(dim=0, keepdim=True)

        # Training - learn to register to reference
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Only optimize STN parameters
        optimizer = torch.optim.Adam(
            self.reg_network_.stn.parameters(), lr=self.learning_rate, weight_decay=0.0
        )

        self.reg_network_.stn.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Forward pass
                _, registered = self.reg_network_(batch_images)

                # Expand reference to batch size
                batch_ref = self.reference_features_.expand(len(batch_images), -1, -1, -1)

                # Compute registration loss
                loss = self._registration_loss(registered, batch_ref)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        return self

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
            Anomaly scores (registration error)
        """
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="predict"))

        self.reg_network_.eval()

        x_tensor = self._preprocess(x_array)
        scores = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract and register features
                _, registered = self.reg_network_(batch)

                # Expand reference
                batch_ref = self.reference_features_.expand(len(batch), -1, -1, -1)

                # Compute registration error
                error = ((registered - batch_ref) ** 2).mean(dim=[1, 2, 3])
                scores.append(error.cpu().numpy())

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

    def get_registration_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        """
        Get pixel-level registration error maps.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        error_maps : NDArray of shape (n_samples, height, width)
            Registration error maps
        """
        self.reg_network_.eval()

        x_array = cast(
            NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="get_registration_map")
        )
        x_tensor = self._preprocess(x_array)
        h, w = x_array.shape[1:3]
        error_maps = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract and register
                _, registered = self.reg_network_(batch)

                # Expand reference
                batch_ref = self.reference_features_.expand(len(batch), -1, -1, -1)

                # Compute error map
                error_map = ((registered - batch_ref) ** 2).mean(dim=1)

                # Resize to original image size
                error_map = F.interpolate(
                    error_map.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
                ).squeeze(1)

                error_maps.append(error_map.cpu().numpy())

        return np.concatenate(error_maps)
