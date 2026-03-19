"""
FAVAE - Feature Adaptive Variational Autoencoder

Reference:
    "Feature Adaptive VAE for Anomaly Detection with Dynamic Latent Adjustment"

This method combines pre-trained feature extraction with an adaptive VAE that
dynamically adjusts its latent representation based on input features.
"""

from typing import Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from ._batch_size import call_with_temporary_attr, validate_batch_size
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class FeatureExtractor(nn.Module):
    """Extract features using pre-trained network."""

    def __init__(self, backbone: str = "wide_resnet50"):
        super().__init__()

        if backbone == "wide_resnet50":
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            resnet = wide_resnet50_2(weights=weights)
            self.out_channels = 1024
        elif backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
            self.out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract feature layers up to layer3
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        with torch.no_grad():
            return self.features(x)


class AdaptiveEncoder(nn.Module):
    """Adaptive encoder with dynamic latent space."""

    def __init__(self, in_channels: int, latent_dim: int = 256, adaptive_channels: int = 64):
        super().__init__()

        self.latent_dim = latent_dim

        # Feature adaptation path
        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels, adaptive_channels, kernel_size=1),
            nn.BatchNorm2d(adaptive_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder path
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + adaptive_channels, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Latent projection
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution.

        Returns
        -------
        z : torch.Tensor
            Sampled latent vector
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        """
        # Adaptive features
        adapted = self.adapt_conv(x)

        # Concatenate adaptive features with original
        x_concat = torch.cat([x, adapted], dim=1)

        # Encode
        h = self.encoder(x_concat)
        h = h.view(h.size(0), -1)

        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class AdaptiveDecoder(nn.Module):
    """Decoder with skip connections."""

    def __init__(self, latent_dim: int, out_channels: int, feature_size: int = 28):
        super().__init__()

        self.feature_size = feature_size

        # Project latent to feature space
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * (feature_size // 4) * (feature_size // 4)),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, 2, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, out_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to features."""
        h = self.fc(z)
        h = h.view(h.size(0), 256, self.feature_size // 4, self.feature_size // 4)
        out = self.decoder(h)
        return out


@register_model(
    "vision_favae",
    tags=("vision", "deep", "favae", "vae", "adaptive", "sota"),
    metadata={
        "description": "Feature Adaptive VAE - Dynamic latent space adaptation",
        "paper": "Feature Adaptive VAE for Anomaly Detection",
        "year": 2023,
        "type": "generative",
    },
)
class VisionFAVAE(BaseVisionDeepDetector):
    """
    FAVAE: Feature Adaptive Variational Autoencoder.

    Combines pre-trained features with an adaptive VAE that dynamically
    adjusts its latent representation for improved anomaly detection.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    latent_dim : int, default=256
        Dimension of latent space
    adaptive_channels : int, default=64
        Number of adaptive feature channels
    learning_rate : float, default=1e-4
        Learning rate
    batch_size : int, default=32
        Batch size for training
    epochs : int, default=50
        Number of training epochs
    beta : float, default=1.0
        Weight for KL divergence term
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    feature_extractor_ : FeatureExtractor
        Pre-trained feature extractor
    encoder_ : AdaptiveEncoder
        Adaptive encoder network
    decoder_ : AdaptiveDecoder
        Decoder network

    Examples
    --------
    >>> from pyimgano.models import VisionFAVAE
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.random((100, 224, 224, 3)).astype(np.float32)
    >>> X_test = rng.random((20, 224, 224, 3)).astype(np.float32)
    >>>
    >>> # Create and train detector
    >>> detector = VisionFAVAE(epochs=20)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        latent_dim: int = 256,
        adaptive_channels: int = 64,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 50,
        beta: float = 1.0,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.adaptive_channels = adaptive_channels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        self.feature_extractor_ = None
        self.encoder_ = None
        self.decoder_ = None
        self.feature_size_ = None

    def _preprocess(self, x: NDArray) -> torch.Tensor:
        """Preprocess images."""
        # Convert to CHW format if needed
        if x.shape[-1] == 3:
            x = np.transpose(x, (0, 3, 1, 2))

        # Normalize
        x = x.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        x = (x - mean) / std

        return torch.from_numpy(x).float()

    def _vae_loss(
        self, recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss.

        Returns
        -------
        total_loss : torch.Tensor
            Total VAE loss
        recon_loss : torch.Tensor
            Reconstruction loss
        kl_loss : torch.Tensor
            KL divergence loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, target, reduction="mean")

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionFAVAE":
        """
        Fit the FAVAE detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionFAVAE
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
            sample_features = self.feature_extractor_(x_tensor[:1].to(self.device))
            in_channels = sample_features.shape[1]
            self.feature_size_ = sample_features.shape[2]

        # Initialize encoder and decoder
        if self.encoder_ is None:
            self.encoder_ = AdaptiveEncoder(
                in_channels=in_channels,
                latent_dim=self.latent_dim,
                adaptive_channels=self.adaptive_channels,
            ).to(self.device)

        if self.decoder_ is None:
            self.decoder_ = AdaptiveDecoder(
                latent_dim=self.latent_dim,
                out_channels=in_channels,
                feature_size=self.feature_size_,
            ).to(self.device)

        # Training
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(
            list(self.encoder_.parameters()) + list(self.decoder_.parameters()),
            lr=self.learning_rate,
            weight_decay=0.0,
        )

        self.encoder_.train()
        self.decoder_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Extract features
                with torch.no_grad():
                    features = self.feature_extractor_(batch_images)

                # Encode
                z, mu, logvar = self.encoder_(features)

                # Decode
                recon = self.decoder_(z)

                # Resize reconstruction to match features
                if recon.shape[2:] != features.shape[2:]:
                    recon = F.interpolate(
                        recon, size=features.shape[2:], mode="bilinear", align_corners=False
                    )

                # Compute loss
                loss, recon_loss, kl_loss = self._vae_loss(recon, features, mu, logvar)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                avg_recon = total_recon / len(dataloader)
                avg_kl = total_kl / len(dataloader)
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Recon: {avg_recon:.4f}, "
                    f"KL: {avg_kl:.4f}"
                )

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
            Anomaly scores
        """
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )

        self.encoder_.eval()
        self.decoder_.eval()

        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        x_tensor = self._preprocess(x_array)
        scores = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract features
                features = self.feature_extractor_(batch)

                # Encode and decode
                z, mu, logvar = self.encoder_(features)
                recon = self.decoder_(z)

                # Resize reconstruction
                if recon.shape[2:] != features.shape[2:]:
                    recon = F.interpolate(
                        recon, size=features.shape[2:], mode="bilinear", align_corners=False
                    )

                # Reconstruction error as anomaly score
                recon_error = ((recon - features) ** 2).mean(dim=[1, 2, 3])

                # KL divergence as additional score
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

                # Combined score
                score = recon_error + 0.1 * kl
                scores.append(score.cpu().numpy())

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
        batch_size_int = validate_batch_size(batch_size)
        if batch_size_int is None:
            return self.predict(x_array)
        return call_with_temporary_attr(
            self,
            "batch_size",
            batch_size_int,
            lambda: self.predict(x_array),
        )
