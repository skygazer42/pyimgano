"""
GLAD - Global and Local Adaptive Diffusion for Anomaly Detection

Reference:
    "GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models"
    ECCV 2024

Uses adaptive diffusion models at both global and local scales for improved
anomaly detection and reconstruction.
"""

from typing import Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from pyimgano.models._imagenet_preprocess import preprocess_imagenet_batch
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._batch_size import call_with_temporary_attr, validate_batch_size
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class SimplifiedDiffusionModel(nn.Module):
    """Simplified diffusion model for anomaly detection."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, num_timesteps: int = 100):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Noise predictor
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),  # +1 for timestep
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise at timestep t.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input (B, C, H, W)
        t : torch.Tensor
            Timestep (B,)

        Returns
        -------
        noise : torch.Tensor
            Predicted noise (B, C, H, W)
        """
        # Embed timestep
        t_emb = (t.float() / self.num_timesteps).view(-1, 1, 1, 1)
        t_emb = t_emb.expand(-1, 1, x.shape[2], x.shape[3])

        # Concatenate timestep embedding
        x_t = torch.cat([x, t_emb], dim=1)

        # Predict noise
        noise = self.noise_predictor(x_t)
        return noise


class GlobalDiffusion(nn.Module):
    """Global-scale diffusion model."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.diffusion = SimplifiedDiffusionModel(in_channels=in_channels, hidden_channels=128)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply global diffusion."""
        return self.diffusion(x, t)


class LocalDiffusion(nn.Module):
    """Local-scale diffusion model with patch-wise processing."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.diffusion = SimplifiedDiffusionModel(in_channels=in_channels, hidden_channels=64)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply local diffusion."""
        return self.diffusion(x, t)


class AdaptiveFusion(nn.Module):
    """Adaptively fuses global and local predictions."""

    def __init__(self, in_channels: int):
        super().__init__()

        self.fusion_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, global_pred: torch.Tensor, local_pred: torch.Tensor) -> torch.Tensor:
        """
        Fuse global and local predictions.

        Parameters
        ----------
        global_pred : torch.Tensor
            Global prediction (B, C, H, W)
        local_pred : torch.Tensor
            Local prediction (B, C, H, W)

        Returns
        -------
        fused : torch.Tensor
            Fused prediction (B, C, H, W)
        """
        # Compute fusion weights
        combined = torch.cat([global_pred, local_pred], dim=1)
        weights = self.fusion_net(combined)  # (B, 2, H, W)

        # Weighted fusion
        w_global = weights[:, 0:1]
        w_local = weights[:, 1:2]
        fused = w_global * global_pred + w_local * local_pred

        return fused


@register_model(
    "vision_glad",
    tags=("vision", "deep", "glad", "diffusion", "adaptive", "eccv2024", "sota"),
    metadata={
        "description": "GLAD - Global-Local Adaptive Diffusion (ECCV 2024)",
        "paper": "GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion",
        "year": 2024,
        "conference": "ECCV",
        "type": "diffusion",
    },
)
class VisionGLAD(BaseVisionDeepDetector):
    """
    GLAD: Global and Local Adaptive Diffusion for Anomaly Detection.

    Uses adaptive diffusion models at both global and local scales,
    with intelligent fusion for improved anomaly detection.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    num_timesteps : int, default=100
        Number of diffusion timesteps
    learning_rate : float, default=1e-4
        Learning rate
    batch_size : int, default=8
        Batch size for training
    epochs : int, default=40
        Number of training epochs
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    feature_extractor_ : nn.Module
        Feature extractor
    global_diffusion_ : GlobalDiffusion
        Global diffusion model
    local_diffusion_ : LocalDiffusion
        Local diffusion model
    fusion_ : AdaptiveFusion
        Adaptive fusion module

    Examples
    --------
    >>> from pyimgano.models import VisionGLAD
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.random((50, 224, 224, 3)).astype(np.float32)
    >>> X_test = rng.random((20, 224, 224, 3)).astype(np.float32)
    >>>
    >>> # Create and train detector
    >>> detector = VisionGLAD(epochs=30, num_timesteps=50)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        num_timesteps: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        epochs: int = 40,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.num_timesteps = num_timesteps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        self.feature_extractor_ = None
        self.global_diffusion_ = None
        self.local_diffusion_ = None
        self.fusion_ = None

    def _preprocess(self, x: NDArray) -> torch.Tensor:
        """Preprocess images."""
        return preprocess_imagenet_batch(x)

    def _build_feature_extractor(self):
        """Build feature extractor."""
        if self.backbone == "wide_resnet50":
            resnet, _ = load_torchvision_model("wide_resnet50", pretrained=True)
        elif self.backbone == "resnet18":
            resnet, _ = load_torchvision_model("resnet18", pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )

        # Freeze
        for param in extractor.parameters():
            param.requires_grad = False
        extractor.eval()

        return extractor

    def _add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise at timestep t."""
        noise = torch.randn_like(x)
        alpha = 1 - t.float() / self.num_timesteps
        alpha = alpha.view(-1, 1, 1, 1)

        noisy_x = alpha.sqrt() * x + (1 - alpha).sqrt() * noise
        return noisy_x, noise

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionGLAD":
        """
        Fit the GLAD detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionGLAD
            Fitted detector
        """
        del y
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        # Preprocess
        x_tensor = self._preprocess(x_array)

        # Initialize feature extractor
        if self.feature_extractor_ is None:
            self.feature_extractor_ = self._build_feature_extractor().to(self.device)

        # Get feature dimensions
        with torch.no_grad():
            sample_features = self.feature_extractor_(x_tensor[:1].to(self.device))
            in_channels = sample_features.shape[1]

        # Initialize diffusion models
        if self.global_diffusion_ is None:
            self.global_diffusion_ = GlobalDiffusion(in_channels).to(self.device)

        if self.local_diffusion_ is None:
            self.local_diffusion_ = LocalDiffusion(in_channels).to(self.device)

        if self.fusion_ is None:
            self.fusion_ = AdaptiveFusion(in_channels).to(self.device)

        # Training
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(
            list(self.global_diffusion_.parameters())
            + list(self.local_diffusion_.parameters())
            + list(self.fusion_.parameters()),
            lr=self.learning_rate,
            weight_decay=0.0,
        )

        self.global_diffusion_.train()
        self.local_diffusion_.train()
        self.fusion_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Extract features
                with torch.no_grad():
                    features = self.feature_extractor_(batch_images)

                # Random timesteps
                t = torch.randint(0, self.num_timesteps, (len(batch_images),), device=self.device)

                # Add noise
                noisy_features, noise = self._add_noise(features, t)

                # Predict noise with both models
                global_pred = self.global_diffusion_(noisy_features, t)
                local_pred = self.local_diffusion_(noisy_features, t)

                # Adaptive fusion
                fused_pred = self.fusion_(global_pred, local_pred)

                # Loss: match true noise
                loss = F.mse_loss(fused_pred, noise)

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
            Anomaly scores (reconstruction error)
        """
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )

        self.global_diffusion_.eval()
        self.local_diffusion_.eval()
        self.fusion_.eval()

        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        x_tensor = self._preprocess(x_array)
        scores = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract features
                features = self.feature_extractor_(batch)

                # Denoise through multiple steps
                reconstruction_errors = []
                for step in range(10, self.num_timesteps, 20):  # Sample steps
                    t = torch.full((len(batch),), step, device=self.device)
                    noisy_features, true_noise = self._add_noise(features, t)

                    # Predict
                    global_pred = self.global_diffusion_(noisy_features, t)
                    local_pred = self.local_diffusion_(noisy_features, t)
                    fused_pred = self.fusion_(global_pred, local_pred)

                    # Reconstruction error
                    error = ((fused_pred - true_noise) ** 2).mean(dim=[1, 2, 3])
                    reconstruction_errors.append(error)

                # Average error across timesteps
                final_error = torch.stack(reconstruction_errors).mean(dim=0)
                scores.append(final_error.cpu().numpy())

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
