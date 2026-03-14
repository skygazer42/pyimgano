"""
One-for-More - Continual Diffusion Model for Anomaly Detection

Reference:
    "One-for-More: Continual Diffusion Model for Anomaly Detection"
    CVPR 2025

Achieves first place in 17/18 settings on MVTec and VisA datasets through
continual learning with gradient projection for stable anomaly detection.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from ._batch_size import call_with_temporary_attr, validate_batch_size
from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class ContinualDiffusion(nn.Module):
    """Continual diffusion model with gradient projection."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # U-Net style denoising network
        self.encoder = nn.ModuleList(
            [
                self._make_block(in_channels, hidden_channels),
                self._make_block(hidden_channels, hidden_channels * 2),
                self._make_block(hidden_channels * 2, hidden_channels * 4),
            ]
        )

        self.bottleneck = self._make_block(hidden_channels * 4, hidden_channels * 4)

        self.decoder = nn.ModuleList(
            [
                self._make_block(hidden_channels * 8, hidden_channels * 2),
                self._make_block(hidden_channels * 4, hidden_channels),
                self._make_block(hidden_channels * 2, in_channels),
            ]
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
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
        # Time embedding
        t_emb = self.time_mlp(t.float().view(-1, 1) / self.num_timesteps)
        t_emb = t_emb.view(-1, t_emb.size(1), 1, 1)

        # Encoder
        enc_features = []
        h = x
        for encoder in self.encoder:
            h = encoder(h)
            enc_features.append(h)
            h = F.max_pool2d(h, 2)

        # Bottleneck
        h = self.bottleneck(h) + t_emb

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoder):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = torch.cat([h, enc_features[-(i + 1)]], dim=1)
            h = decoder(h)

        return h


class GradientProjection:
    """Gradient projection for continual learning."""

    def __init__(self, device: str):
        self.device = device
        self.task_gradients = []

    def store_gradients(self, model: nn.Module):
        """Store gradients from current task."""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone().flatten())
        if grads:
            self.task_gradients.append(torch.cat(grads))

    def project_gradients(self, model: nn.Module):
        """Project current gradients to avoid catastrophic forgetting."""
        if not self.task_gradients:
            return

        # Get current gradients
        current_grads = []
        for param in model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.flatten())

        if not current_grads:
            return

        current_grad = torch.cat(current_grads)

        # Project away from previous task gradients
        for prev_grad in self.task_gradients:
            similarity = torch.dot(current_grad, prev_grad)
            if similarity < 0:  # Only project if gradients conflict
                current_grad = current_grad - similarity * prev_grad / (
                    prev_grad.norm() ** 2 + 1e-8
                )

        # Update model gradients
        idx = 0
        for param in model.parameters():
            if param.grad is not None:
                param_size = param.grad.numel()
                param.grad.copy_(current_grad[idx : idx + param_size].view_as(param.grad))
                idx += param_size


@register_model(
    "vision_oneformore",
    tags=("vision", "deep", "oneformore", "continual", "diffusion", "cvpr2025", "sota"),
    metadata={
        "description": "One-for-More - Continual Diffusion Model (CVPR 2025, #1 on MVTec/VisA)",
        "paper": "One-for-More: Continual Diffusion Model for Anomaly Detection",
        "year": 2025,
        "conference": "CVPR",
        "rank": "1st place on 17/18 MVTec & VisA settings",
        "type": "continual-diffusion",
    },
)
class VisionOneForMore(BaseVisionDeepDetector):
    """
    One-for-More: Continual Diffusion Model for Anomaly Detection.

    Achieves first place in 17/18 settings on MVTec and VisA datasets
    through continual learning with gradient projection.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    hidden_channels : int, default=128
        Hidden channels in diffusion model
    num_timesteps : int, default=1000
        Number of diffusion timesteps
    learning_rate : float, default=1e-4
        Learning rate
    batch_size : int, default=8
        Batch size for training
    epochs : int, default=50
        Number of training epochs
    use_gradient_projection : bool, default=True
        Whether to use gradient projection for continual learning
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    feature_extractor_ : nn.Module
        Feature extraction network
    diffusion_model_ : ContinualDiffusion
        Continual diffusion model
    gradient_projection_ : GradientProjection
        Gradient projection module

    Examples
    --------
    >>> from pyimgano.models import VisionOneForMore
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    >>> X_test = np.random.rand(20, 224, 224, 3).astype(np.float32)
    >>>
    >>> # Create and train detector (CVPR 2025 #1 method!)
    >>> detector = VisionOneForMore(epochs=30)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        hidden_channels: int = 128,
        num_timesteps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        epochs: int = 50,
        use_gradient_projection: bool = True,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.hidden_channels = hidden_channels
        self.num_timesteps = num_timesteps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_gradient_projection = use_gradient_projection
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.feature_extractor_ = None
        self.diffusion_model_ = None
        self.gradient_projection_ = None

    def _preprocess(self, X: NDArray) -> torch.Tensor:
        """Preprocess images."""
        if X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))

        X = X.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        X = (X - mean) / std

        return torch.from_numpy(X).float()

    def _build_feature_extractor(self):
        """Build feature extractor."""
        if self.backbone == "wide_resnet50":
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            resnet = wide_resnet50_2(weights=weights)
        elif self.backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
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
        """Add noise to clean images."""
        noise = torch.randn_like(x)

        alphas_cumprod = self.diffusion_model_.alphas_cumprod.to(x.device)
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)

        noisy_x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise
        return noisy_x, noise

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "VisionOneForMore":
        """
        Fit the One-for-More detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionOneForMore
            Fitted detector
        """
        # Preprocess
        X_tensor = self._preprocess(X)

        # Initialize feature extractor
        if self.feature_extractor_ is None:
            self.feature_extractor_ = self._build_feature_extractor().to(self.device)

        # Get feature dimensions
        with torch.no_grad():
            sample_features = self.feature_extractor_(X_tensor[:1].to(self.device))
            in_channels = sample_features.shape[1]

        # Initialize diffusion model
        if self.diffusion_model_ is None:
            self.diffusion_model_ = ContinualDiffusion(
                in_channels=in_channels,
                hidden_channels=self.hidden_channels,
                num_timesteps=self.num_timesteps,
            ).to(self.device)

        # Initialize gradient projection
        if self.use_gradient_projection and self.gradient_projection_ is None:
            self.gradient_projection_ = GradientProjection(self.device)

        # Training
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.diffusion_model_.parameters(), lr=self.learning_rate)

        self.diffusion_model_.train()

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

                # Predict noise
                pred_noise = self.diffusion_model_(noisy_features, t)

                # Loss
                loss = F.mse_loss(pred_noise, noise)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient projection for continual learning
                if self.use_gradient_projection and self.gradient_projection_ is not None:
                    self.gradient_projection_.project_gradients(self.diffusion_model_)

                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # Store gradients for future continual learning
        if self.use_gradient_projection:
            self.gradient_projection_.store_gradients(self.diffusion_model_)

        return self

    def predict(self, X: NDArray, return_confidence: bool = False) -> NDArray:
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

        self.diffusion_model_.eval()

        X_tensor = self._preprocess(X)
        scores = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size].to(self.device)

                # Extract features
                features = self.feature_extractor_(batch)

                # Compute reconstruction error across multiple timesteps
                reconstruction_errors = []
                sample_timesteps = [100, 300, 500, 700, 900]

                for t_val in sample_timesteps:
                    t = torch.full((len(batch),), t_val, device=self.device)

                    # Add noise
                    noisy_features, true_noise = self._add_noise(features, t)

                    # Predict noise
                    pred_noise = self.diffusion_model_(noisy_features, t)

                    # Reconstruction error
                    error = ((pred_noise - true_noise) ** 2).mean(dim=[1, 2, 3])
                    reconstruction_errors.append(error)

                # Average error across timesteps
                final_error = torch.stack(reconstruction_errors).mean(dim=0)
                scores.append(final_error.cpu().numpy())

        return np.concatenate(scores)

    def decision_function(self, X: NDArray, batch_size: Optional[int] = None) -> NDArray:
        """Alias for predict."""
        batch_size_int = validate_batch_size(batch_size)
        if batch_size_int is None:
            return self.predict(X)
        return call_with_temporary_attr(
            self,
            "batch_size",
            batch_size_int,
            lambda: self.predict(X),
        )
