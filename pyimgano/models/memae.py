"""
MemAE: Memory-Augmented Autoencoder

Uses a memory module to store prototypical patterns of normal data.
During reconstruction, features are retrieved from memory, making it
harder to reconstruct anomalies.

Reference:
    Gong, D., et al. (2019). "Memorizing Normality to Detect Anomaly:
    Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection"
    ICCV 2019.

Usage:
    >>> from pyimgano.models import MemAE
    >>> model = MemAE(mem_dim=500, shrink_thres=0.0025)
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


class MemoryModule(nn.Module):
    """Paper memory addressing with cosine attention and sparse shrinkage."""

    def __init__(self, mem_dim: int, fea_dim: int, shrink_thres: float = 0.0025):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres

        self.memory = nn.Parameter(torch.empty(mem_dim, fea_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize memory with the authors' uniform distribution."""
        stdv = 1.0 / np.sqrt(self.memory.size(1))
        with torch.no_grad():
            self.memory.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor):
        """
        Retrieve from memory using attention mechanism.

        Parameters
        ----------
        x : torch.Tensor
            Input features (B, C, H, W)

        Returns
        -------
        output : torch.Tensor
            Retrieved features
        att_weight : torch.Tensor
            Attention weights
        """
        batch_size = x.size(0)

        # Reshape: (B, C, H, W) -> (B, HW, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.fea_dim)

        # Equations 4-5: softmax over cosine similarities to memory items.
        queries = F.normalize(x_flat, p=2, dim=2)
        memory_directions = F.normalize(self.memory, p=2, dim=1)
        att_weight = F.linear(queries, memory_directions)  # (B, HW, mem_dim)
        att_weight = F.softmax(att_weight, dim=2)

        # Hard shrinkage
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, self.shrink_thres)
            # Re-normalize
            att_weight = F.normalize(att_weight, p=1, dim=2)

        # Retrieve from memory: (B, HW, C)
        output = F.linear(att_weight, self.memory.permute(1, 0))

        # Reshape back: (B, HW, C) -> (B, C, H, W)
        h, w = x.size(2), x.size(3)
        output = output.reshape(batch_size, h, w, self.fea_dim)
        output = output.permute(0, 3, 1, 2)

        # Match the paper/author module convention: memory is the channel axis.
        att_weight = att_weight.reshape(batch_size, h, w, self.mem_dim)
        att_weight = att_weight.permute(0, 3, 1, 2)
        return output, att_weight


def hard_shrink_relu(x: torch.Tensor, threshold: float = 0.5):
    """Differentiable hard shrinkage from MemAE equation 7."""
    return (F.relu(x - threshold) * x) / (torch.abs(x - threshold) + 1e-12)


def memory_entropy(att_weight: torch.Tensor) -> torch.Tensor:
    """Mean entropy of each spatial memory-addressing vector."""
    if att_weight.ndim != 4:
        raise ValueError("att_weight must have shape (B, M, H, W).")
    flattened = att_weight.permute(0, 2, 3, 1).reshape(-1, att_weight.shape[1])
    return -(flattened * torch.log(flattened + 1e-12)).sum(dim=1).mean()


def _initialize_memae_weights(module: nn.Module) -> None:
    """Initialization used by the authors' released training code."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


class MemAENetwork(nn.Module):
    """MemAE's CIFAR-10 2D convolutional architecture."""

    def __init__(self, in_channels: int = 3, mem_dim: int = 500, shrink_thres: float = 0.0025):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # Memory module
        self.memory = MemoryModule(mem_dim, 256, shrink_thres)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, in_channels, 3, stride=2, padding=1, output_padding=1),
        )
        self.apply(_initialize_memae_weights)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        # Encode
        z = self.encoder(x)

        # Memory read
        z_mem, att_weight = self.memory(z)

        # Decode
        recon = self.decoder(z_mem)

        return recon, z, z_mem, att_weight


@register_model(
    "vision_memae",
    tags=("vision", "deep", "memae", "autoencoder", "memory", "reconstruction"),
    metadata={
        "description": "MemAE CIFAR-10 network adapted to industrial RGB images with paper memory addressing",
        "paper": "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection",
        "paper_url": "https://arxiv.org/abs/1904.02639",
        "year": 2019,
        "implementation_status": "paper-cifar-network-adapted-to-industrial-images",
        "paper_fidelity": "paper-adaptation",
    },
)
class MemAE(BaseVisionDeepDetector):
    """
    Memory-Augmented Autoencoder for anomaly detection.

    Uses a memory module to store prototypical patterns of normal data.
    Anomalies are detected based on reconstruction error, as they cannot
    be well reconstructed using normal patterns from memory.

    Parameters
    ----------
    mem_dim : int, default=500
        Memory dimension (number of memory items)
    shrink_thres : float, default=0.0025
        Shrinkage threshold for hard attention
    entropy_weight : float, default=0.0002
        Weight for entropy loss (encourages sparse addressing)
    learning_rate : float, default=1e-4
        Learning rate for Adam optimizer
    batch_size : int, default=32
        Batch size for training
    epochs : int, default=100
        Number of training epochs
    device : str, default='cuda'
        Device for training

    Attributes
    ----------
    network_ : MemAENetwork
        The memory-augmented autoencoder network

    Examples
    --------
    >>> model = MemAE(mem_dim=500, shrink_thres=0.0025)
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
    """

    def __init__(
        self,
        mem_dim: int = 500,
        shrink_thres: float = 0.0025,
        entropy_weight: float = 0.0002,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = "cuda",
    ):
        super().__init__()
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.entropy_weight = entropy_weight
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.network_ = None

    def fit(self, x: NDArray, y: Optional[NDArray] = None) -> "MemAE":
        """
        Fit MemAE model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width, channels)
            Training images (normal only)
        y : ndarray, optional
            Ignored

        Returns
        -------
        self : MemAE
            Fitted estimator
        """
        del y
        x = np.asarray(x)
        x_original = x.copy()
        # Convert to torch tensor
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)

        x = np.transpose(x, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(np.ascontiguousarray(x)).float() / 127.5 - 1.0

        # Initialize network
        self.network_ = MemAENetwork(
            in_channels=x.shape[1], mem_dim=self.mem_dim, shrink_thres=self.shrink_thres
        ).to(self.device)

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=0.0
        )

        # Training loop
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.network_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0

            for (batch,) in dataloader:
                batch = batch.to(self.device)

                # Forward pass
                recon, _, _, att_weight = self.network_(batch)

                # Reconstruction loss
                recon_loss = F.mse_loss(recon, batch)

                # Entropy loss (encourages sparse memory addressing)
                entropy_loss = memory_entropy(att_weight)

                # Total loss
                loss = recon_loss + self.entropy_weight * entropy_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, self.epochs, avg_loss)

        self.is_fitted_ = True
        self.decision_scores_ = self.decision_function(x_original)
        self._process_decision_scores()
        self._set_n_classes(None)
        return self

    def decision_function(self, x: NDArray) -> NDArray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (reconstruction error)
        """
        self._check_is_fitted()

        # Preprocess
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)

        x = np.transpose(x, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(np.ascontiguousarray(x)).float() / 127.5 - 1.0

        # Compute scores
        self.network_.eval()
        scores = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)
                recon, _, _, _ = self.network_(batch)

                # MSE reconstruction error
                mse = torch.mean((recon - batch) ** 2, dim=(1, 2, 3))
                scores.append(mse.cpu().numpy())

        return np.concatenate(scores)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "mem_dim": self.mem_dim,
            "shrink_thres": self.shrink_thres,
            "entropy_weight": self.entropy_weight,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": str(self.device),
        }
