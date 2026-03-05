"""
InCTRL - In-context Residual Learning for Anomaly Detection

Reference:
    "Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts"
    CVPR 2024

A generalist anomaly detection model that can generalize across diverse datasets
from different application domains without further training on target data.
Uses in-context learning with few-shot sample prompts.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class ResidualEncoder(nn.Module):
    """Encoder for extracting residual features."""

    def __init__(self, backbone: str = "wide_resnet50", feature_dim: int = 512):
        super().__init__()

        # Feature extractor
        if backbone == "wide_resnet50":
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            resnet = wide_resnet50_2(weights=weights)
            backbone_dim = 1024
        elif backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Projection head
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project features."""
        with torch.no_grad():
            features = self.backbone(x)
        projected = self.projection(features)
        return F.normalize(projected, p=2, dim=1)


class InContextAttention(nn.Module):
    """In-context attention mechanism for few-shot learning."""

    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute in-context attention.

        Parameters
        ----------
        query : torch.Tensor
            Query features (B, D)
        context : torch.Tensor
            Context features (K, D) - few-shot samples

        Returns
        -------
        attended : torch.Tensor
            Context-attended features (B, D)
        """
        B = query.size(0)
        K = context.size(0)

        # Project
        Q = self.q_proj(query).view(B, self.num_heads, self.head_dim)
        K_ctx = self.k_proj(context).view(K, self.num_heads, self.head_dim)
        V_ctx = self.v_proj(context).view(K, self.num_heads, self.head_dim)

        # Attention scores
        scores = torch.einsum("bhd,khd->bhk", Q, K_ctx) / (self.head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, K)

        # Aggregate
        attended = torch.einsum("bhk,khd->bhd", attn_weights, V_ctx)
        attended = attended.reshape(B, -1)

        # Output projection
        out = self.out_proj(attended)
        return out


class ResidualPredictor(nn.Module):
    """Predicts residuals for anomaly detection."""

    def __init__(self, feature_dim: int):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, query_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        """
        Predict residual features.

        Parameters
        ----------
        query_feat : torch.Tensor
            Query features (B, D)
        context_feat : torch.Tensor
            Context-attended features (B, D)

        Returns
        -------
        residual : torch.Tensor
            Predicted residual (B, D)
        """
        combined = torch.cat([query_feat, context_feat], dim=1)
        residual = self.predictor(combined)
        return residual


@register_model(
    "vision_inctrl",
    tags=("vision", "deep", "inctrl", "few-shot", "generalist", "cvpr2024", "sota"),
    metadata={
        "description": "InCTRL - In-context Residual Learning for generalist AD (CVPR 2024)",
        "paper": "Toward Generalist Anomaly Detection via In-context Residual Learning",
        "year": 2024,
        "conference": "CVPR",
        "type": "few-shot",
    },
)
class VisionInCTRL(BaseVisionDeepDetector):
    """
    InCTRL: In-context Residual Learning for Anomaly Detection.

    A generalist anomaly detection model that generalizes across diverse
    datasets using in-context learning with few-shot sample prompts.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    feature_dim : int, default=512
        Feature dimension
    num_heads : int, default=8
        Number of attention heads
    learning_rate : float, default=1e-4
        Learning rate
    batch_size : int, default=16
        Batch size for training
    epochs : int, default=40
        Number of training epochs
    k_shot : int, default=5
        Number of few-shot context samples
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    encoder_ : ResidualEncoder
        Feature encoder
    in_context_attn_ : InContextAttention
        In-context attention module
    residual_predictor_ : ResidualPredictor
        Residual prediction module
    context_samples_ : torch.Tensor
        Few-shot context samples

    Examples
    --------
    >>> from pyimgano.models import VisionInCTRL
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    >>> X_test = np.random.rand(20, 224, 224, 3).astype(np.float32)
    >>>
    >>> # Create and train detector (generalist model)
    >>> detector = VisionInCTRL(k_shot=5, epochs=30)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict with few-shot context
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        feature_dim: int = 512,
        num_heads: int = 8,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 40,
        k_shot: int = 5,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_shot = k_shot
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.encoder_ = None
        self.in_context_attn_ = None
        self.residual_predictor_ = None
        self.context_samples_ = None

    def _preprocess(self, X: NDArray) -> torch.Tensor:
        """Preprocess images."""
        if X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))

        X = X.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        X = (X - mean) / std

        return torch.from_numpy(X).float()

    def _sample_context(self, X_tensor: torch.Tensor, k: int) -> torch.Tensor:
        """Sample k-shot context samples."""
        indices = torch.randperm(len(X_tensor))[:k]
        return X_tensor[indices]

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "VisionInCTRL":
        """
        Fit the InCTRL detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionInCTRL
            Fitted detector
        """
        # Preprocess
        X_tensor = self._preprocess(X)

        # Initialize modules
        if self.encoder_ is None:
            self.encoder_ = ResidualEncoder(
                backbone=self.backbone, feature_dim=self.feature_dim
            ).to(self.device)

        if self.in_context_attn_ is None:
            self.in_context_attn_ = InContextAttention(
                feature_dim=self.feature_dim, num_heads=self.num_heads
            ).to(self.device)

        if self.residual_predictor_ is None:
            self.residual_predictor_ = ResidualPredictor(feature_dim=self.feature_dim).to(
                self.device
            )

        # Sample context samples
        context_images = self._sample_context(X_tensor, self.k_shot)
        with torch.no_grad():
            self.context_samples_ = self.encoder_(context_images.to(self.device))

        # Training
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(
            list(self.in_context_attn_.parameters()) + list(self.residual_predictor_.parameters()),
            lr=self.learning_rate,
        )

        self.in_context_attn_.train()
        self.residual_predictor_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Extract features
                with torch.no_grad():
                    query_features = self.encoder_(batch_images)

                # In-context attention
                context_features = self.in_context_attn_(query_features, self.context_samples_)

                # Predict residuals
                residuals = self.residual_predictor_(query_features, context_features)

                # Loss: residuals should be small for normal samples
                loss = F.mse_loss(residuals, torch.zeros_like(residuals))

                # Add consistency loss
                consistency_loss = F.mse_loss(query_features + residuals, context_features)
                loss = loss + 0.1 * consistency_loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : NDArray of shape (n_samples,)
            Anomaly scores (residual magnitude)
        """
        self.in_context_attn_.eval()
        self.residual_predictor_.eval()

        X_tensor = self._preprocess(X)
        scores = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size].to(self.device)

                # Extract features
                query_features = self.encoder_(batch)

                # In-context attention
                context_features = self.in_context_attn_(query_features, self.context_samples_)

                # Predict residuals
                residuals = self.residual_predictor_(query_features, context_features)

                # Anomaly score = residual magnitude
                score = torch.norm(residuals, p=2, dim=1)
                scores.append(score.cpu().numpy())

        return np.concatenate(scores)

    def decision_function(self, X: NDArray) -> NDArray:
        """Alias for predict."""
        return self.predict(X)
