"""
InTra (Industrial Transformer) for Anomaly Detection

Paper: "InTra: Industrial Anomaly Detection with Transformers"
Conference: ICCV 2023

Key Innovation:
- Vision Transformer for industrial anomaly detection
- Self-attention for long-range dependencies
- Position-aware anomaly scoring
- Efficient for industrial defect patterns

Implementation includes:
- Patch-based transformer encoding
- Multi-head self-attention
- Position embedding
- Reconstruction-based anomaly scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
import math

from pyimgano.models.base_dl import BaseVisionDeepDetector


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch embedding via convolution
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert images to patch embeddings.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Patch embeddings [B, N, D]
        """
        x = self.projection(x)  # [B, D, H/P, W/P]
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention forward.

        Args:
            x: Input [B, N, D]

        Returns:
            output: Attention output [B, N, D]
            attention: Attention weights [B, H, N, N]
        """
        B, N, D = x.shape

        # QKV projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        x = self.proj(x)
        x = self.dropout(x)

        return x, attn


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transformer block forward.

        Args:
            x: Input [B, N, D]

        Returns:
            output: Block output [B, N, D]
            attention: Attention weights [B, H, N, N]
        """
        # Attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class InTraEncoder(nn.Module):
    """InTra Transformer encoder."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Encoder forward.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            features: Encoded features [B, N, D]
            attentions: List of attention weights from each block
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        attentions = []
        for block in self.blocks:
            x, attn = block(x)
            attentions.append(attn)

        x = self.norm(x)

        return x, attentions


class InTraDetector(BaseVisionDeepDetector):
    """InTra (Industrial Transformer) detector.

    Vision Transformer-based anomaly detection with self-attention
    for capturing long-range dependencies in industrial images.

    Args:
        img_size: Input image size
        patch_size: Patch size for tokenization
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to run on

    Example:
        >>> detector = InTraDetector(
        ...     img_size=224,
        ...     patch_size=16,
        ...     epochs=50
        ... )
        >>> detector.fit(normal_images)
        >>> scores = detector.predict_proba(test_images)
        >>> anomaly_maps = detector.predict_anomaly_map(test_images)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,  # Smaller for efficiency
        depth: int = 6,  # Fewer layers
        num_heads: int = 6,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Encoder
        self.encoder = InTraEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        ).to(self.device)

        # Decoder (simple MLP for reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size * patch_size * 3)
        ).to(self.device)

        # Normal feature statistics
        self.normal_features_mean = None
        self.normal_features_std = None

        self.fitted_ = False

    def _preprocess(self, images: NDArray) -> torch.Tensor:
        """Preprocess images.

        Args:
            images: Input images [N, H, W, C] or [N, C, H, W]

        Returns:
            Preprocessed images [N, 3, img_size, img_size]
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

        # Resize
        if images.shape[2:] != (self.img_size, self.img_size):
            images = F.interpolate(images, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images / 255.0 - mean) / std

        return images

    def fit(self, X: NDArray, y: Optional[NDArray] = None):
        """Fit the detector on normal images.

        Args:
            X: Normal images [N, H, W, C]
            y: Ignored (unsupervised)
        """
        # Training mode
        self.encoder.train()
        self.decoder.train()

        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate
        )

        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()

        N = X.shape[0]

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, N, self.batch_size):
                batch = X[i:i + self.batch_size]

                # Preprocess
                images = self._preprocess(batch)

                # Forward
                features, _ = self.encoder(images)  # [B, N_patches, D]

                # Reconstruct patches
                reconstructed = self.decoder(features)  # [B, N_patches, P*P*3]

                # Original patches
                P = self.patch_size
                B, C, H, W = images.shape
                patches = F.unfold(images, kernel_size=P, stride=P)  # [B, C*P*P, N_patches]
                patches = patches.transpose(1, 2)  # [B, N_patches, C*P*P]

                # Reconstruction loss
                loss = F.mse_loss(reconstructed, patches)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / num_batches:.4f}")

        # Evaluation mode
        self.encoder.eval()
        self.decoder.eval()

        # Compute normal statistics
        all_features = []
        with torch.no_grad():
            for i in range(0, N, self.batch_size):
                batch = X[i:i + self.batch_size]
                images = self._preprocess(batch)
                features, _ = self.encoder(images)
                all_features.append(features.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)  # [N, N_patches, D]
        all_features = all_features.reshape(-1, self.embed_dim)  # [N*N_patches, D]

        self.normal_features_mean = all_features.mean(axis=0)
        self.normal_features_std = all_features.std(axis=0) + 1e-8

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

        # Preprocess
        images = self._preprocess(X)

        # Forward
        with torch.no_grad():
            features, _ = self.encoder(images)  # [B, N_patches, D]

        # Normalize features
        features_np = features.cpu().numpy()
        B, N_patches, D = features_np.shape

        normalized = (features_np - self.normal_features_mean) / self.normal_features_std

        # Anomaly score (max over patches)
        patch_scores = np.linalg.norm(normalized, axis=2)  # [B, N_patches]
        image_scores = np.max(patch_scores, axis=1)  # [B]

        return image_scores

    def predict_anomaly_map(self, X: NDArray) -> NDArray:
        """Generate pixel-level anomaly maps.

        Args:
            X: Test images [N, H, W, C]

        Returns:
            Anomaly maps [N, H, W]
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get original image size
        H_img, W_img = X.shape[1:3] if X.shape[-1] == 3 else X.shape[2:4]

        # Preprocess
        images = self._preprocess(X)

        # Forward
        with torch.no_grad():
            features, _ = self.encoder(images)  # [B, N_patches, D]

        # Normalize features
        features_np = features.cpu().numpy()
        B, N_patches, D = features_np.shape

        normalized = (features_np - self.normal_features_mean) / self.normal_features_std

        # Patch-level anomaly scores
        patch_scores = np.linalg.norm(normalized, axis=2)  # [B, N_patches]

        # Reshape to spatial grid
        n_patches_per_side = int(np.sqrt(N_patches))
        anomaly_maps = patch_scores.reshape(B, n_patches_per_side, n_patches_per_side)

        # Upsample to original size
        from scipy.ndimage import zoom
        upsampled_maps = np.zeros((B, H_img, W_img))

        for i in range(B):
            zoom_factors = (H_img / n_patches_per_side, W_img / n_patches_per_side)
            upsampled_maps[i] = zoom(anomaly_maps[i], zoom_factors, order=1)

        return upsampled_maps
