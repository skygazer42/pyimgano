from __future__ import annotations

"""
RIAD: Reconstruction from Adjacent Image Decomposition for Anomaly Detection.

Paper: https://arxiv.org/abs/2108.11092
Concept: Self-supervised learning through image inpainting from adjacent regions

Key Features:
- Self-supervised learning
- Image inpainting approach
- No anomaly samples needed
- Good reconstruction quality
- Efficient training
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from torch.utils.data import DataLoader, TensorDataset

from ._image_batch import coerce_rgb_image_batch
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


class ImageDecomposer:
    """Decomposes images into adjacent regions for RIAD training."""

    def __init__(
        self,
        n_splits: int = 16,
        mask_ratio: float = 0.5,
        random_state: Optional[int] = None,
    ):
        """Initialize decomposer.

        Args:
            n_splits: Number of splits (must be perfect square, e.g., 4, 9, 16).
            mask_ratio: Ratio of regions to mask.
            random_state: Optional seed for reproducible masking.
        """
        self.n_splits = n_splits
        self.grid_size = int(np.sqrt(n_splits))
        self.mask_ratio = mask_ratio
        self.rng = np.random.default_rng(random_state)

        if self.grid_size**2 != n_splits:
            raise ValueError("n_splits must be a perfect square")

    def decompose(self, image: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """Decompose image into masked and context regions.

        Args:
            image: Input image (H, W, C).

        Returns:
            Tuple of (masked_image, mask, target_regions).
        """
        h, w, c = image.shape

        # Calculate patch size
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size

        # Create mask
        n_mask = int(self.n_splits * self.mask_ratio)
        mask_indices = self.rng.choice(self.n_splits, n_mask, replace=False)

        # Create binary mask
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for idx in mask_indices:
            row = idx // self.grid_size
            col = idx % self.grid_size
            mask[row, col] = 1.0

        # Upsample mask to image size
        mask_upsampled = np.repeat(np.repeat(mask, patch_h, axis=0), patch_w, axis=1)

        # Extend to all channels
        mask_full = mask_upsampled[:, :, np.newaxis].repeat(c, axis=2)

        # Masked image (0 where masked)
        masked_image = image * (1 - mask_full)

        return masked_image, mask_full, image

    def random_mask(self, image: NDArray) -> Tuple[NDArray, NDArray]:
        """Create random mask for image.

        Args:
            image: Input image (H, W, C).

        Returns:
            Tuple of (masked_image, mask).
        """
        masked, mask, _ = self.decompose(image)
        return masked, mask


class UNet(nn.Module):
    """U-Net for image reconstruction."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)

    def _conv_block(self, in_c: int, out_c: int):
        """Convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Reconstructed tensor.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.out(d1)

        return out


@register_model(
    "vision_riad",
    tags=("vision", "deep", "riad", "reconstruction", "self-supervised", "pixel_map"),
    metadata={
        "description": "RIAD - reconstruction by adjacent image decomposition (2020-style)",
        "paper": "Reconstruction by Inpainting for Visual Anomaly Detection",
        "year": 2020,
    },
)
@register_model(
    "riad",
    tags=("vision", "deep", "riad", "reconstruction", "self-supervised", "pixel_map"),
    metadata={
        "description": "RIAD (legacy alias) - reconstruction by adjacent image decomposition",
        "year": 2020,
    },
)
class RIADDetector(BaseVisionDeepDetector):
    """RIAD anomaly detector.

    Self-supervised learning through reconstruction from adjacent image
    decomposition.

    Args:
        n_splits: Number of image splits (must be perfect square).
        mask_ratio: Ratio of regions to mask during training.
        image_size: Input image size (H, W).
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        device: Device to use ("cuda" or "cpu").

    References:
        RIAD: Reconstruction from Adjacent Image Decomposition.
    """

    def __init__(
        self,
        n_splits: int = 16,
        mask_ratio: float = 0.5,
        image_size: Tuple[int, int] = (256, 256),
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.0002,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_splits = n_splits
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

        # Decomposer
        self.decomposer = ImageDecomposer(n_splits, mask_ratio, random_state=random_state)

    def _build_model(self):
        """Build the RIAD model."""
        self.model = UNet(in_channels=3, out_channels=3).to(self.device)

    def fit(self, x: NDArray, y: Optional[NDArray] = None, **kwargs):
        """Train the RIAD model.

        Args:
            X: Training images (N, H, W, C).
            y: Not used (unsupervised).
        """
        del y, kwargs
        logger.info("Training RIAD model...")
        x = coerce_rgb_image_batch(x)

        if x.max() > 1.0:
            x = x.astype(np.float32) / 255.0

        # Prepare training data
        train_data = []

        for img in x:
            # Resize if needed
            if img.shape[:2] != self.image_size:
                import cv2

                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

            # Create masked version
            masked, _, target = self.decomposer.decompose(img)

            train_data.append((masked, target))

        # Create dataloader
        x_masked = torch.stack([torch.from_numpy(m.transpose(2, 0, 1)) for m, _ in train_data])
        x_target = torch.stack([torch.from_numpy(t.transpose(2, 0, 1)) for _, t in train_data])

        dataset = TensorDataset(x_masked.float(), x_target.float())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.0
        )
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for masked, target in dataloader:
                masked = masked.to(self.device)
                target = target.to(self.device)

                # Forward
                reconstructed = self.model(masked)

                # Loss
                loss = criterion(reconstructed, target)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.info("Epoch [%d/%d] Loss: %.6f", epoch + 1, self.epochs, avg_loss)

        self.model.eval()
        logger.info("Training completed!")

    def predict_proba(self, x: NDArray, **kwargs) -> NDArray:
        """Predict anomaly scores.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            Anomaly scores.
        """
        del kwargs
        x = coerce_rgb_image_batch(x)
        if x.max() > 1.0:
            x = x.astype(np.float32) / 255.0

        self.model.eval()
        scores = []

        with torch.no_grad():
            for img in x:
                # Resize if needed
                if img.shape[:2] != self.image_size:
                    import cv2

                    img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
                img_tensor = img_tensor.to(self.device)

                # Reconstruct
                reconstructed = self.model(img_tensor)

                # Reconstruction error
                error = F.mse_loss(reconstructed, img_tensor, reduction="none")
                error = error.mean(dim=1).squeeze()  # Average across channels

                # Image-level score (max error)
                score = error.max().item()
                scores.append(score)

        return np.array(scores)

    def decision_function(self, x: NDArray, batch_size: int | None = None, **kwargs) -> NDArray:
        del batch_size
        return np.asarray(self.predict_proba(x, **kwargs), dtype=np.float64).reshape(-1)

    def predict_anomaly_map(self, x: NDArray) -> list:
        """Predict pixel-level anomaly maps.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            List of anomaly maps.
        """
        if x.max() > 1.0:
            x = x.astype(np.float32) / 255.0

        self.model.eval()
        anomaly_maps = []

        with torch.no_grad():
            for img in x:
                # Resize if needed
                original_size = img.shape[:2]
                if img.shape[:2] != self.image_size:
                    import cv2

                    img_resized = cv2.resize(img, (self.image_size[1], self.image_size[0]))
                else:
                    img_resized = img

                img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float()
                img_tensor = img_tensor.to(self.device)

                # Reconstruct
                reconstructed = self.model(img_tensor)

                # Reconstruction error
                error = F.mse_loss(reconstructed, img_tensor, reduction="none")
                error = error.mean(dim=1).squeeze()  # Average across channels

                # Resize to original size if needed
                if original_size != self.image_size:
                    error = error.unsqueeze(0).unsqueeze(0)
                    error = F.interpolate(
                        error, size=original_size, mode="bilinear", align_corners=False
                    )
                    error = error.squeeze()

                anomaly_map = error.cpu().numpy()
                anomaly_maps.append(anomaly_map)

        return anomaly_maps
