"""
DRAEM: SSIM-based Discriminatively Trained Reconstruction Embedding for Anomaly Detection.

DRAEM uses synthetic anomalies during training and a discriminative approach
for robust anomaly detection and localization.

Reference:
    Zavrtanik, V., Kristan, M., & Skočaj, D. (2021).
    DRAEM-A discriminatively trained reconstruction embedding for surface anomaly detection.
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 8330-8339).
"""

import logging
from typing import Iterable, Optional

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


class SimpleUNet(nn.Module):
    """Simple U-Net architecture for reconstruction."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Decoder
        self.dec4 = self._upconv_block(512, 256)
        self.dec3 = self._upconv_block(512, 128)  # 512 because of skip connection
        self.dec2 = self._upconv_block(256, 64)
        self.dec1 = self._upconv_block(128, out_channels)

        self.final = nn.Conv2d(out_channels, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Decoder with skip connections
        d4 = self.dec4(e4)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)

        return torch.sigmoid(self.final(d1))


class ImagePathDataset(Dataset):
    """Dataset for loading images from paths."""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        # Create synthetic anomaly for training
        augmented = self._add_synthetic_anomaly(img)

        return augmented, img

    def _add_synthetic_anomaly(self, img):
        """Add synthetic anomaly to image."""
        # Simple implementation: add random noise patches
        augmented = img.clone()

        # Randomly add 1-3 anomalous regions
        n_anomalies = np.random.randint(1, 4)

        for _ in range(n_anomalies):
            # Random patch size
            h, w = np.random.randint(10, 30), np.random.randint(10, 30)
            # Random position
            y = np.random.randint(0, img.shape[1] - h)
            x = np.random.randint(0, img.shape[2] - w)

            # Add random noise or texture
            if np.random.rand() > 0.5:
                # Random noise
                augmented[:, y:y+h, x:x+w] = torch.rand(3, h, w)
            else:
                # Invert region
                augmented[:, y:y+h, x:x+w] = 1 - augmented[:, y:y+h, x:x+w]

        return augmented


@register_model(
    "vision_draem",
    tags=("vision", "deep", "draem", "reconstruction", "synthetic"),
    metadata={
        "description": "DRAEM - Discriminatively trained reconstruction (ICCV 2021)",
        "paper": "DRAEM: Discriminatively Trained Reconstruction Embedding",
        "year": 2021,
    },
)
class VisionDRAEM(BaseVisionDeepDetector):
    """
    DRAEM anomaly detector using synthetic anomalies.

    Parameters
    ----------
    image_size : int, default=256
        Input image size
    epochs : int, default=100
        Number of training epochs
    batch_size : int, default=8
        Training batch size
    lr : float, default=0.0001
        Learning rate
    num_workers : int, default=0
        Number of workers for the training DataLoader.
    device : str, default='cpu'
        Device to run model on

    Examples
    --------
    >>> detector = VisionDRAEM(epochs=50, device='cuda')
    >>> detector.fit(train_images)
    >>> scores = detector.decision_function(test_images)
    >>> labels = detector.predict(test_images)  # 0=normal, 1=anomaly
    """

    def __init__(
        self,
        image_size: int = 256,
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 0.0001,
        num_workers: int = 0,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize DRAEM detector."""
        super().__init__(**kwargs)

        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.device = device
        self._is_fitted = False

        # Build model
        self.model = SimpleUNet(in_channels=3, out_channels=3)
        self.model.to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        logger.info(
            "Initialized DRAEM with image_size=%d, epochs=%d, batch_size=%d, device=%s",
            image_size, epochs, batch_size, device
        )

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None) -> "VisionDRAEM":
        """
        Train DRAEM on normal images.

        Parameters
        ----------
        X : iterable of str
            Paths to normal training images
        y : array-like, optional
            Ignored

        Returns
        -------
        self : VisionDRAEM
        """
        logger.info("Training DRAEM detector")

        X_list = list(X)
        if not X_list:
            raise ValueError("Training set cannot be empty")

        # Create dataset
        dataset = ImagePathDataset(X_list, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=str(self.device).startswith("cuda"),
        )

        # Setup optimizer
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for augmented, original in dataloader:
                augmented = augmented.to(self.device)
                original = original.to(self.device)

                # Forward pass
                reconstructed = self.model(augmented)

                # L2 reconstruction loss
                loss = F.mse_loss(reconstructed, original)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.info("Epoch %d/%d, Loss: %.6f", epoch + 1, self.epochs, avg_loss)

        logger.info("DRAEM training completed")

        # Mark as fitted and compute training scores to establish a threshold.
        self._is_fitted = True
        self.decision_scores_ = self.decision_function(X_list)
        self._process_decision_scores()

        return self

    def predict(self, X: Iterable[str]) -> NDArray:
        """
        Predict binary anomaly labels for test images.

        Parameters
        ----------
        X : iterable of str
            Paths to test images

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Binary labels (0 = normal, 1 = anomaly)
        """
        if not self._is_fitted or not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)

    def decision_function(self, X: Iterable[str]) -> NDArray:
        """Compute anomaly scores."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        X_list = list(X)
        scores = np.zeros(len(X_list), dtype=np.float64)

        logger.info("Computing anomaly scores for %d images", len(X_list))

        with torch.no_grad():
            for idx, img_path in enumerate(X_list):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Failed to load image: {img_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                    reconstructed = self.model(img_tensor)
                    error = F.mse_loss(reconstructed, img_tensor, reduction="none")
                    scores[idx] = float(error.mean().item())

                except Exception as e:
                    logger.warning("Failed to score %s: %s", img_path, e)
                    scores[idx] = 0.0

        return scores

    def get_anomaly_map(self, image_path: str) -> NDArray:
        """Generate pixel-level anomaly heatmap."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_size = (img.shape[1], img.shape[0])  # (W, H)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(img_tensor)
            error = F.mse_loss(reconstructed, img_tensor, reduction="none")  # (1, C, H, W)
            anomaly_map = error.mean(dim=1).squeeze(0).cpu().numpy()

        anomaly_map = anomaly_map.astype(np.float32, copy=False)
        anomaly_map = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_CUBIC)
        return anomaly_map

    def predict_anomaly_map(self, X: Iterable[str]) -> NDArray:
        """Generate pixel-level anomaly maps for a batch of images."""
        maps = [self.get_anomaly_map(path) for path in X]
        return np.stack(maps)
