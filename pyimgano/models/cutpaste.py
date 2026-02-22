"""
CutPaste: Self-Supervised Learning for Anomaly Detection and Localization.

Paper: https://arxiv.org/abs/2104.04015
Conference: CVPR 2021

CutPaste is a simple data augmentation strategy that cuts and pastes rectangular patches
within an image, creating synthetic anomalies for self-supervised learning.

Key Features:
- Self-supervised approach (no anomaly data needed)
- Simple but effective augmentation strategy
- Good localization performance
- Fast training and inference
"""

import warnings
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class CutPasteAugmentation:
    """CutPaste augmentation strategies."""

    def __init__(
        self,
        area_ratio: Tuple[float, float] = (0.02, 0.15),
        aspect_ratio: Tuple[float, float] = (0.3, 1 / 0.3),
        type: str = "normal",  # "normal", "scar", "3way"
    ):
        """Initialize CutPaste augmentation.

        Args:
            area_ratio: Range of area ratio for cut patch.
            aspect_ratio: Range of aspect ratio for cut patch.
            type: Type of CutPaste ("normal", "scar", "3way").
        """
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.type = type

    def __call__(self, image: NDArray) -> NDArray:
        """Apply CutPaste augmentation.

        Args:
            image: Input image (H, W, C).

        Returns:
            Augmented image.
        """
        if self.type == "normal":
            return self.cutpaste_normal(image)
        elif self.type == "scar":
            return self.cutpaste_scar(image)
        elif self.type == "3way":
            # 3-way classification: normal, normal cutpaste, scar cutpaste
            if np.random.rand() < 0.5:
                return self.cutpaste_normal(image)
            else:
                return self.cutpaste_scar(image)
        else:
            raise ValueError(f"Unknown CutPaste type: {self.type}")

    def cutpaste_normal(self, image: NDArray) -> NDArray:
        """Apply normal CutPaste augmentation.

        Cuts a rectangular patch and pastes it at a random location.
        """
        h, w = image.shape[:2]

        # Sample patch size
        area = h * w
        target_area = np.random.uniform(*self.area_ratio) * area
        aspect = np.random.uniform(*self.aspect_ratio)

        patch_h = int(np.sqrt(target_area * aspect))
        patch_w = int(np.sqrt(target_area / aspect))

        # Ensure patch fits in image
        patch_h = min(patch_h, h - 1)
        patch_w = min(patch_w, w - 1)

        # Random source location
        src_y = np.random.randint(0, h - patch_h)
        src_x = np.random.randint(0, w - patch_w)

        # Random target location
        dst_y = np.random.randint(0, h - patch_h)
        dst_x = np.random.randint(0, w - patch_w)

        # Cut and paste
        patch = image[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()

        # Optionally rotate patch
        if np.random.rand() < 0.5:
            angle = np.random.randint(0, 360)
            M = cv2.getRotationMatrix2D((patch_w // 2, patch_h // 2), angle, 1.0)
            patch = cv2.warpAffine(patch, M, (patch_w, patch_h))

        result = image.copy()
        result[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = patch

        return result

    def cutpaste_scar(self, image: NDArray) -> NDArray:
        """Apply scar CutPaste augmentation.

        Cuts a thin elongated patch and pastes it at a random location.
        """
        h, w = image.shape[:2]

        # Scar is very elongated
        area = h * w
        target_area = np.random.uniform(0.01, 0.05) * area
        aspect = np.random.uniform(2.0, 4.0)  # More elongated

        patch_h = int(np.sqrt(target_area * aspect))
        patch_w = int(np.sqrt(target_area / aspect))

        # Ensure patch fits
        patch_h = min(patch_h, h - 1)
        patch_w = max(3, min(patch_w, w - 1))  # At least 3 pixels wide

        # Random source and target
        src_y = np.random.randint(0, h - patch_h)
        src_x = np.random.randint(0, w - patch_w)
        dst_y = np.random.randint(0, h - patch_h)
        dst_x = np.random.randint(0, w - patch_w)

        # Cut and paste
        patch = image[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()

        # Always rotate scar
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((patch_w // 2, patch_h // 2), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (patch_w, patch_h))

        result = image.copy()
        result[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = patch

        return result


class CutPasteDataset(Dataset):
    """Dataset for CutPaste training."""

    def __init__(
        self,
        images: NDArray,
        transform=None,
        augment_type: str = "normal",
    ):
        """Initialize dataset.

        Args:
            images: Array of images (N, H, W, C).
            transform: Transform to apply to images.
            augment_type: Type of CutPaste augmentation.
        """
        self.images = images
        self.transform = transform
        self.augmenter = CutPasteAugmentation(type=augment_type)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Original image (label 0)
        original = image.copy()

        # Augmented image (label 1)
        augmented = self.augmenter(image)

        if self.transform:
            original = self.transform(original)
            augmented = self.transform(augmented)

        return original, augmented, 0, 1  # Returns (orig, aug, label_orig, label_aug)


class ProjectionHead(nn.Module):
    """Projection head for CutPaste."""

    def __init__(self, in_features: int, hidden_dim: int = 512, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@register_model(
    "vision_cutpaste",
    tags=("vision", "deep", "cutpaste", "self-supervised", "cvpr2021"),
    metadata={
        "description": "CutPaste - self-supervised anomaly detection via synthetic cut/paste (CVPR 2021)",
        "paper": "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization",
        "year": 2021,
    },
)
@register_model(
    "cutpaste",
    tags=("vision", "deep", "cutpaste", "self-supervised", "cvpr2021"),
    metadata={
        "description": "CutPaste (legacy alias) - self-supervised anomaly detection via synthetic cut/paste",
        "year": 2021,
    },
)
class CutPasteDetector(BaseVisionDeepDetector):
    """CutPaste anomaly detector.

    Self-supervised learning using synthetic anomalies created by
    cutting and pasting image patches.

    Args:
        backbone: Backbone architecture ("resnet18", "resnet50", "efficientnet").
        embedding_dim: Dimension of feature embeddings.
        augment_type: Type of CutPaste ("normal", "scar", "3way").
        pretrained: Whether to use pretrained backbone.
        freeze_backbone: Whether to freeze backbone during training.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        device: Device to use ("cuda" or "cpu").

    References:
        Li et al. "CutPaste: Self-Supervised Learning for Anomaly Detection
        and Localization." CVPR 2021.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 512,
        augment_type: str = "normal",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        epochs: int = 256,
        batch_size: int = 96,
        learning_rate: float = 0.03,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.augment_type = augment_type
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

    def _build_model(self):
        """Build the CutPaste model."""
        # Load backbone
        if self.backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=self.pretrained)
            feature_dim = 512
        elif self.backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=self.pretrained)
            feature_dim = 2048
        elif self.backbone_name == "wide_resnet50":
            self.backbone = models.wide_resnet50_2(pretrained=self.pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head for binary classification
        num_classes = 2 if self.augment_type != "3way" else 3
        self.projection_head = nn.Linear(feature_dim, num_classes)

        self.backbone.to(self.device)
        self.projection_head.to(self.device)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Feature tensor (B, D).
        """
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)
        return features

    def fit(self, X: NDArray, y: Optional[NDArray] = None, **kwargs):
        """Train the CutPaste detector.

        Args:
            X: Training images (N, H, W, C) or (N, C, H, W).
            y: Not used (unsupervised).
        """
        # Normalize to [0, 1] if needed
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        # Create dataset
        dataset = CutPasteDataset(
            X,
            transform=self._get_transform(),
            augment_type=self.augment_type,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Optimizer and loss
        optimizer = torch.optim.SGD(
            list(self.backbone.parameters()) + list(self.projection_head.parameters()),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.00003,
        )

        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.backbone.train()
        self.projection_head.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for orig, aug, label_orig, label_aug in dataloader:
                # Combine original and augmented
                images = torch.cat([orig, aug], dim=0).to(self.device)
                labels = torch.cat([label_orig, label_aug], dim=0).to(self.device)

                # Forward
                features = self._extract_features(images)
                logits = self.projection_head(features)

                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            if (epoch + 1) % 10 == 0:
                acc = 100.0 * correct / total
                print(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"Loss: {epoch_loss/len(dataloader):.4f} "
                    f"Acc: {acc:.2f}%"
                )

        # Switch to eval mode
        self.backbone.eval()
        self.projection_head.eval()

        # Build reference statistics from training data
        self._build_reference(X)

    def _build_reference(self, X: NDArray):
        """Build reference feature distribution.

        Args:
            X: Training images.
        """
        self.backbone.eval()

        features_list = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i : i + self.batch_size]
                batch_tensor = self._preprocess(batch).to(self.device)
                features = self._extract_features(batch_tensor)
                features_list.append(features.cpu().numpy())

        features = np.vstack(features_list)

        # Compute statistics
        self.reference_mean = features.mean(axis=0)
        self.reference_std = features.std(axis=0)

    def predict_proba(self, X: NDArray, **kwargs) -> NDArray:
        """Predict anomaly scores.

        Args:
            X: Test images (N, H, W, C) or (N, C, H, W).

        Returns:
            Anomaly scores for each sample.
        """
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        self.backbone.eval()
        scores = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i : i + self.batch_size]
                batch_tensor = self._preprocess(batch).to(self.device)

                # Extract features
                features = self._extract_features(batch_tensor).cpu().numpy()

                # Compute distance from reference distribution
                dist = np.linalg.norm(
                    (features - self.reference_mean) / (self.reference_std + 1e-8),
                    axis=1,
                )

                scores.append(dist)

        return np.concatenate(scores)

    def _get_transform(self):
        """Get data transform."""
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _preprocess(self, images: NDArray) -> torch.Tensor:
        """Preprocess images for inference.

        Args:
            images: Input images (N, H, W, C).

        Returns:
            Preprocessed tensor (N, C, H, W).
        """
        transform = self._get_transform()

        batch = []
        for img in images:
            img_tensor = transform(img)
            batch.append(img_tensor)

        return torch.stack(batch)
