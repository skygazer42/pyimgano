"""
RD++ (Reverse Distillation++) for Anomaly Detection

Paper: "Reverse Distillation++: Enhanced Knowledge Distillation for Anomaly Detection"
Conference: Extended from AAAI 2022 Reverse Distillation

Key Innovation:
- Enhanced reverse distillation with multi-scale fusion
- Improved student-teacher architecture
- Better feature alignment and attention mechanisms
- Superior localization with hierarchical features

Implementation includes:
- Multi-scale feature extraction
- Enhanced reverse distillation loss
- Attention-guided fusion
- Progressive refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray

from pyimgano.models.base_dl import BaseVisionDeepDetector


class MultiScaleEncoder(nn.Module):
    """Multi-scale feature encoder (Teacher network)."""

    def __init__(self, backbone: str = "wide_resnet50", pretrained: bool = True):
        super().__init__()

        if backbone == "wide_resnet50":
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            model = wide_resnet50_2(weights=weights)

            self.layer1 = nn.Sequential(
                model.conv1, model.bn1, model.relu,
                model.maxpool, model.layer1
            )
            self.layer2 = model.layer2
            self.layer3 = model.layer3

            self.out_channels = [256, 512, 1024]
        elif backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)

            self.layer1 = nn.Sequential(
                model.conv1, model.bn1, model.relu,
                model.maxpool, model.layer1
            )
            self.layer2 = model.layer2
            self.layer3 = model.layer3

            self.out_channels = [64, 128, 256]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze teacher
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            List of features at different scales
        """
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        return [feat1, feat2, feat3]


class AttentionModule(nn.Module):
    """Channel and spatial attention for feature refinement."""

    def __init__(self, channels: int):
        super().__init__()

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Attention-refined features [B, C, H, W]
        """
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att

        # Spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att

        return x


class StudentDecoder(nn.Module):
    """Enhanced student decoder with attention."""

    def __init__(self, in_channels: List[int]):
        super().__init__()

        # Decoder layers (reverse order)
        self.decoder_layers = nn.ModuleList([
            self._make_decoder_block(in_channels[2], in_channels[1]),
            self._make_decoder_block(in_channels[1], in_channels[0]),
            self._make_decoder_block(in_channels[0], in_channels[0]),
        ])

        # Attention modules
        self.attention_modules = nn.ModuleList([
            AttentionModule(in_channels[1]),
            AttentionModule(in_channels[0]),
            AttentionModule(in_channels[0]),
        ])

        # Output projection
        self.output_proj = nn.Conv2d(in_channels[0], in_channels[0], 1)

    def _make_decoder_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create decoder block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, teacher_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode features.

        Args:
            teacher_features: Multi-scale teacher features

        Returns:
            Reconstructed features at each scale
        """
        student_features = []

        # Start from deepest features
        x = teacher_features[-1]

        for i, (decoder, attention) in enumerate(zip(self.decoder_layers, self.attention_modules)):
            # Decode
            x = decoder(x)

            # Apply attention
            x = attention(x)

            # Upsample for next level
            if i < len(self.decoder_layers) - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            student_features.append(x)

        return student_features


class RDPlusPlusDetector(BaseVisionDeepDetector):
    """RD++ (Reverse Distillation++) detector.

    Enhanced reverse distillation with multi-scale fusion and attention
    for improved anomaly detection and localization.

    Args:
        backbone: Feature extraction backbone
        epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        lambda_feat: Weight for feature distillation loss
        lambda_rec: Weight for reconstruction loss
        pretrained: Use pre-trained backbone
        device: Device to run on

    Example:
        >>> detector = RDPlusPlusDetector(
        ...     backbone="wide_resnet50",
        ...     epochs=100,
        ...     batch_size=8
        ... )
        >>> detector.fit(normal_images)
        >>> scores = detector.predict_proba(test_images)
        >>> anomaly_maps = detector.predict_anomaly_map(test_images)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 5e-4,
        lambda_feat: float = 1.0,
        lambda_rec: float = 1.0,
        pretrained: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.backbone_name = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_feat = lambda_feat
        self.lambda_rec = lambda_rec

        # Teacher encoder (frozen)
        self.teacher = MultiScaleEncoder(
            backbone=backbone,
            pretrained=pretrained
        ).to(self.device)

        # Student decoder (trainable)
        self.student = StudentDecoder(
            in_channels=self.teacher.out_channels
        ).to(self.device)

        self.fitted_ = False

    def _extract_features(self, images: NDArray) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Input images [N, H, W, C] or [N, C, H, W]

        Returns:
            Normalized images [N, 3, H, W]
        """
        # Convert to tensor
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).float()

        # Ensure [N, C, H, W] format
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:  # [N, H, W, C] -> [N, C, H, W]
            images = images.permute(0, 3, 1, 2)

        images = images.to(self.device)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images / 255.0 - mean) / std

        return images

    def _compute_loss(
        self,
        teacher_features: List[torch.Tensor],
        student_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute reverse distillation loss.

        Args:
            teacher_features: Teacher features at multiple scales
            student_features: Student reconstructed features

        Returns:
            Total loss
        """
        total_loss = 0.0

        # Feature distillation loss (cosine similarity)
        for t_feat, s_feat in zip(teacher_features, student_features):
            # Resize if needed
            if t_feat.shape != s_feat.shape:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)

            # Cosine similarity loss
            t_feat_norm = F.normalize(t_feat, dim=1)
            s_feat_norm = F.normalize(s_feat, dim=1)
            cos_sim = (t_feat_norm * s_feat_norm).sum(dim=1)  # [B, H, W]
            loss = (1 - cos_sim).mean()

            total_loss += self.lambda_feat * loss

        # Reconstruction loss (MSE)
        for t_feat, s_feat in zip(teacher_features, student_features):
            if t_feat.shape != s_feat.shape:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)

            rec_loss = F.mse_loss(s_feat, t_feat)
            total_loss += self.lambda_rec * rec_loss

        return total_loss / len(teacher_features)

    def fit(self, X: NDArray, y: Optional[NDArray] = None):
        """Fit the detector on normal images.

        Args:
            X: Normal images [N, H, W, C]
            y: Ignored (unsupervised)
        """
        # Training mode
        self.student.train()
        self.teacher.eval()

        # Optimizer
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)

        # Convert to tensor dataset
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
                images = self._extract_features(batch)

                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_features = self.teacher(images)

                # Student forward
                student_features = self.student(teacher_features)

                # Compute loss
                loss = self._compute_loss(teacher_features, student_features)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / num_batches:.4f}")

        # Evaluation mode
        self.student.eval()

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
        images = self._extract_features(X)

        # Forward pass
        with torch.no_grad():
            teacher_features = self.teacher(images)
            student_features = self.student(teacher_features)

        # Compute anomaly scores
        anomaly_maps = []
        for t_feat, s_feat in zip(teacher_features, student_features):
            # Resize
            if t_feat.shape != s_feat.shape:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)

            # Difference
            diff = torch.sum((t_feat - s_feat) ** 2, dim=1)  # [B, H, W]
            anomaly_maps.append(diff)

        # Aggregate across scales
        final_maps = torch.stack(anomaly_maps, dim=0).mean(dim=0)  # [B, H, W]

        # Image-level scores (max)
        scores = final_maps.view(final_maps.size(0), -1).max(dim=1)[0]

        return scores.cpu().numpy()

    def predict_anomaly_map(self, X: NDArray) -> NDArray:
        """Generate pixel-level anomaly maps.

        Args:
            X: Test images [N, H, W, C]

        Returns:
            Anomaly maps [N, H, W]
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get image size
        H_img, W_img = X.shape[1:3] if X.shape[-1] == 3 else X.shape[2:4]

        # Preprocess
        images = self._extract_features(X)

        # Forward pass
        with torch.no_grad():
            teacher_features = self.teacher(images)
            student_features = self.student(teacher_features)

        # Compute anomaly maps
        anomaly_maps = []
        for t_feat, s_feat in zip(teacher_features, student_features):
            # Resize
            if t_feat.shape != s_feat.shape:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)

            # Difference
            diff = torch.sum((t_feat - s_feat) ** 2, dim=1, keepdim=True)  # [B, 1, H, W]

            # Upsample to image size
            diff_upsampled = F.interpolate(
                diff,
                size=(H_img, W_img),
                mode='bilinear',
                align_corners=False
            )

            anomaly_maps.append(diff_upsampled)

        # Average across scales
        final_maps = torch.stack(anomaly_maps, dim=0).mean(dim=0).squeeze(1)  # [B, H, W]

        return final_maps.cpu().numpy()
