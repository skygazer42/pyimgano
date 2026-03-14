"""
AST - Anomaly-aware Student-Teacher Network

Reference:
    "Anomaly-aware Student-Teacher Network for Industrial Inspection"

Enhances student-teacher framework with anomaly-aware training using
synthetic anomalies to improve detection sensitivity.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class TeacherEncoder(nn.Module):
    """Pre-trained teacher encoder."""

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

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # Freeze
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        with torch.no_grad():
            return self.features(x)


class AnomalyAwareStudent(nn.Module):
    """Student network with anomaly awareness."""

    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()

        # Feature decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

        # Anomaly score predictor
        self.anomaly_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        reconstructed : torch.Tensor
            Reconstructed features
        anomaly_map : torch.Tensor
            Predicted anomaly score map
        """
        reconstructed = self.decoder(x)
        anomaly_map = self.anomaly_predictor(x)
        return reconstructed, anomaly_map


@register_model(
    "vision_ast",
    tags=("vision", "deep", "ast", "student-teacher", "anomaly-aware", "sota"),
    metadata={
        "description": "AST - Anomaly-aware Student-Teacher with synthetic anomalies",
        "paper": "Anomaly-aware Student-Teacher Network",
        "year": 2023,
        "type": "knowledge-distillation",
    },
)
class VisionAST(BaseVisionDeepDetector):
    """
    AST: Anomaly-aware Student-Teacher Network.

    Enhances student-teacher framework by training with synthetic anomalies
    to improve anomaly detection sensitivity.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Teacher backbone network
    hidden_channels : int, default=256
        Hidden channels in student decoder
    learning_rate : float, default=1e-4
        Learning rate for training
    batch_size : int, default=16
        Batch size for training
    epochs : int, default=50
        Number of training epochs
    anomaly_ratio : float, default=0.3
        Ratio of synthetic anomalies in training
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    teacher_ : TeacherEncoder
        Pre-trained teacher encoder
    student_ : AnomalyAwareStudent
        Anomaly-aware student network

    Examples
    --------
    >>> from pyimgano.models import VisionAST
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    >>> X_test = np.random.rand(20, 224, 224, 3).astype(np.float32)
    >>>
    >>> # Create and train detector
    >>> detector = VisionAST(epochs=30)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        hidden_channels: int = 256,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 50,
        anomaly_ratio: float = 0.3,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.anomaly_ratio = anomaly_ratio
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.teacher_ = None
        self.student_ = None

    def _preprocess(self, X: NDArray) -> torch.Tensor:
        """Preprocess images."""
        # Convert to CHW format if needed
        if X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))

        # Normalize
        X = X.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        X = (X - mean) / std

        return torch.from_numpy(X).float()

    def _generate_synthetic_anomalies(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic anomalies for training.

        Parameters
        ----------
        images : torch.Tensor
            Normal images (B, C, H, W)

        Returns
        -------
        anomalous_images : torch.Tensor
            Images with synthetic anomalies
        anomaly_masks : torch.Tensor
            Binary masks indicating anomaly locations
        """
        B, C, H, W = images.shape
        anomalous_images = images.clone()
        anomaly_masks = torch.zeros(B, 1, H, W, device=images.device)

        for i in range(B):
            # Random anomaly type
            anomaly_type = np.random.choice(["cutpaste", "noise", "blur"])

            # Random anomaly region
            h_size = np.random.randint(H // 8, H // 3)
            w_size = np.random.randint(W // 8, W // 3)
            y = np.random.randint(0, H - h_size)
            x = np.random.randint(0, W - w_size)

            if anomaly_type == "cutpaste":
                # Paste from random location
                src_y = np.random.randint(0, H - h_size)
                src_x = np.random.randint(0, W - w_size)
                anomalous_images[i, :, y : y + h_size, x : x + w_size] = images[
                    i, :, src_y : src_y + h_size, src_x : src_x + w_size
                ]

            elif anomaly_type == "noise":
                # Add Gaussian noise
                noise = torch.randn(C, h_size, w_size, device=images.device) * 0.5
                anomalous_images[i, :, y : y + h_size, x : x + w_size] += noise

            elif anomaly_type == "blur":
                # Apply blur
                region = anomalous_images[i, :, y : y + h_size, x : x + w_size]
                region = F.avg_pool2d(
                    region.unsqueeze(0), kernel_size=5, stride=1, padding=2
                ).squeeze(0)
                anomalous_images[i, :, y : y + h_size, x : x + w_size] = region

            # Mark anomaly location
            anomaly_masks[i, 0, y : y + h_size, x : x + w_size] = 1.0

        return anomalous_images, anomaly_masks

    def _compute_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        anomaly_map: torch.Tensor,
        ground_truth_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training loss.

        Returns
        -------
        total_loss : torch.Tensor
            Total loss
        recon_loss : torch.Tensor
            Reconstruction loss
        anomaly_loss : torch.Tensor
            Anomaly prediction loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(student_features, teacher_features)

        # Anomaly prediction loss
        # Resize ground truth to match anomaly map
        if anomaly_map.shape[2:] != ground_truth_mask.shape[2:]:
            ground_truth_mask = F.interpolate(
                ground_truth_mask, size=anomaly_map.shape[2:], mode="nearest"
            )

        anomaly_loss = F.binary_cross_entropy(anomaly_map, ground_truth_mask)

        # Total loss
        total_loss = recon_loss + 0.5 * anomaly_loss

        return total_loss, recon_loss, anomaly_loss

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "VisionAST":
        """
        Fit the AST detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionAST
            Fitted detector
        """
        # Preprocess
        X_tensor = self._preprocess(X)

        # Initialize teacher
        if self.teacher_ is None:
            self.teacher_ = TeacherEncoder(backbone=self.backbone).to(self.device)

        # Get feature dimensions
        with torch.no_grad():
            sample_features = self.teacher_(X_tensor[:1].to(self.device))
            in_channels = sample_features.shape[1]

        # Initialize student
        if self.student_ is None:
            self.student_ = AnomalyAwareStudent(
                in_channels=in_channels, hidden_channels=self.hidden_channels
            ).to(self.device)

        # Training
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.student_.parameters(), lr=self.learning_rate)

        self.student_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_anomaly = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Create mixed batch: normal + synthetic anomalies
                n_normal = int(len(batch_images) * (1 - self.anomaly_ratio))
                n_anomaly = len(batch_images) - n_normal

                normal_images = batch_images[:n_normal]
                anomaly_images = batch_images[n_normal:]

                # Generate synthetic anomalies
                if n_anomaly > 0:
                    synthetic_anomalies, anomaly_masks = self._generate_synthetic_anomalies(
                        anomaly_images
                    )
                    # Combine
                    mixed_images = torch.cat([normal_images, synthetic_anomalies], dim=0)
                    mixed_masks = torch.cat(
                        [
                            torch.zeros(n_normal, 1, *normal_images.shape[2:], device=self.device),
                            anomaly_masks,
                        ],
                        dim=0,
                    )
                else:
                    mixed_images = normal_images
                    mixed_masks = torch.zeros(
                        n_normal, 1, *normal_images.shape[2:], device=self.device
                    )

                # Extract teacher features
                teacher_features = self.teacher_(mixed_images)

                # Student prediction
                student_features, anomaly_map = self.student_(teacher_features)

                # Compute loss
                loss, recon_loss, anomaly_loss = self._compute_loss(
                    student_features, teacher_features, anomaly_map, mixed_masks
                )

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_anomaly += anomaly_loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                avg_recon = total_recon / len(dataloader)
                avg_anomaly = total_anomaly / len(dataloader)
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Recon: {avg_recon:.4f}, "
                    f"Anomaly: {avg_anomaly:.4f}"
                )

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

        self.student_.eval()

        X_tensor = self._preprocess(X)
        scores = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size].to(self.device)

                # Extract teacher features
                teacher_features = self.teacher_(batch)

                # Student prediction
                student_features, anomaly_map = self.student_(teacher_features)

                # Reconstruction error
                recon_error = ((student_features - teacher_features) ** 2).mean(dim=[1, 2, 3])

                # Anomaly score (predicted by student)
                anomaly_score = anomaly_map.mean(dim=[1, 2, 3])

                # Combined score
                combined = recon_error + anomaly_score
                scores.append(combined.cpu().numpy())

        return np.concatenate(scores)

    def decision_function(self, X: NDArray, batch_size: Optional[int] = None) -> NDArray:
        """Alias for predict."""
        if batch_size is None:
            return self.predict(X)

        batch_size_int = int(batch_size)
        if batch_size_int <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")

        old_batch_size = self.batch_size
        try:
            self.batch_size = batch_size_int
            return self.predict(X)
        finally:
            self.batch_size = old_batch_size

    def get_anomaly_map(self, X: NDArray) -> NDArray:
        """
        Get pixel-level anomaly maps.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        anomaly_maps : NDArray of shape (n_samples, height, width)
            Pixel-level anomaly scores
        """
        self.student_.eval()

        X_tensor = self._preprocess(X)
        H, W = X.shape[1:3]
        anomaly_maps = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size].to(self.device)

                # Extract features
                teacher_features = self.teacher_(batch)

                # Student prediction
                student_features, anomaly_map = self.student_(teacher_features)

                # Combine reconstruction error and predicted anomaly map
                recon_error = ((student_features - teacher_features) ** 2).mean(dim=1, keepdim=True)

                # Resize to original size
                recon_error = F.interpolate(
                    recon_error, size=(H, W), mode="bilinear", align_corners=False
                )
                anomaly_map = F.interpolate(
                    anomaly_map, size=(H, W), mode="bilinear", align_corners=False
                )

                # Combine
                combined_map = (recon_error + anomaly_map).squeeze(1)
                anomaly_maps.append(combined_map.cpu().numpy())

        return np.concatenate(anomaly_maps)
