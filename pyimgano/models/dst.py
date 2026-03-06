"""
DST - Double Student-Teacher Network for Anomaly Detection

Reference:
    "Double Student-Teacher Network for Industrial Anomaly Detection"

Uses two student networks learning from a single teacher, providing
complementary anomaly detection capabilities through different perspectives.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class TeacherNetwork(nn.Module):
    """Pre-trained teacher network for feature extraction."""

    def __init__(self, backbone: str = "wide_resnet50"):
        super().__init__()

        if backbone == "wide_resnet50":
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            resnet = wide_resnet50_2(weights=weights)
        elif backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract multiple layers
        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        return features


class StudentNetwork(nn.Module):
    """Student network for knowledge distillation."""

    def __init__(self, in_channels: List[int], out_channels: List[int], use_attention: bool = True):
        super().__init__()

        self.decoders = nn.ModuleList()
        self.use_attention = use_attention

        for in_ch, out_ch in zip(in_channels, out_channels):
            decoder = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
            )
            self.decoders.append(decoder)

        # Attention modules
        if use_attention:
            self.attention = nn.ModuleList()
            for out_ch in out_channels:
                attention = nn.Sequential(
                    nn.Conv2d(out_ch, out_ch // 4, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch // 4, out_ch, kernel_size=1),
                    nn.Sigmoid(),
                )
                self.attention.append(attention)

    def forward(self, teacher_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Predict teacher features."""
        student_features = []

        for i, (feat, decoder) in enumerate(zip(teacher_features, self.decoders)):
            pred = decoder(feat)

            # Apply attention
            if self.use_attention:
                att = self.attention[i](pred)
                pred = pred * att

            student_features.append(pred)

        return student_features


@register_model(
    "vision_dst",
    tags=("vision", "deep", "dst", "student-teacher", "distillation", "sota"),
    metadata={
        "description": "DST - Double Student-Teacher with complementary learning",
        "paper": "Double Student-Teacher Network for Anomaly Detection",
        "year": 2023,
        "type": "knowledge-distillation",
    },
)
class VisionDST(BaseVisionDeepDetector):
    """
    DST: Double Student-Teacher Network for Anomaly Detection.

    Uses two student networks with different architectures learning from
    a single teacher, providing complementary anomaly detection.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Teacher network backbone
    learning_rate : float, default=1e-4
        Learning rate for training
    batch_size : int, default=16
        Batch size for training
    epochs : int, default=60
        Number of training epochs
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    teacher_ : TeacherNetwork
        Pre-trained teacher network
    student1_ : StudentNetwork
        First student network (with attention)
    student2_ : StudentNetwork
        Second student network (without attention)

    Examples
    --------
    >>> from pyimgano.models import VisionDST
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    >>> X_test = np.random.rand(20, 224, 224, 3).astype(np.float32)
    >>>
    >>> # Create and train detector
    >>> detector = VisionDST(epochs=30)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 60,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.teacher_ = None
        self.student1_ = None
        self.student2_ = None

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

    def _distillation_loss(
        self, student_features: List[torch.Tensor], teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.

        Parameters
        ----------
        student_features : List[torch.Tensor]
            Predicted features from student
        teacher_features : List[torch.Tensor]
            Ground truth features from teacher

        Returns
        -------
        loss : torch.Tensor
            Distillation loss
        """
        total_loss = 0.0

        for s_feat, t_feat in zip(student_features, teacher_features):
            # Resize student to match teacher if needed
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(
                    s_feat, size=t_feat.shape[2:], mode="bilinear", align_corners=False
                )

            # MSE loss
            mse = F.mse_loss(s_feat, t_feat)

            # Cosine similarity loss
            s_flat = s_feat.view(s_feat.size(0), s_feat.size(1), -1)
            t_flat = t_feat.view(t_feat.size(0), t_feat.size(1), -1)

            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=2).mean()
            cos_loss = 1 - cos_sim

            # Combined loss
            total_loss += mse + 0.1 * cos_loss

        return total_loss / len(student_features)

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "VisionDST":
        """
        Fit the DST detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionDST
            Fitted detector
        """
        # Preprocess
        X_tensor = self._preprocess(X)

        # Initialize teacher
        if self.teacher_ is None:
            self.teacher_ = TeacherNetwork(backbone=self.backbone).to(self.device)

        # Get feature dimensions
        with torch.no_grad():
            sample_features = self.teacher_(X_tensor[:1].to(self.device))
            in_channels = [f.shape[1] for f in sample_features]

        # Initialize students
        if self.student1_ is None:
            # Student 1: with attention mechanism
            self.student1_ = StudentNetwork(
                in_channels=in_channels, out_channels=in_channels, use_attention=True
            ).to(self.device)

        if self.student2_ is None:
            # Student 2: without attention mechanism
            self.student2_ = StudentNetwork(
                in_channels=in_channels, out_channels=in_channels, use_attention=False
            ).to(self.device)

        # Training
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Optimize both students
        optimizer = torch.optim.Adam(
            list(self.student1_.parameters()) + list(self.student2_.parameters()),
            lr=self.learning_rate,
        )

        self.student1_.train()
        self.student2_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_loss1 = 0.0
            total_loss2 = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Extract teacher features
                with torch.no_grad():
                    teacher_features = self.teacher_(batch_images)

                # Student 1 prediction
                student1_features = self.student1_(teacher_features)
                loss1 = self._distillation_loss(student1_features, teacher_features)

                # Student 2 prediction
                student2_features = self.student2_(teacher_features)
                loss2 = self._distillation_loss(student2_features, teacher_features)

                # Combined loss
                loss = loss1 + loss2

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                avg_loss1 = total_loss1 / len(dataloader)
                avg_loss2 = total_loss2 / len(dataloader)
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"S1: {avg_loss1:.4f}, "
                    f"S2: {avg_loss2:.4f}"
                )

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
            Anomaly scores (combined from both students)
        """
        self.student1_.eval()
        self.student2_.eval()

        X_tensor = self._preprocess(X)
        scores = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size].to(self.device)

                # Extract teacher features
                teacher_features = self.teacher_(batch)

                # Student predictions
                student1_features = self.student1_(teacher_features)
                student2_features = self.student2_(teacher_features)

                # Compute errors for both students
                error1 = 0.0
                error2 = 0.0

                for s1, s2, t in zip(student1_features, student2_features, teacher_features):
                    # Resize if needed
                    if s1.shape != t.shape:
                        s1 = F.interpolate(
                            s1, size=t.shape[2:], mode="bilinear", align_corners=False
                        )
                    if s2.shape != t.shape:
                        s2 = F.interpolate(
                            s2, size=t.shape[2:], mode="bilinear", align_corners=False
                        )

                    error1 += ((s1 - t) ** 2).mean(dim=[1, 2, 3])
                    error2 += ((s2 - t) ** 2).mean(dim=[1, 2, 3])

                # Combine scores from both students (max provides robustness)
                combined_score = torch.maximum(error1, error2)
                scores.append(combined_score.cpu().numpy())

        return np.concatenate(scores)

    def decision_function(self, X: NDArray) -> NDArray:
        """Alias for predict."""
        return self.predict(X)
