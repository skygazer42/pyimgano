"""
STFPM: Student-Teacher Feature Pyramid Matching for Anomaly Detection.

STFPM uses a pre-trained teacher network and trains a student network to match
the teacher's feature pyramid, enabling efficient anomaly detection.

Reference:
    Wang, G., Han, S., Ding, E., & Huang, D. (2021).
    Student-teacher feature pyramid matching for anomaly detection.
    British Machine Vision Conference (BMVC).
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


class ImagePathDataset(Dataset):
    """Dataset for loading images from paths."""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, img_path


@register_model(
    "vision_stfpm",
    tags=("vision", "deep", "stfpm", "student-teacher", "pyramid"),
    metadata={
        "description": "STFPM - Student-Teacher Feature Pyramid Matching (BMVC 2021)",
        "paper": "Student-Teacher Feature Pyramid Matching for Anomaly Detection",
        "year": 2021,
    },
)
class VisionSTFPM(BaseVisionDeepDetector):
    """
    STFPM anomaly detector using Student-Teacher paradigm.

    This implementation uses:
    - Pre-trained ResNet18 as teacher network (frozen)
    - Trainable ResNet18 as student network
    - Multi-layer feature pyramid matching
    - MSE loss for feature alignment

    Parameters
    ----------
    backbone : str, default='resnet18'
        Backbone architecture ('resnet18')
    layers : List[str], default=['layer1', 'layer2', 'layer3']
        Layers to extract features from
    pretrained_teacher : bool, default=True
        Whether to load ImageNet-pretrained weights for the teacher backbone.
    epochs : int, default=100
        Number of training epochs
    batch_size : int, default=32
        Training batch size
    lr : float, default=0.4
        Learning rate for SGD optimizer
    num_workers : int, default=0
        Number of workers for the training DataLoader.
    device : str, default='cpu'
        Device to run model on ('cpu' or 'cuda')

    Examples
    --------
    >>> detector = VisionSTFPM(epochs=100, device='cuda')
    >>> detector.fit(['normal_img1.jpg', 'normal_img2.jpg'])
    >>> scores = detector.decision_function(['test_img.jpg'])
    >>> labels = detector.predict(['test_img.jpg'])  # 0=normal, 1=anomaly
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: List[str] = None,
        pretrained_teacher: bool = True,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.4,
        num_workers: int = 0,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize STFPM detector."""
        super().__init__(**kwargs)

        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.backbone_name = backbone
        self.layers = layers or ["layer1", "layer2", "layer3"]
        self.pretrained_teacher = pretrained_teacher
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.device = device

        # Build teacher and student networks
        self._build_model()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Track training statistics
        self.mean_scores: Optional[float] = None
        self.std_scores: Optional[float] = None

        logger.info(
            "Initialized STFPM with backbone=%s, layers=%s, "
            "epochs=%d, batch_size=%d, lr=%.3f, device=%s",
            backbone, self.layers, epochs, batch_size, lr, device
        )

    def _build_model(self) -> None:
        """Build teacher and student networks."""
        if self.backbone_name == "resnet18":
            # Teacher network (frozen)
            try:
                weights = (
                    models.ResNet18_Weights.DEFAULT if self.pretrained_teacher else None
                )
                self.teacher = models.resnet18(weights=weights)
            except Exception:  # pragma: no cover - fallback for older torchvision
                self.teacher = models.resnet18(pretrained=self.pretrained_teacher)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

            # Student network (trainable)
            try:
                self.student = models.resnet18(weights=None)
            except Exception:  # pragma: no cover - fallback for older torchvision
                self.student = models.resnet18(pretrained=False)
            self.student.train()
        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                "Currently only 'resnet18' is supported"
            )

        self.teacher.to(self.device)
        self.student.to(self.device)

    def _extract_features(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract multi-layer features from model.

        Parameters
        ----------
        model : nn.Module
            ResNet model
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        features : dict
            Dictionary mapping layer names to feature tensors
        """
        features = {}

        # Initial convolution and pooling
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        # Extract features from specified layers
        if "layer1" in self.layers:
            x = model.layer1(x)
            features["layer1"] = x

        if "layer2" in self.layers:
            x = model.layer2(x)
            features["layer2"] = x

        if "layer3" in self.layers:
            x = model.layer3(x)
            features["layer3"] = x

        return features

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None) -> "VisionSTFPM":
        """
        Train student network to match teacher on normal images.

        Parameters
        ----------
        X : iterable of str
            Paths to normal (non-anomalous) training images
        y : array-like, optional
            Ignored, present for API consistency

        Returns
        -------
        self : VisionSTFPM
            Fitted detector
        """
        logger.info("Training STFPM detector on normal images")

        X_list = list(X)
        if not X_list:
            raise ValueError("Training set cannot be empty")

        # Create dataset and dataloader
        dataset = ImagePathDataset(X_list, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=str(self.device).startswith("cuda"),
        )

        # Setup optimizer
        optimizer = SGD(
            self.student.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0001
        )

        # Training loop
        self.student.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)

                # Extract features from teacher and student
                with torch.no_grad():
                    teacher_features = self._extract_features(self.teacher, images)

                student_features = self._extract_features(self.student, images)

                # Compute MSE loss for each layer
                loss = 0.0
                for layer in self.layers:
                    t_feat = teacher_features[layer]
                    s_feat = student_features[layer]

                    # Normalize features
                    t_feat = F.normalize(t_feat, dim=1)
                    s_feat = F.normalize(s_feat, dim=1)

                    # MSE loss
                    layer_loss = F.mse_loss(s_feat, t_feat)
                    loss += layer_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info("Epoch %d/%d, Loss: %.6f", epoch + 1, self.epochs, avg_loss)

        logger.info("STFPM training completed")

        # Compute normalization statistics on training set
        self._compute_normalization_stats(X_list)

        # Compute training scores to establish a threshold (PyOD semantics).
        # This enables `predict()` to return binary labels consistently.
        self.decision_scores_ = self.decision_function(X_list)
        self._process_decision_scores()

        return self

    def _compute_normalization_stats(self, X: List[str]) -> None:
        """Compute mean and std of anomaly scores on training set."""
        logger.debug("Computing normalization statistics")

        self.student.eval()
        scores = []

        with torch.no_grad():
            for img_path in X[:min(100, len(X))]:  # Use subset for efficiency
                try:
                    score = self._compute_anomaly_score(img_path)
                    scores.append(score)
                except Exception as e:
                    logger.warning("Failed to score %s: %s", img_path, e)

        if scores:
            self.mean_scores = float(np.mean(scores))
            self.std_scores = float(np.std(scores)) + 1e-8
            logger.debug(
                "Normalization stats: mean=%.4f, std=%.4f",
                self.mean_scores, self.std_scores
            )
        else:
            self.mean_scores = 0.0
            self.std_scores = 1.0

    def _compute_anomaly_score(self, image_path: str) -> float:
        """
        Compute anomaly score for a single image.

        Parameters
        ----------
        image_path : str
            Path to input image

        Returns
        -------
        score : float
            Anomaly score (higher = more anomalous)
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            teacher_features = self._extract_features(self.teacher, img_tensor)
            student_features = self._extract_features(self.student, img_tensor)

        # Compute feature differences
        total_score = 0.0

        for layer in self.layers:
            t_feat = teacher_features[layer]
            s_feat = student_features[layer]

            # Normalize
            t_feat = F.normalize(t_feat, dim=1)
            s_feat = F.normalize(s_feat, dim=1)

            # MSE at each spatial location
            diff = (t_feat - s_feat) ** 2
            diff = diff.sum(dim=1)  # Sum over channels

            # Max pooling to get image-level score
            layer_score = diff.max().item()
            total_score += layer_score

        return total_score

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
        if self.mean_scores is None or not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)

    def decision_function(self, X: Iterable[str]) -> NDArray:
        """
        Compute anomaly scores for test images.

        Parameters
        ----------
        X : iterable of str
            Paths to test images

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (higher = more anomalous)
        """
        if self.mean_scores is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.student.eval()

        X_list = list(X)
        scores = np.zeros(len(X_list), dtype=np.float64)

        logger.info("Computing anomaly scores for %d images", len(X_list))

        for idx, image_path in enumerate(X_list):
            try:
                score = self._compute_anomaly_score(image_path)

                # Normalize score
                score = (score - self.mean_scores) / self.std_scores
                scores[idx] = score

            except Exception as e:
                logger.warning("Failed to score %s: %s", image_path, e)
                scores[idx] = 0.0

        logger.debug("Anomaly scores: min=%.4f, max=%.4f", scores.min(), scores.max())
        return scores

    def get_anomaly_map(self, image_path: str) -> NDArray:
        """
        Generate pixel-level anomaly heatmap.

        Parameters
        ----------
        image_path : str
            Path to input image

        Returns
        -------
        anomaly_map : ndarray of shape (H, W)
            Anomaly heatmap (higher values = more anomalous)
        """
        if self.mean_scores is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_size = (img.shape[1], img.shape[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract features
        self.student.eval()
        with torch.no_grad():
            teacher_features = self._extract_features(self.teacher, img_tensor)
            student_features = self._extract_features(self.student, img_tensor)

        # Compute spatial anomaly maps
        anomaly_maps = []

        for layer in self.layers:
            t_feat = teacher_features[layer]
            s_feat = student_features[layer]

            # Normalize
            t_feat = F.normalize(t_feat, dim=1)
            s_feat = F.normalize(s_feat, dim=1)

            # Compute differences
            diff = (t_feat - s_feat) ** 2
            diff = diff.sum(dim=1, keepdim=True)  # (1, 1, H, W)

            # Upsample to common size
            diff = F.interpolate(
                diff,
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            )

            anomaly_maps.append(diff)

        # Aggregate multi-scale maps
        anomaly_map = torch.mean(torch.stack(anomaly_maps), dim=0)
        anomaly_map = anomaly_map.squeeze().cpu().numpy()

        # Resize to original image size
        anomaly_map = cv2.resize(
            anomaly_map,
            original_size,
            interpolation=cv2.INTER_CUBIC
        )

        return anomaly_map

    def predict_anomaly_map(self, X: Iterable[str]) -> NDArray:
        """Generate pixel-level anomaly maps for a batch of images."""
        maps = [self.get_anomaly_map(path) for path in X]
        return np.stack(maps)
