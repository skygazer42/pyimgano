"""
DFM: Deep Feature Modeling for Anomaly Detection.

DFM uses discriminative features from pre-trained networks with Gaussian modeling
for fast and effective anomaly detection.

This is a simplified, fast implementation focusing on efficiency.
"""

import logging
from typing import Iterable, Optional

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.covariance import LedoitWolf

from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    "vision_dfm",
    tags=("vision", "deep", "dfm", "fast", "gaussian"),
    metadata={
        "description": "DFM - Fast discriminative feature modeling",
        "speed": "very-fast",
        "training": "none",
    },
)
class VisionDFM(BaseVisionDeepDetector):
    """
    DFM anomaly detector using discriminative features.

    This is a training-free method that models normal feature distributions
    using Gaussian models with Mahalanobis distance for scoring.

    Parameters
    ----------
    backbone : str, default='resnet18'
        Feature extraction backbone
    layers : list, default=['layer2', 'layer3']
        Layers to extract features from
    pretrained : bool, default=True
        Whether to load ImageNet-pretrained weights for the backbone.
    device : str, default='cpu'
        Device to run model on

    Examples
    --------
    >>> detector = VisionDFM(device='cuda')
    >>> detector.fit(train_images)  # Fast feature extraction only
    >>> scores = detector.decision_function(test_images)
    >>> labels = detector.predict(test_images)  # 0=normal, 1=anomaly

    Notes
    -----
    - No training required (training-free)
    - Very fast inference
    - Good for quick prototyping
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list = None,
        pretrained: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize DFM detector."""
        super().__init__(**kwargs)

        self.backbone_name = backbone
        self.layers = layers or ["layer2", "layer3"]
        self.pretrained = pretrained
        self.device = device

        # Build model
        self._build_model()

        # Statistics
        self.mean = None
        self.inv_cov = None

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        logger.info(
            "Initialized DFM with backbone=%s, layers=%s, device=%s",
            backbone, self.layers, device
        )

    def _build_model(self):
        """Build feature extractor."""
        if self.backbone_name == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT if self.pretrained else None
                self.model = models.resnet18(weights=weights)
            except Exception:  # pragma: no cover - fallback for older torchvision
                self.model = models.resnet18(pretrained=self.pretrained)
        elif self.backbone_name == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
                self.model = models.resnet50(weights=weights)
            except Exception:  # pragma: no cover - fallback for older torchvision
                self.model = models.resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

        # Register hooks
        self.feature_maps = {}

        def get_activation(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook

        for layer in self.layers:
            if hasattr(self.model, layer):
                getattr(self.model, layer).register_forward_hook(
                    get_activation(layer)
                )

    def _extract_features(self, image_path: str) -> NDArray:
        """Extract features from image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model(img_tensor)

        # Aggregate features
        features = []
        for layer in self.layers:
            feat = self.feature_maps[layer]

            # Global average pooling
            feat_pooled = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
            features.append(feat_pooled.squeeze().cpu().numpy())

        return np.concatenate(features)

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None) -> "VisionDFM":
        """
        Fit DFM on normal images (feature extraction only).

        Parameters
        ----------
        X : iterable of str
            Paths to normal training images
        y : array-like, optional
            Ignored

        Returns
        -------
        self : VisionDFM
        """
        logger.info("Fitting DFM detector (training-free)")

        X_list = list(X)
        if not X_list:
            raise ValueError("Training set cannot be empty")

        # Extract features
        features = []

        for idx, img_path in enumerate(X_list):
            if idx % 10 == 0:
                logger.debug("Processing image %d/%d", idx + 1, len(X_list))

            try:
                feat = self._extract_features(img_path)
                features.append(feat)
            except Exception as e:
                logger.warning("Failed to process %s: %s", img_path, e)

        if not features:
            raise ValueError("Failed to extract features from any image")

        features = np.vstack(features)
        logger.info("Extracted features: %s", features.shape)

        # Compute statistics
        self.mean = features.mean(axis=0)

        # Use Ledoit-Wolf for robust covariance estimation
        logger.debug("Computing covariance matrix")
        cov_estimator = LedoitWolf()
        cov_estimator.fit(features)

        try:
            self.inv_cov = np.linalg.inv(cov_estimator.covariance_)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using pseudo-inverse")
            self.inv_cov = np.linalg.pinv(cov_estimator.covariance_)

        # Compute training scores to establish a threshold (PyOD semantics).
        # This enables `predict()` to return binary labels consistently.
        diff = features - self.mean
        train_scores = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_cov, diff))
        self.decision_scores_ = train_scores
        self._process_decision_scores()

        logger.info("DFM fitting completed")
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
        if self.mean is None or self.inv_cov is None or not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)

    def decision_function(self, X: Iterable[str]) -> NDArray:
        """Compute anomaly scores using Mahalanobis distance."""
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_list = list(X)
        scores = np.zeros(len(X_list), dtype=np.float64)

        logger.info("Computing anomaly scores for %d images", len(X_list))

        for idx, img_path in enumerate(X_list):
            try:
                # Extract features
                feat = self._extract_features(img_path)

                # Compute Mahalanobis distance
                diff = feat - self.mean
                score = np.sqrt(diff @ self.inv_cov @ diff.T)

                scores[idx] = score

            except Exception as e:
                logger.warning("Failed to score %s: %s", img_path, e)
                scores[idx] = 0.0

        logger.debug("Scores: min=%.4f, max=%.4f", scores.min(), scores.max())
        return scores
