"""
PatchCore: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022).

PatchCore achieves state-of-the-art performance on MVTec AD benchmark using
locally aware patch-level representations and coreset selection.

Reference:
    Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022).
    Towards total recall in industrial anomaly detection.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14318-14328).
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from .baseCv import BaseVisionDeepDetector
from .knn_index import KNNIndex, build_knn_index
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    "vision_patchcore",
    tags=("vision", "deep", "patchcore", "sota", "cvpr2022"),
    metadata={
        "description": "PatchCore - SOTA patch-level anomaly detection (CVPR 2022)",
        "paper": "Towards Total Recall in Industrial Anomaly Detection",
        "benchmark_rank": "state-of-the-art",
        "year": 2022,
    },
)
class VisionPatchCore(BaseVisionDeepDetector):
    """
    PatchCore anomaly detector using WideResNet50 backbone.

    This implementation uses:
    - Pre-trained WideResNet50 for feature extraction
    - Locally aware patch features from multiple layers
    - Coreset subsampling for efficient memory bank
    - k-NN based anomaly scoring

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone ('wide_resnet50' or 'resnet50')
    layers : List[str], default=['layer2', 'layer3']
        Layers to extract features from
    coreset_sampling_ratio : float, default=0.1
        Ratio of training patches to keep in memory bank (0.0-1.0)
    n_neighbors : int, default=9
        Number of nearest neighbors for anomaly scoring
    device : str, default='cpu'
        Device to run model on ('cpu' or 'cuda')

    Examples
    --------
    >>> detector = VisionPatchCore(coreset_sampling_ratio=0.1, device='cuda')
    >>> detector.fit(['normal_img1.jpg', 'normal_img2.jpg'])
    >>> scores = detector.decision_function(['test_img.jpg'])
    >>> labels = detector.predict(['test_img.jpg'])  # 0=normal, 1=anomaly
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        layers: List[str] = None,
        coreset_sampling_ratio: float = 0.1,
        n_neighbors: int = 9,
        knn_backend: str = "sklearn",
        pretrained: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize PatchCore detector."""
        super().__init__(**kwargs)

        if not 0.0 < coreset_sampling_ratio <= 1.0:
            raise ValueError(
                f"coreset_sampling_ratio must be in (0.0, 1.0], got {coreset_sampling_ratio}"
            )

        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")

        self.backbone_name = backbone
        self.layers = layers or ["layer2", "layer3"]
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.n_neighbors = n_neighbors
        self.knn_backend = knn_backend
        self.pretrained = pretrained
        self.device = device

        # Initialize backbone
        self._build_model()

        # Memory bank for patch features
        self.memory_bank: Optional[NDArray] = None
        self.nn_index: Optional[KNNIndex] = None

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
            "Initialized PatchCore with backbone=%s, layers=%s, "
            "coreset_ratio=%.2f, k=%d, device=%s",
            backbone, self.layers, coreset_sampling_ratio, n_neighbors, device
        )

    def _build_model(self) -> None:
        """Build feature extraction backbone."""
        # TorchVision changed API from `pretrained=True` to `weights=...`.
        # Keep backward compatibility with older torchvision versions.
        if self.backbone_name == "wide_resnet50":
            try:
                weights = (
                    models.Wide_ResNet50_2_Weights.DEFAULT if self.pretrained else None
                )
                self.model = models.wide_resnet50_2(weights=weights)
            except Exception:  # pragma: no cover - fallback for older torchvision
                self.model = models.wide_resnet50_2(pretrained=self.pretrained)
        elif self.backbone_name == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
                self.model = models.resnet50(weights=weights)
            except Exception:  # pragma: no cover - fallback for older torchvision
                self.model = models.resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                "Choose 'wide_resnet50' or 'resnet50'"
            )

        self.model.eval()
        self.model.to(self.device)

        # Register hooks for feature extraction
        self.feature_maps: Dict[str, torch.Tensor] = {}

        def get_activation(name: str):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook

        for layer in self.layers:
            if not hasattr(self.model, layer):
                raise ValueError(f"Model has no layer named '{layer}'")
            getattr(self.model, layer).register_forward_hook(
                get_activation(layer)
            )

    def _extract_patch_features(self, image_path: str) -> Tuple[NDArray, Tuple[int, int]]:
        """
        Extract patch-level features from an image.

        Parameters
        ----------
        image_path : str
            Path to input image

        Returns
        -------
        features : ndarray of shape (n_patches, feature_dim)
            Extracted patch features
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            _ = self.model(img_tensor)

        # Aggregate multi-scale features
        features_list = []
        target_size: Optional[Tuple[int, int]] = None

        for layer in self.layers:
            feat = self.feature_maps[layer]  # (1, C, H, W)

            # Resize to common spatial size
            if target_size is None:
                target_size = (int(feat.shape[-2]), int(feat.shape[-1]))
            else:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            features_list.append(feat)

        # Concatenate features from different layers
        features = torch.cat(features_list, dim=1)  # (1, C_total, H, W)

        # Reshape to patch features: (H*W, C)
        features = features.squeeze(0)  # (C, H, W)
        features = features.permute(1, 2, 0)  # (H, W, C)
        features = features.reshape(-1, features.shape[-1])  # (H*W, C)

        if target_size is None:
            raise RuntimeError("Failed to infer PatchCore feature map spatial size")

        return features.cpu().numpy(), target_size

    def _coreset_sampling(self, features: NDArray) -> NDArray:
        """
        Perform greedy coreset selection to reduce memory bank size.

        Uses sparse greedy k-Center selection for efficient sampling.

        Parameters
        ----------
        features : ndarray of shape (n_samples, feature_dim)
            Input features

        Returns
        -------
        coreset : ndarray of shape (n_coreset, feature_dim)
            Selected coreset
        """
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * self.coreset_sampling_ratio))

        if n_coreset >= n_samples:
            logger.debug("Coreset size >= total samples, using all features")
            return features

        logger.debug(
            "Performing coreset sampling: %d -> %d samples (%.1f%%)",
            n_samples, n_coreset, 100 * self.coreset_sampling_ratio
        )

        # Greedy k-Center coreset selection
        selected_indices = []

        # Start with random point
        min_distances = np.full(n_samples, np.inf)
        selected_indices.append(np.random.randint(n_samples))

        for _ in range(n_coreset - 1):
            # Update minimum distances to selected set
            last_selected = features[selected_indices[-1]]
            distances = np.linalg.norm(
                features - last_selected,
                axis=1
            )
            min_distances = np.minimum(min_distances, distances)

            # Select point with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)

        return features[selected_indices]

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None) -> "VisionPatchCore":
        """
        Fit PatchCore on normal training images.

        Parameters
        ----------
        X : iterable of str
            Paths to normal (non-anomalous) training images
        y : array-like, optional
            Ignored, present for API consistency

        Returns
        -------
        self : VisionPatchCore
            Fitted detector
        """
        logger.info("Fitting PatchCore detector on training images")

        X_list = list(X)
        if not X_list:
            raise ValueError("Training set cannot be empty")

        # Extract features from all training images
        all_features = []

        for idx, image_path in enumerate(X_list):
            if idx % 10 == 0:
                logger.debug("Processing image %d/%d", idx + 1, len(X_list))

            try:
                features, _ = self._extract_patch_features(image_path)
                all_features.append(features)
            except Exception as e:
                logger.warning("Failed to process %s: %s", image_path, e)
                continue

        if not all_features:
            raise ValueError("Failed to extract features from any training image")

        # Stack all features
        all_features = np.vstack(all_features)
        logger.info(
            "Extracted %d patch features (dim=%d)",
            all_features.shape[0], all_features.shape[1]
        )

        # Perform coreset sampling
        self.memory_bank = self._coreset_sampling(all_features)
        logger.info(
            "Memory bank created: %d patches (%.2f%% of original)",
            self.memory_bank.shape[0],
            100 * self.memory_bank.shape[0] / all_features.shape[0]
        )

        # Build k-NN index
        self.nn_index = build_knn_index(
            backend=self.knn_backend,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            n_jobs=-1,
        )
        self.nn_index.fit(self.memory_bank)

        # Compute training scores to establish a threshold (PyOD semantics).
        # This enables `predict()` to return binary labels consistently.
        self.decision_scores_ = self.decision_function(X_list)
        self._process_decision_scores()

        logger.info("PatchCore training completed")
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
        if self.memory_bank is None or self.nn_index is None or not hasattr(self, "threshold_"):
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
        if self.memory_bank is None or self.nn_index is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_list = list(X)
        scores = np.zeros(len(X_list))

        logger.info("Computing anomaly scores for %d images", len(X_list))

        for idx, image_path in enumerate(X_list):
            try:
                # Extract patch features
                features, _ = self._extract_patch_features(image_path)

                # Find k nearest neighbors in memory bank
                distances, _ = self.nn_index.kneighbors(features)

                # Aggregate patch-level scores to image-level score
                # Use max distance to k-th nearest neighbor
                patch_scores = distances[:, -1]  # Distance to k-th neighbor
                image_score = np.max(patch_scores)  # Max patch score

                scores[idx] = image_score

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
        if self.memory_bank is None or self.nn_index is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract patch features
        features, (h, w) = self._extract_patch_features(image_path)

        # Compute patch-level anomaly scores
        distances, _ = self.nn_index.kneighbors(features)
        patch_scores = distances[:, -1]

        # Reshape to spatial dimensions
        expected = int(h * w)
        if patch_scores.shape[0] != expected:
            side = int(np.sqrt(len(patch_scores)))
            h = w = side
        anomaly_map = patch_scores.reshape(int(h), int(w))

        # Resize to original image size
        img = cv2.imread(image_path)
        if img is not None:
            anomaly_map = cv2.resize(
                anomaly_map,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )

        return anomaly_map
