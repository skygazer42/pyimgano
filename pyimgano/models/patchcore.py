"""
PatchCore: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022).

PatchCore achieves state-of-the-art performance on MVTec AD benchmark using
locally aware patch-level representations and coreset selection.

Reference:
    Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022).
    Towards total recall in industrial anomaly detection.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14318-14328).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple, Union

from pyimgano.utils.optional_deps import require

from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:  # pragma: no cover - typing-only dependency
    from numpy.typing import NDArray
except Exception:  # pragma: no cover - minimal env without numpy
    NDArray = Any  # type: ignore[misc,assignment]

ImageInput = Union[str, NDArray]


@register_model(
    "vision_patchcore",
    tags=("vision", "deep", "patchcore", "sota", "cvpr2022", "numpy", "pixel_map"),
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
        feature_projection_dim: Optional[int] = None,
        projection_fit_samples: int = 10,
        n_neighbors: int = 9,
        knn_backend: str = "sklearn",
        memory_bank_dtype: str = "float32",
        random_seed: int = 0,
        pretrained: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize PatchCore detector."""
        super().__init__(**kwargs)

        self._np = require("numpy", purpose="PatchCore feature processing")
        self._cv2 = require("cv2", purpose="PatchCore image loading and resizing")
        self._torch = require("torch", purpose="PatchCore backbone inference")
        self._F = require("torch.nn.functional", purpose="PatchCore feature resizing")
        self._tv_models = require("torchvision.models", purpose="PatchCore backbone models")
        self._tv_transforms = require(
            "torchvision.transforms", purpose="PatchCore preprocessing transforms"
        )

        if not 0.0 < coreset_sampling_ratio <= 1.0:
            raise ValueError(
                f"coreset_sampling_ratio must be in (0.0, 1.0], got {coreset_sampling_ratio}"
            )

        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")

        self.backbone_name = backbone
        self.layers = layers or ["layer2", "layer3"]
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.feature_projection_dim = (
            int(feature_projection_dim) if feature_projection_dim is not None else None
        )
        self.projection_fit_samples = int(projection_fit_samples)
        self.n_neighbors = n_neighbors
        self.knn_backend = knn_backend
        self.memory_bank_dtype = str(memory_bank_dtype)
        self.random_seed = int(random_seed)
        self.pretrained = pretrained
        self.device = device

        if self.feature_projection_dim is not None and self.feature_projection_dim < 1:
            raise ValueError(
                f"feature_projection_dim must be >= 1 (or None), got {self.feature_projection_dim}"
            )
        if self.projection_fit_samples < 1:
            raise ValueError(
                f"projection_fit_samples must be >= 1, got {self.projection_fit_samples}"
            )
        if self.memory_bank_dtype not in ("float32", "float16"):
            raise ValueError(
                "memory_bank_dtype must be 'float32' or 'float16'. "
                f"Got {self.memory_bank_dtype!r}."
            )

        # Initialize backbone
        self._build_model()

        # Memory bank for patch features
        self.memory_bank: Optional[NDArray] = None
        self.nn_index = None
        self._projection = None

        # Image preprocessing
        transforms = self._tv_transforms
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        logger.info(
            "Initialized PatchCore with backbone=%s, layers=%s, "
            "coreset_ratio=%.2f, k=%d, device=%s, proj_dim=%s",
            backbone,
            self.layers,
            coreset_sampling_ratio,
            n_neighbors,
            device,
            str(self.feature_projection_dim),
        )

    def _build_model(self) -> None:
        """Build feature extraction backbone."""
        models = self._tv_models
        # TorchVision changed API from `pretrained=True` to `weights=...`.
        # Keep backward compatibility with older torchvision versions.
        if self.backbone_name == "wide_resnet50":
            try:
                weights = models.Wide_ResNet50_2_Weights.DEFAULT if self.pretrained else None
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
        self.feature_maps = {}

        def get_activation(name: str):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()

            return hook

        for layer in self.layers:
            if not hasattr(self.model, layer):
                raise ValueError(f"Model has no layer named '{layer}'")
            getattr(self.model, layer).register_forward_hook(get_activation(layer))

    def _load_image_rgb(self, image: ImageInput) -> NDArray:
        np = self._np
        cv2 = self._cv2

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB image, got dtype={image.dtype}")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected shape (H,W,3), got {image.shape}")
            return np.ascontiguousarray(image)

        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Failed to load image: {image}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _extract_patch_features(self, image: ImageInput) -> Tuple[NDArray, Tuple[int, int]]:
        """
        Extract patch-level features from an image.

        Parameters
        ----------
        image : str | np.ndarray
            Path to input image, or a canonical RGB/u8/HWC numpy image.

        Returns
        -------
        features : ndarray of shape (n_patches, feature_dim)
            Extracted patch features
        """
        cv2 = self._cv2
        torch = self._torch
        F = self._F

        # Load and preprocess image
        img = self._load_image_rgb(image)
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
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)

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

    def _ensure_projection(self, features_fit: NDArray) -> None:
        """Create and fit a random projection for PatchCore features if enabled."""

        if self.feature_projection_dim is None:
            return
        if self._projection is not None:
            return

        np = self._np
        from sklearn.random_projection import GaussianRandomProjection

        fit_mat = np.asarray(features_fit, dtype=np.float32)
        if fit_mat.ndim != 2:
            raise ValueError(f"Expected 2D fit matrix, got shape {fit_mat.shape}")

        dim = int(fit_mat.shape[1])
        if self.feature_projection_dim >= dim:
            # Nothing to do: requested dim is not a reduction.
            self._projection = None
            return

        self._projection = GaussianRandomProjection(
            n_components=int(self.feature_projection_dim),
            random_state=int(self.random_seed),
        )
        self._projection.fit(fit_mat)

    def _maybe_project(self, features: NDArray) -> NDArray:
        if self._projection is None:
            return features
        np = self._np
        projected = self._projection.transform(features)
        return np.asarray(projected, dtype=np.float32)

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
        np = self._np

        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * self.coreset_sampling_ratio))

        if n_coreset >= n_samples:
            logger.debug("Coreset size >= total samples, using all features")
            return features

        logger.debug(
            "Performing coreset sampling: %d -> %d samples (%.1f%%)",
            n_samples,
            n_coreset,
            100 * self.coreset_sampling_ratio,
        )

        # Greedy k-Center coreset selection.
        #
        # Implementation notes:
        # - Use squared L2 distances to avoid an unnecessary sqrt (monotonic).
        # - Use an explicit RNG for reproducibility across runs.
        rng = np.random.default_rng(int(self.random_seed))

        selected_indices: list[int] = []
        min_distances_sq = np.full(n_samples, np.inf, dtype=np.float64)
        selected_indices.append(int(rng.integers(0, n_samples)))

        for _ in range(int(n_coreset) - 1):
            last_selected = features[selected_indices[-1]]
            diff = features - last_selected
            distances_sq = np.sum(diff * diff, axis=1)
            min_distances_sq = np.minimum(min_distances_sq, distances_sq)

            next_idx = int(np.argmax(min_distances_sq))
            selected_indices.append(next_idx)

        return features[selected_indices]

    def fit(self, X: Iterable[ImageInput], y: Optional[NDArray] = None) -> "VisionPatchCore":
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
        np = self._np
        from .knn_index import build_knn_index

        logger.info("Fitting PatchCore detector on training images")

        X_list = list(X)
        if not X_list:
            raise ValueError("Training set cannot be empty")

        # Optional: fit a projection on a small subset of training patches.
        if self.feature_projection_dim is not None:
            fit_patches: list[NDArray] = []
            for image in X_list[: min(int(self.projection_fit_samples), len(X_list))]:
                feat, _ = self._extract_patch_features(image)
                fit_patches.append(np.asarray(feat, dtype=np.float32))
            if fit_patches:
                self._ensure_projection(np.vstack(fit_patches))

        # Extract features from all training images.
        all_features: list[NDArray] = []

        for idx, image in enumerate(X_list):
            if idx % 10 == 0:
                logger.debug("Processing image %d/%d", idx + 1, len(X_list))

            try:
                features, _ = self._extract_patch_features(image)
                features = self._maybe_project(features)
                all_features.append(np.asarray(features, dtype=np.float32))
            except Exception as e:
                logger.warning("Failed to process item %d: %s", idx, e)
                continue

        if not all_features:
            raise ValueError("Failed to extract features from any training image")

        # Stack all features
        all_features = np.vstack(all_features)
        logger.info(
            "Extracted %d patch features (dim=%d)", all_features.shape[0], all_features.shape[1]
        )

        # Perform coreset sampling
        sampled = self._coreset_sampling(all_features)
        if self.memory_bank_dtype == "float16":
            sampled = np.asarray(sampled, dtype=np.float16)
        else:
            sampled = np.asarray(sampled, dtype=np.float32)

        self.memory_bank = sampled
        logger.info(
            "Memory bank created: %d patches (%.2f%% of original)",
            self.memory_bank.shape[0],
            100 * self.memory_bank.shape[0] / all_features.shape[0],
        )

        # Build k-NN index
        self.nn_index = build_knn_index(
            backend=self.knn_backend,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            n_jobs=-1,
        )
        self.nn_index.fit(np.asarray(self.memory_bank, dtype=np.float32))

        # Compute training scores to establish a threshold (PyOD semantics).
        # This enables `predict()` to return binary labels consistently.
        self.decision_scores_ = self.decision_function(X_list)
        self._process_decision_scores()

        logger.info("PatchCore training completed")
        return self

    def predict(self, X: Iterable[ImageInput]) -> NDArray:
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

    def decision_function(self, X: Iterable[ImageInput]) -> NDArray:
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
        np = self._np
        if self.memory_bank is None or self.nn_index is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_list = list(X)
        scores = np.zeros(len(X_list))

        logger.info("Computing anomaly scores for %d images", len(X_list))

        for idx, image in enumerate(X_list):
            try:
                # Extract patch features
                features, _ = self._extract_patch_features(image)
                features = self._maybe_project(features)

                # Find k nearest neighbors in memory bank
                distances, _ = self.nn_index.kneighbors(features)

                # Aggregate patch-level scores to image-level score
                # Use max distance to k-th nearest neighbor
                patch_scores = distances[:, -1]  # Distance to k-th neighbor
                image_score = np.max(patch_scores)  # Max patch score

                scores[idx] = image_score

            except Exception as e:
                logger.warning("Failed to score item %d: %s", idx, e)
                scores[idx] = 0.0

        logger.debug("Anomaly scores: min=%.4f, max=%.4f", scores.min(), scores.max())
        return scores

    def get_anomaly_map(self, image_path: ImageInput) -> NDArray:
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
        np = self._np
        cv2 = self._cv2
        if self.memory_bank is None or self.nn_index is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract patch features
        features, (h, w) = self._extract_patch_features(image_path)
        features = self._maybe_project(features)

        # Compute patch-level anomaly scores
        distances, _ = self.nn_index.kneighbors(features)
        patch_scores = distances[:, -1]

        # Reshape to spatial dimensions
        expected = int(h * w)
        if patch_scores.shape[0] != expected:
            side = int(np.sqrt(len(patch_scores)))
            h = w = side
        anomaly_map = patch_scores.reshape(int(h), int(w))

        # Resize to original image size (if known)
        if isinstance(image_path, np.ndarray):
            original_h, original_w = int(image_path.shape[0]), int(image_path.shape[1])
            anomaly_map = cv2.resize(
                anomaly_map,
                (original_w, original_h),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            img = cv2.imread(str(image_path))
            if img is not None:
                anomaly_map = cv2.resize(
                    anomaly_map,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

        return np.asarray(anomaly_map, dtype=np.float32)
