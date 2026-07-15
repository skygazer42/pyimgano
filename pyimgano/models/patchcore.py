"""Core-aligned PatchCore implementation.

The CVPR 2022 method combines locally aware patch representations, a coreset
memory bank, nearest-neighbor scoring, and image-level score reweighting. This
module implements those defining components; it does not claim the paper's
published benchmark numbers. Its patchification and 1024-to-1024 multi-layer
embedding follow the authors' reference implementation.

Reference:
    Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022).
    Towards total recall in industrial anomaly detection.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14318-14328).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union, cast

from pyimgano.utils.optional_deps import require
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .deep_io import safe_torch_load
from .knn_index import build_knn_index
from .patchknn_core import approximate_greedy_coreset_indices
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."

logger = logging.getLogger(__name__)

try:  # pragma: no cover - typing-only dependency
    from numpy.typing import NDArray
except Exception:  # pragma: no cover - minimal env without numpy
    NDArray = Any  # type: ignore[misc,assignment]

ImageInput = Union[str, NDArray]


@register_model(
    "vision_patchcore",
    tags=("vision", "deep", "patchcore", "memory_bank", "cvpr2022", "numpy", "pixel_map"),
    metadata={
        "description": "PatchCore core algorithm with local patch aggregation and coreset memory",
        "paper": "Towards Total Recall in Industrial Anomaly Detection",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html",
        "year": 2022,
        "supervision": "one-class",
        "implementation_status": "core-aligned",
        "paper_fidelity": "core-aligned",
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
    backbone : str, default='wide_resnet50_2'
        Feature extraction backbone ('wide_resnet50_2' or 'resnet50')
    layers : List[str], default=['layer2', 'layer3']
        Layers to extract features from
    coreset_sampling_ratio : float, default=0.1
        Ratio of training patches to keep in memory bank (0.0-1.0)
    n_neighbors : int, default=1
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
        backbone: str = "wide_resnet50_2",
        layers: List[str] = None,
        coreset_sampling_ratio: float = 0.1,
        pretrain_embed_dimension: int = 1024,
        target_embed_dimension: int = 1024,
        patch_size: int = 3,
        patch_stride: int = 1,
        coreset_projection_dim: Optional[int] = 128,
        coreset_starting_points: int = 10,
        feature_projection_dim: Optional[int] = None,
        projection_fit_samples: int = 10,
        n_neighbors: int = 1,
        knn_backend: str = "sklearn",
        memory_bank_dtype: str = "float32",
        gaussian_sigma: float = 4.0,
        random_seed: int = 0,
        pretrained: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize PatchCore detector."""
        super().__init__(**kwargs)

        self._np = require("numpy", purpose="PatchCore feature processing")
        self._cv2 = require("cv2", purpose="PatchCore image loading and resizing")
        self._torch = require("torch", purpose="PatchCore backbone inference")
        self._F = require("torch.nn.functional", purpose="PatchCore feature resizing")
        self._tv_transforms = require(
            "torchvision.transforms", purpose="PatchCore preprocessing transforms"
        )

        if not 0.0 < coreset_sampling_ratio <= 1.0:
            raise ValueError(
                f"coreset_sampling_ratio must be in (0.0, 1.0], got {coreset_sampling_ratio}"
            )

        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        if pretrain_embed_dimension < 1 or target_embed_dimension < 1:
            raise ValueError("PatchCore embedding dimensions must be positive")
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}")
        if patch_stride < 1:
            raise ValueError(f"patch_stride must be >= 1, got {patch_stride}")
        if coreset_projection_dim is not None and int(coreset_projection_dim) < 1:
            raise ValueError(
                "coreset_projection_dim must be >= 1 (or None), "
                f"got {coreset_projection_dim}"
            )
        if coreset_starting_points < 1:
            raise ValueError(
                f"coreset_starting_points must be >= 1, got {coreset_starting_points}"
            )

        self.backbone_name = backbone
        self.layers = layers or ["layer2", "layer3"]
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.pretrain_embed_dimension = int(pretrain_embed_dimension)
        self.target_embed_dimension = int(target_embed_dimension)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.coreset_projection_dim = (
            int(coreset_projection_dim) if coreset_projection_dim is not None else None
        )
        self.coreset_starting_points = int(coreset_starting_points)
        self.feature_projection_dim = (
            int(feature_projection_dim) if feature_projection_dim is not None else None
        )
        self.projection_fit_samples = int(projection_fit_samples)
        self.n_neighbors = n_neighbors
        self.knn_backend = knn_backend
        self.memory_bank_dtype = str(memory_bank_dtype)
        self.gaussian_sigma = float(gaussian_sigma)
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
        if self.gaussian_sigma < 0:
            raise ValueError(f"gaussian_sigma must be >= 0, got {self.gaussian_sigma}")

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
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        logger.info(
            "Initialized PatchCore with backbone=%s, layers=%s, "
            "coreset_ratio=%.2f, embed=%d->%d, patch=%d/%d, k=%d, device=%s, proj_dim=%s",
            backbone,
            self.layers,
            coreset_sampling_ratio,
            self.pretrain_embed_dimension,
            self.target_embed_dimension,
            self.patch_size,
            self.patch_stride,
            n_neighbors,
            device,
            str(self.feature_projection_dim),
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.memory_bank is None or self.nn_index is None or not hasattr(self, "threshold_"):
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="PatchCore checkpoint saving")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        raw_state = self.model.state_dict()
        model_state_dict: dict[str, object] = {}
        for key, value in dict(raw_state).items():
            detach = getattr(value, "detach", None)
            cpu = getattr(value, "cpu", None)
            if callable(detach) and callable(cpu):
                model_state_dict[str(key)] = detach().cpu()
            else:
                model_state_dict[str(key)] = value

        projection_state: dict[str, object] | None = None
        if self._projection is not None and hasattr(self._projection, "components_"):
            projection_state = {
                "components_": self._np.asarray(
                    self._projection.components_, dtype=self._np.float32
                ),
                "n_features_in_": int(self._projection.n_features_in_),
                "n_components_": int(self._projection.components_.shape[0]),
            }

        torch.save(
            {
                "schema_version": 2,
                "embedding_contract": self._embedding_contract(),
                "model_state_dict": model_state_dict,
                "memory_bank": self._np.asarray(self.memory_bank, dtype=self._np.float32),
                "decision_scores_": self._np.asarray(self.decision_scores_, dtype=self._np.float64),
                "threshold_": float(self.threshold_),
                "n_neighbors_fit": int(
                    getattr(self, "_n_neighbors_fit", self.n_neighbors)
                ),
                "projection_state": projection_state,
                "gaussian_sigma": float(self.gaussian_sigma),
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        from sklearn.random_projection import GaussianRandomProjection

        state = safe_torch_load(Path(path), map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Invalid VisionPatchCore checkpoint payload.")
        if int(state.get("schema_version", 0)) != 2:
            raise ValueError(
                "Unsupported legacy PatchCore checkpoint: refit with the paper-aligned "
                "patch embedding and save a new checkpoint."
            )
        if state.get("embedding_contract") != self._embedding_contract():
            raise ValueError(
                "VisionPatchCore checkpoint embedding contract does not match this detector."
            )

        model_state_dict = state.get("model_state_dict", None)
        if not isinstance(model_state_dict, dict):
            raise ValueError("VisionPatchCore checkpoint is missing model_state_dict.")
        self.model.load_state_dict(dict(model_state_dict), strict=False)
        self.model.to(self.device)
        self.model.eval()

        projection_state = state.get("projection_state", None)
        self._projection = None
        if isinstance(projection_state, dict) and projection_state.get("components_") is not None:
            self._projection = GaussianRandomProjection(
                n_components=int(projection_state["n_components_"]),
                random_state=int(self.random_seed),
            )
            self._projection.components_ = self._np.asarray(
                projection_state["components_"],
                dtype=self._np.float32,
            )
            self._projection.n_features_in_ = int(projection_state["n_features_in_"])
            self._projection.n_components_ = int(projection_state["n_components_"])

        self.memory_bank = self._np.asarray(state["memory_bank"], dtype=self._np.float32)
        self._n_neighbors_fit = min(
            int(state.get("n_neighbors_fit", self.n_neighbors)),
            int(self.memory_bank.shape[0]),
        )
        self.nn_index = build_knn_index(
            backend=self.knn_backend,
            n_neighbors=self._n_neighbors_fit,
            metric="euclidean",
            n_jobs=-1,
        )
        self.nn_index.fit(self._np.asarray(self.memory_bank, dtype=self._np.float32))
        self.decision_scores_ = self._np.asarray(state["decision_scores_"], dtype=self._np.float64)
        self.threshold_ = float(state["threshold_"])
        self.gaussian_sigma = float(state.get("gaussian_sigma", self.gaussian_sigma))

    def _build_model(self) -> None:
        """Build feature extraction backbone."""
        if self.backbone_name in {"wide_resnet50", "wide_resnet50_2"}:
            self.model, _ = load_torchvision_model(
                "wide_resnet50_2",
                pretrained=bool(self.pretrained),
            )
        elif self.backbone_name == "resnet50":
            self.model, _ = load_torchvision_model(
                "resnet50",
                pretrained=bool(self.pretrained),
            )
        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                "Choose 'wide_resnet50_2' or 'resnet50'"
            )

        self.model.eval()
        self.model.to(self.device)

        # Register hooks for feature extraction
        self.feature_maps = {}

        def get_activation(name: str):
            def hook(module, input, output):
                del input, module
                self.feature_maps[name] = output.detach()

            return hook

        for layer in self.layers:
            if not hasattr(self.model, layer):
                raise ValueError(f"Model has no layer named '{layer}'")
            getattr(self.model, layer).register_forward_hook(get_activation(layer))

    def _embedding_contract(self) -> dict[str, object]:
        backbone = (
            "wide_resnet50_2"
            if self.backbone_name in {"wide_resnet50", "wide_resnet50_2"}
            else str(self.backbone_name)
        )
        return {
            "backbone": backbone,
            "layers": list(self.layers),
            "pretrain_embed_dimension": int(self.pretrain_embed_dimension),
            "target_embed_dimension": int(self.target_embed_dimension),
            "patch_size": int(self.patch_size),
            "patch_stride": int(self.patch_stride),
            "feature_projection_dim": self.feature_projection_dim,
        }

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

    def _patchify_feature_map(self, feature_map: Any) -> Tuple[Any, Tuple[int, int]]:
        """Apply the author's padded unfold operation to one feature map."""

        padding = (self.patch_size - 1) // 2
        unfolded = self._F.unfold(
            feature_map,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            padding=padding,
        )
        height, width = (int(value) for value in feature_map.shape[-2:])
        patch_height = (height + 2 * padding - self.patch_size) // self.patch_stride + 1
        patch_width = (width + 2 * padding - self.patch_size) // self.patch_stride + 1
        batch, channels = (int(value) for value in feature_map.shape[:2])
        patches = unfolded.reshape(
            batch,
            channels,
            self.patch_size,
            self.patch_size,
            patch_height * patch_width,
        ).permute(0, 4, 1, 2, 3)
        return patches, (patch_height, patch_width)

    def _embed_feature_maps(self, feature_maps: list[Any]) -> Tuple[Any, Tuple[int, int]]:
        """Run PatchCore's MeanMapper and Aggregator embedding path."""

        if not feature_maps:
            raise ValueError("PatchCore requires at least one feature map")

        patched = [self._patchify_feature_map(feature_map) for feature_map in feature_maps]
        patch_shapes = [item[1] for item in patched]
        features = [item[0] for item in patched]
        reference_shape = patch_shapes[0]

        for index in range(1, len(features)):
            current = features[index]
            height, width = patch_shapes[index]
            current = current.reshape(
                current.shape[0], height, width, *current.shape[2:]
            ).permute(0, 3, 4, 5, 1, 2)
            base_shape = current.shape
            current = current.reshape(-1, height, width)
            current = self._F.interpolate(
                current.unsqueeze(1),
                size=reference_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            current = current.reshape(*base_shape[:-2], *reference_shape)
            current = current.permute(0, 4, 5, 1, 2, 3)
            features[index] = current.reshape(
                current.shape[0], -1, *current.shape[-3:]
            )

        features = [feature.reshape(-1, *feature.shape[-3:]) for feature in features]
        preprocessed = [
            self._F.adaptive_avg_pool1d(
                feature.reshape(len(feature), 1, -1),
                self.pretrain_embed_dimension,
            ).squeeze(1)
            for feature in features
        ]
        stacked = self._torch.stack(preprocessed, dim=1)
        embedded = self._F.adaptive_avg_pool1d(
            stacked.reshape(len(stacked), 1, -1),
            self.target_embed_dimension,
        ).reshape(len(stacked), -1)
        return embedded, reference_shape

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
        torch = self._torch
        # Load and preprocess image
        img = self._load_image_rgb(image)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            _ = self.model(img_tensor)

        features, patch_shape = self._embed_feature_maps(
            [self.feature_maps[layer] for layer in self.layers]
        )
        return features.cpu().numpy(), patch_shape

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

        selected_indices = approximate_greedy_coreset_indices(
            features,
            sampling_ratio=float(self.coreset_sampling_ratio),
            projection_dim=self.coreset_projection_dim,
            starting_points=int(self.coreset_starting_points),
            random_seed=int(self.random_seed),
        )
        return features[selected_indices]

    def fit(self, x: Iterable[ImageInput], y: Optional[NDArray] = None) -> "VisionPatchCore":
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
        del y
        np = self._np
        from .knn_index import build_knn_index

        logger.info("Fitting PatchCore detector on training images")

        x_list = list(x)
        if not x_list:
            raise ValueError("Training set cannot be empty")

        # Optional: fit a projection on a small subset of training patches.
        if self.feature_projection_dim is not None:
            fit_patches: list[NDArray] = []
            for image in x_list[: min(int(self.projection_fit_samples), len(x_list))]:
                feat, _ = self._extract_patch_features(image)
                fit_patches.append(np.asarray(feat, dtype=np.float32))
            if fit_patches:
                self._ensure_projection(np.vstack(fit_patches))

        # Extract features from all training images.
        all_features: list[NDArray] = []

        for idx, image in enumerate(x_list):
            if idx % 10 == 0:
                logger.debug("Processing image %d/%d", idx + 1, len(x_list))

            features, _ = self._extract_patch_features(image)
            features = self._maybe_project(features)
            all_features.append(np.asarray(features, dtype=np.float32))

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
        self._n_neighbors_fit = min(int(self.n_neighbors), int(self.memory_bank.shape[0]))
        self.nn_index = build_knn_index(
            backend=self.knn_backend,
            n_neighbors=self._n_neighbors_fit,
            metric="euclidean",
            n_jobs=-1,
        )
        self.nn_index.fit(np.asarray(self.memory_bank, dtype=np.float32))

        # Compute training scores to establish a threshold.
        # This enables `predict()` to return binary labels consistently.
        self.decision_scores_ = self.decision_function(x_list)
        self._process_decision_scores()

        logger.info("PatchCore training completed")
        return self

    def _patch_nearest_neighbors(self, features: NDArray) -> Tuple[NDArray, NDArray]:
        """Return the nearest memory item and distance for every query patch."""

        if self.nn_index is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        distances, indices = self.nn_index.kneighbors(features, n_neighbors=1)
        return (
            self._np.asarray(distances, dtype=self._np.float32).reshape(-1),
            self._np.asarray(indices, dtype=self._np.int64).reshape(-1),
        )

    def _image_anomaly_score(
        self,
        features: NDArray,
        patch_scores: NDArray,
        nearest_indices: NDArray,
    ) -> float:
        """Apply PatchCore's neighbourhood reweighting to the worst patch."""

        np = self._np
        if self.memory_bank is None or self.nn_index is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        if patch_scores.size == 0:
            return 0.0

        worst_idx = int(np.argmax(patch_scores))
        max_distance = float(patch_scores[worst_idx])
        neighbourhood_size = min(int(self.n_neighbors), int(self.memory_bank.shape[0]))

        # The paper uses b>1 neighbours for its softmax-style weighting. With
        # a one-neighbour lightweight configuration, retain the nearest-patch
        # distance instead of producing the degenerate zero weight.
        if neighbourhood_size <= 1:
            return max_distance

        nearest_memory_idx = int(nearest_indices[worst_idx])
        memory_query = np.asarray(
            self.memory_bank[nearest_memory_idx : nearest_memory_idx + 1],
            dtype=np.float32,
        )
        _distances, neighbour_indices = self.nn_index.kneighbors(
            memory_query,
            n_neighbors=neighbourhood_size,
        )
        support = np.asarray(
            self.memory_bank[np.asarray(neighbour_indices, dtype=np.int64).reshape(-1)],
            dtype=np.float32,
        )
        query = np.asarray(features[worst_idx], dtype=np.float32)
        support_distances = np.linalg.norm(support - query[None, :], axis=1)

        # w = 1 - exp(s*) / sum_b exp(s_b), evaluated stably.
        max_support = float(np.max(support_distances))
        denominator = float(np.exp(support_distances - max_support).sum())
        numerator = float(np.exp(max_distance - max_support))
        weight = float(np.clip(1.0 - numerator / max(denominator, 1e-12), 0.0, 1.0))
        return weight * max_distance

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
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
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )

        if self.memory_bank is None or self.nn_index is None or not hasattr(self, "threshold_"):
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        x_iter = cast(
            Iterable[ImageInput], resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        )
        scores = self.decision_function(x_iter)
        return (scores >= self.threshold_).astype(int)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
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
        # This detector scores one image at a time. Keep `batch_size` for
        # interface compatibility with BaseDeepLearningDetector.
        if batch_size is not None:
            batch_size_int = int(batch_size)
            if batch_size_int <= 0:
                raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")

        np = self._np
        if self.memory_bank is None or self.nn_index is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        x_iter = cast(
            Iterable[ImageInput],
            resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
        )
        x_list = list(x_iter)
        scores = np.zeros(len(x_list))
        if not x_list:
            return scores

        logger.info("Computing anomaly scores for %d images", len(x_list))

        for idx, image in enumerate(x_list):
            features, _ = self._extract_patch_features(image)
            features = self._maybe_project(features)

            patch_scores, nearest_indices = self._patch_nearest_neighbors(features)
            scores[idx] = self._image_anomaly_score(
                features,
                patch_scores,
                nearest_indices,
            )

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
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        # Extract patch features
        features, (h, w) = self._extract_patch_features(image_path)
        features = self._maybe_project(features)

        # Pixel localization uses the nearest memory patch distance. The
        # image-level neighbourhood weight is intentionally not broadcast to
        # every location.
        patch_scores, _nearest_indices = self._patch_nearest_neighbors(features)

        # Reshape to spatial dimensions
        expected = int(h * w)
        if patch_scores.shape[0] != expected:
            raise RuntimeError(
                f"Patch score shape mismatch: expected {expected}, got {patch_scores.shape[0]}"
            )
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
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            anomaly_map = cv2.resize(
                anomaly_map,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        if self.gaussian_sigma > 0:
            anomaly_map = cv2.GaussianBlur(
                anomaly_map,
                (0, 0),
                sigmaX=self.gaussian_sigma,
                sigmaY=self.gaussian_sigma,
            )

        return np.asarray(anomaly_map, dtype=np.float32)
