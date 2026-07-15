"""Paper-aligned SoftPatch detector.

SoftPatch extends PatchCore with position-wise patch outlier scoring, hard
denoising before coreset construction, and soft memory weights at inference.

Reference:
    Jiang et al., "SoftPatch: Unsupervised Anomaly Detection with Noisy Data",
    NeurIPS 2022.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .anomalydino import (
    PatchEmbedder,
    _embedder_from_checkpoint_payload,
    _embedder_to_checkpoint_payload,
)
from .deep_io import safe_torch_load
from .knn_index import KNNIndex, build_knn_index
from .patchknn_core import (
    AggregationMethod,
    aggregate_patch_scores,
    approximate_greedy_coreset_indices,
    reshape_patch_scores,
)
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."
WeightMethod = Literal["lof", "nearest", "gaussian"]


@dataclass
class _EmbeddedImage:
    patch_embeddings: NDArray
    grid_shape: Tuple[int, int]
    original_size: Tuple[int, int]


@dataclass
class _SoftPatchPatchCoreEmbedder:
    """Lazy adapter over the shared paper-aligned PatchCore feature path."""

    backbone: str = "wide_resnet50_2"
    layers: Tuple[str, ...] = ("layer2", "layer3")
    pretrain_embed_dimension: int = 1024
    target_embed_dimension: int = 1024
    patch_size: int = 3
    patch_stride: int = 1
    pretrained: bool = False
    device: str = "cpu"
    resize_size: int = 256
    image_size: int = 224
    _extractor: Any = None

    def _ensure_ready(self) -> Any:
        if self._extractor is not None:
            return self._extractor

        from .patchcore import VisionPatchCore

        extractor = VisionPatchCore(
            backbone=self.backbone,
            layers=list(self.layers),
            coreset_sampling_ratio=1.0,
            pretrain_embed_dimension=self.pretrain_embed_dimension,
            target_embed_dimension=self.target_embed_dimension,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            coreset_projection_dim=None,
            pretrained=self.pretrained,
            device=self.device,
            gaussian_sigma=0.0,
        )
        transforms = extractor._tv_transforms
        extractor.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(int(self.resize_size)),
                transforms.CenterCrop(int(self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._extractor = extractor
        return extractor

    def embed(
        self, image: Union[str, np.ndarray]
    ) -> Tuple[NDArray, Tuple[int, int], Tuple[int, int]]:
        extractor = self._ensure_ready()
        rgb = extractor._load_image_rgb(image)
        features, grid_shape = extractor._extract_patch_features(rgb)
        return (
            np.asarray(features, dtype=np.float32),
            (int(grid_shape[0]), int(grid_shape[1])),
            (int(rgb.shape[0]), int(rgb.shape[1])),
        )


def _cpu_state_dict(model: Any) -> dict[str, object]:
    state: dict[str, object] = {}
    for key, value in dict(model.state_dict()).items():
        detach = getattr(value, "detach", None)
        cpu = getattr(value, "cpu", None)
        state[str(key)] = detach().cpu() if callable(detach) and callable(cpu) else value
    return state


def _softpatch_embedder_to_payload(embedder: PatchEmbedder) -> dict[str, object]:
    if not isinstance(embedder, _SoftPatchPatchCoreEmbedder):
        return _embedder_to_checkpoint_payload(embedder)

    extractor = embedder._ensure_ready()
    return {
        "type": "softpatch_patchcore",
        "config": {
            "backbone": str(embedder.backbone),
            "layers": list(embedder.layers),
            "pretrain_embed_dimension": int(embedder.pretrain_embed_dimension),
            "target_embed_dimension": int(embedder.target_embed_dimension),
            "patch_size": int(embedder.patch_size),
            "patch_stride": int(embedder.patch_stride),
            "device": str(embedder.device),
            "resize_size": int(embedder.resize_size),
            "image_size": int(embedder.image_size),
        },
        "model_state_dict": _cpu_state_dict(extractor.model),
    }


def _softpatch_embedder_from_payload(payload: dict[str, object], *, device: str) -> PatchEmbedder:
    if str(payload.get("type", "")) != "softpatch_patchcore":
        return _embedder_from_checkpoint_payload(payload)

    config = dict(cast(dict[str, object], payload.get("config", {})))
    layers_value = config.get("layers", ["layer2", "layer3"])
    layers = tuple(str(layer) for layer in cast(Iterable[object], layers_value))
    embedder = _SoftPatchPatchCoreEmbedder(
        backbone=str(config.get("backbone", "wide_resnet50_2")),
        layers=layers,
        pretrain_embed_dimension=int(config.get("pretrain_embed_dimension", 1024)),
        target_embed_dimension=int(config.get("target_embed_dimension", 1024)),
        patch_size=int(config.get("patch_size", 3)),
        patch_stride=int(config.get("patch_stride", 1)),
        pretrained=False,
        device=str(device),
        resize_size=int(config.get("resize_size", 256)),
        image_size=int(config.get("image_size", 224)),
    )
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("SoftPatch checkpoint is missing backbone model_state_dict.")
    embedder._ensure_ready().model.load_state_dict(dict(model_state), strict=True)
    return embedder


@register_model(
    "vision_softpatch",
    tags=(
        "vision",
        "deep",
        "softpatch",
        "patchknn",
        "robust",
        "numpy",
        "pixel_map",
        "memory_bank",
        "neighbors",
        "neurips2022",
        "wide_resnet50_2",
        "lof",
    ),
    metadata={
        "description": "SoftPatch with position-wise LOF denoising and soft-weighted coreset",
        "paper": "SoftPatch: Unsupervised Anomaly Detection with Noisy Data",
        "paper_url": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html",
        "year": 2022,
        "supervision": "unsupervised",
        "implementation_status": "core-aligned",
        "paper_fidelity": "core-aligned",
    },
)
class VisionSoftPatch:
    """SoftPatch using the authors' WideResNet50-2/PatchCore feature contract.

    The offline-safe default keeps the paper architecture but initializes the
    backbone without ImageNet weights. Set ``pretrained=True`` to reproduce the
    authors' feature extractor weights (which may download once).
    """

    def __init__(
        self,
        *,
        embedder: Optional[PatchEmbedder] = None,
        contamination: float = 0.1,
        backbone: str = "wide_resnet50_2",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        pretrained: bool = False,
        knn_backend: str = "sklearn",
        n_neighbors: int = 1,
        coreset_sampling_ratio: float = 0.1,
        coreset_projection_dim: Optional[int] = 128,
        coreset_starting_points: int = 10,
        random_seed: int = 0,
        weight_method: WeightMethod = "lof",
        lof_k: int = 6,
        train_patch_outlier_quantile: float = 0.15,
        soft_weight: bool = True,
        noise_projection_dim: int = 128,
        gaussian_regularization: float = 0.01,
        aggregation_method: AggregationMethod = "max",
        aggregation_topk: float = 0.01,
        device: str = "cpu",
        resize_size: int = 256,
        image_size: int = 224,
        pretrain_embed_dimension: int = 1024,
        target_embed_dimension: int = 1024,
        patch_size: int = 3,
        patch_stride: int = 1,
        gaussian_sigma: float = 4.0,
    ) -> None:
        if not layers:
            raise ValueError("layers must contain at least one feature layer")
        if int(resize_size) < int(image_size):
            raise ValueError("resize_size must be >= image_size")
        if int(image_size) < 1:
            raise ValueError("image_size must be positive")
        if int(patch_size) < 1 or int(patch_size) % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if int(patch_stride) < 1:
            raise ValueError("patch_stride must be positive")

        self.backbone = str(backbone)
        self.layers = tuple(str(layer) for layer in layers)
        self.pretrained = bool(pretrained)
        self.device = str(device)
        self.resize_size = int(resize_size)
        self.image_size = int(image_size)
        self.pretrain_embed_dimension = int(pretrain_embed_dimension)
        self.target_embed_dimension = int(target_embed_dimension)
        if self.pretrain_embed_dimension < 1 or self.target_embed_dimension < 1:
            raise ValueError("SoftPatch embedding dimensions must be positive")
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

        if embedder is None:
            embedder = _SoftPatchPatchCoreEmbedder(
                backbone=self.backbone,
                layers=self.layers,
                pretrain_embed_dimension=self.pretrain_embed_dimension,
                target_embed_dimension=self.target_embed_dimension,
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                pretrained=self.pretrained,
                device=self.device,
                resize_size=self.resize_size,
                image_size=self.image_size,
            )
        self.embedder = embedder

        self.contamination = float(contamination)
        if not 0.0 < self.contamination < 0.5:
            raise ValueError(f"contamination must be in (0, 0.5), got {self.contamination}")

        self.knn_backend = str(knn_backend)
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors}")

        self.coreset_sampling_ratio = float(coreset_sampling_ratio)
        if not 0.0 < self.coreset_sampling_ratio <= 1.0:
            raise ValueError(
                f"coreset_sampling_ratio must be in (0, 1], got {self.coreset_sampling_ratio}"
            )
        self.coreset_projection_dim = (
            None if coreset_projection_dim is None else int(coreset_projection_dim)
        )
        if self.coreset_projection_dim is not None and self.coreset_projection_dim < 1:
            raise ValueError("coreset_projection_dim must be positive or None")
        self.coreset_starting_points = int(coreset_starting_points)
        if self.coreset_starting_points < 1:
            raise ValueError("coreset_starting_points must be positive")
        self.random_seed = int(random_seed)

        self.weight_method = str(weight_method).lower()
        if self.weight_method not in {"lof", "nearest", "gaussian"}:
            raise ValueError("weight_method must be one of: lof, nearest, gaussian")
        self.lof_k = int(lof_k)
        if self.lof_k < 1:
            raise ValueError("lof_k must be positive")
        self.train_patch_outlier_quantile = float(train_patch_outlier_quantile)
        if not 0.0 <= self.train_patch_outlier_quantile < 1.0:
            raise ValueError(
                "train_patch_outlier_quantile must be in [0, 1), "
                f"got {self.train_patch_outlier_quantile}"
            )
        self.soft_weight = bool(soft_weight)
        self.noise_projection_dim = int(noise_projection_dim)
        if self.noise_projection_dim < 1:
            raise ValueError("noise_projection_dim must be positive")
        self.gaussian_regularization = float(gaussian_regularization)
        if self.gaussian_regularization <= 0.0:
            raise ValueError("gaussian_regularization must be positive")

        self.aggregation_method = aggregation_method
        self.aggregation_topk = float(aggregation_topk)
        self.gaussian_sigma = float(gaussian_sigma)
        if self.gaussian_sigma < 0.0:
            raise ValueError("gaussian_sigma must be non-negative")

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None
        self._memory_bank: Optional[NDArray] = None
        self._memory_bank_weights: Optional[NDArray] = None
        self._knn_index: Optional[KNNIndex] = None
        self._n_neighbors_fit: Optional[int] = None
        self.filtered_patches_: int = 0

    def save_checkpoint(self, path: str | Path) -> Path:
        if (
            self._memory_bank is None
            or self._memory_bank_weights is None
            or self._knn_index is None
            or self._n_neighbors_fit is None
            or self.threshold_ is None
        ):
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="VisionSoftPatch checkpoint saving")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 2,
                "embedder": _softpatch_embedder_to_payload(self.embedder),
                "memory_bank": np.asarray(self._memory_bank, dtype=np.float32),
                "memory_bank_weights": np.asarray(self._memory_bank_weights, dtype=np.float32),
                "n_neighbors_fit": int(self._n_neighbors_fit),
                "decision_scores_": np.asarray(self.decision_scores_, dtype=np.float64),
                "threshold_": float(self.threshold_),
                "filtered_patches_": int(self.filtered_patches_),
                "soft_weight": bool(self.soft_weight),
                "aggregation_method": str(self.aggregation_method),
                "aggregation_topk": float(self.aggregation_topk),
                "gaussian_sigma": float(self.gaussian_sigma),
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        state = safe_torch_load(Path(path), map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Invalid VisionSoftPatch checkpoint payload.")
        if int(state.get("schema_version", 0)) != 2:
            raise ValueError(
                "Unsupported legacy SoftPatch checkpoint: refit to store paper-aligned "
                "position-wise outlier weights."
            )

        embedder_payload = state.get("embedder")
        if not isinstance(embedder_payload, dict):
            raise ValueError("VisionSoftPatch checkpoint is missing embedder payload.")
        self.embedder = _softpatch_embedder_from_payload(dict(embedder_payload), device=self.device)
        self._memory_bank = np.asarray(state["memory_bank"], dtype=np.float32)
        weights = state.get("memory_bank_weights")
        if weights is None:
            raise ValueError("SoftPatch checkpoint is missing memory_bank_weights.")
        self._memory_bank_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if self._memory_bank_weights.shape[0] != self._memory_bank.shape[0]:
            raise ValueError("SoftPatch checkpoint memory weights do not match memory bank.")
        self._n_neighbors_fit = min(
            int(state.get("n_neighbors_fit", self.n_neighbors)),
            int(self._memory_bank.shape[0]),
        )
        self._knn_index = build_knn_index(
            backend=self.knn_backend,
            n_neighbors=self._n_neighbors_fit,
        )
        self._knn_index.fit(self._memory_bank)
        self.decision_scores_ = np.asarray(state["decision_scores_"], dtype=np.float64)
        self.threshold_ = float(state["threshold_"])
        self.filtered_patches_ = int(state.get("filtered_patches_", 0))
        self.soft_weight = bool(state.get("soft_weight", self.soft_weight))
        self.aggregation_method = cast(
            AggregationMethod, state.get("aggregation_method", self.aggregation_method)
        )
        self.aggregation_topk = float(state.get("aggregation_topk", self.aggregation_topk))
        self.gaussian_sigma = float(state.get("gaussian_sigma", self.gaussian_sigma))

    @property
    def memory_bank_size_(self) -> int:
        if self._memory_bank is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        return int(self._memory_bank.shape[0])

    @property
    def memory_bank_weights_(self) -> NDArray:
        if self._memory_bank_weights is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        return np.asarray(self._memory_bank_weights, dtype=np.float32).copy()

    def _embed(self, image: Union[str, np.ndarray]) -> _EmbeddedImage:
        patch_embeddings, grid_shape, original_size = self.embedder.embed(image)
        embeddings = np.asarray(patch_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D patch embeddings, got shape {embeddings.shape}")

        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if embeddings.shape[0] != grid_h * grid_w:
            raise ValueError(
                "Patch embedding count does not match grid shape: "
                f"{embeddings.shape[0]} patches for {grid_h}x{grid_w}."
            )
        original_h, original_w = int(original_size[0]), int(original_size[1])
        if original_h <= 0 or original_w <= 0:
            raise ValueError(f"Invalid original_size: {original_size}")
        return _EmbeddedImage(
            patch_embeddings=embeddings,
            grid_shape=(grid_h, grid_w),
            original_size=(original_h, original_w),
        )

    def _project_for_noise_discriminator(self, features: NDArray) -> NDArray:
        n_images, n_patches, dimension = features.shape
        flat = np.asarray(features, dtype=np.float32).reshape(-1, dimension)
        if dimension != self.noise_projection_dim:
            rng = np.random.default_rng(self.random_seed)
            projection = rng.standard_normal((dimension, self.noise_projection_dim)).astype(
                np.float32
            )
            projection /= math.sqrt(float(self.noise_projection_dim))
            flat = flat @ projection
        return flat.reshape(n_images, n_patches, -1).transpose(1, 0, 2)

    def _compute_patch_outlier_weights(self, features: NDArray) -> NDArray:
        """Return one discriminator score per image and spatial patch."""

        grouped = self._project_for_noise_discriminator(features)
        n_patches, n_images, dimension = grouped.shape
        scores = np.ones((n_patches, n_images), dtype=np.float32)

        if self.weight_method == "lof":
            if n_images > 1:
                from sklearn.neighbors import LocalOutlierFactor

                effective_k = min(self.lof_k, n_images - 1)
                for position in range(n_patches):
                    detector = LocalOutlierFactor(
                        n_neighbors=effective_k,
                        metric="euclidean",
                    )
                    detector.fit(grouped[position])
                    scores[position] = -np.asarray(
                        detector.negative_outlier_factor_, dtype=np.float32
                    )
        elif self.weight_method == "nearest":
            scores.fill(0.0)
            if n_images > 1:
                for position in range(n_patches):
                    values = grouped[position]
                    norms = np.sum(values * values, axis=1)
                    distances_sq = norms[:, None] + norms[None, :] - 2.0 * values @ values.T
                    np.fill_diagonal(distances_sq, np.inf)
                    scores[position] = np.sqrt(np.maximum(np.min(distances_sq, axis=1), 0.0))
            scores += 1.0
        else:  # gaussian
            identity = np.eye(dimension, dtype=np.float32)
            for position in range(n_patches):
                values = grouped[position]
                centered = values - values.mean(axis=0, keepdims=True)
                covariance = centered.T @ centered / max(n_images - 1, 1)
                covariance = covariance + self.gaussian_regularization * identity
                inverse = np.linalg.inv(covariance)
                distances_sq = np.einsum("ni,ij,nj->n", centered, inverse, centered, optimize=True)
                scores[position] = np.sqrt(np.maximum(distances_sq, 0.0))
            scores += 1.0

        return np.maximum(scores.T, 0.0).astype(np.float32, copy=False)

    def fit(self, x: object = MISSING, y=None, **kwargs: object):
        del y
        items = list(
            cast(
                Iterable[Union[str, np.ndarray]],
                resolve_legacy_x_keyword(x, kwargs, method_name="fit"),
            )
        )
        if not items:
            raise ValueError("X must contain at least one training image.")

        embedded = [self._embed(item) for item in items]
        reference_grid = embedded[0].grid_shape
        reference_shape = embedded[0].patch_embeddings.shape
        for item in embedded[1:]:
            if item.grid_shape != reference_grid or item.patch_embeddings.shape != reference_shape:
                raise ValueError(
                    "SoftPatch position-wise discrimination requires identical training patch grids."
                )

        stacked = np.stack([item.patch_embeddings for item in embedded], axis=0)
        patch_weights = self._compute_patch_outlier_weights(stacked)
        memory = stacked.reshape(-1, stacked.shape[-1])
        weights = patch_weights.reshape(-1)
        original_patch_count = int(memory.shape[0])

        if self.train_patch_outlier_quantile > 0.0:
            cutoff = float(np.quantile(weights, 1.0 - self.train_patch_outlier_quantile))
            keep_mask = weights <= cutoff
            if not np.any(keep_mask):
                keep_mask[int(np.argmin(weights))] = True
        else:
            keep_mask = np.ones(weights.shape[0], dtype=bool)
        self.filtered_patches_ = int(weights.shape[0] - int(np.sum(keep_mask)))
        memory = memory[keep_mask]
        weights = weights[keep_mask]

        selected = approximate_greedy_coreset_indices(
            memory,
            sampling_ratio=self.coreset_sampling_ratio,
            projection_dim=self.coreset_projection_dim,
            starting_points=self.coreset_starting_points,
            random_seed=self.random_seed,
            sample_count=min(
                max(1, int(original_patch_count * self.coreset_sampling_ratio)),
                int(memory.shape[0]),
            ),
        )
        self._memory_bank = np.asarray(memory[selected], dtype=np.float32)
        self._memory_bank_weights = np.asarray(weights[selected], dtype=np.float32)

        self._n_neighbors_fit = min(self.n_neighbors, int(self._memory_bank.shape[0]))
        self._knn_index = build_knn_index(
            backend=self.knn_backend,
            n_neighbors=self._n_neighbors_fit,
        )
        self._knn_index.fit(self._memory_bank)

        self.decision_scores_ = self.decision_function(items)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _patch_scores(self, embedded: _EmbeddedImage) -> NDArray:
        if self._knn_index is None or self._memory_bank_weights is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        if self._n_neighbors_fit is None:
            raise RuntimeError("Internal error: missing fitted neighbor count.")

        distances, indices = self._knn_index.kneighbors(
            embedded.patch_embeddings,
            n_neighbors=self._n_neighbors_fit,
        )
        distances_array = np.asarray(distances, dtype=np.float32)
        indices_array = np.asarray(indices, dtype=np.int64)
        if distances_array.ndim != 2 or indices_array.shape != distances_array.shape:
            raise RuntimeError("kNN backend returned invalid SoftPatch neighbor arrays.")

        patch_scores = distances_array.mean(axis=1)
        if self.soft_weight:
            patch_scores = patch_scores * self._memory_bank_weights[indices_array[:, 0]]
        return np.asarray(patch_scores, dtype=np.float32)

    def decision_function(self, x: object = MISSING, **kwargs: object) -> NDArray:
        items = list(
            cast(
                Iterable[Union[str, np.ndarray]],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        scores = np.zeros(len(items), dtype=np.float64)
        for index, item in enumerate(items):
            patch_scores = self._patch_scores(self._embed(item))
            scores[index] = aggregate_patch_scores(
                patch_scores,
                method=self.aggregation_method,
                topk=self.aggregation_topk,
            )
        return scores

    def predict(self, x: object = MISSING, **kwargs: object) -> NDArray:
        if self.threshold_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        scores = self.decision_function(
            cast(
                Iterable[Union[str, np.ndarray]],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict"),
            )
        )
        return (scores > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image: Union[str, np.ndarray]) -> NDArray:
        embedded = self._embed(image)
        patch_grid = reshape_patch_scores(
            self._patch_scores(embedded),
            grid_h=embedded.grid_shape[0],
            grid_w=embedded.grid_shape[1],
        )
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required to upsample SoftPatch anomaly maps."
            ) from exc

        original_h, original_w = embedded.original_size
        anomaly_map = cv2.resize(
            np.asarray(patch_grid, dtype=np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.gaussian_sigma > 0.0:
            anomaly_map = cv2.GaussianBlur(
                anomaly_map,
                (0, 0),
                sigmaX=self.gaussian_sigma,
                sigmaY=self.gaussian_sigma,
            )
        return np.asarray(anomaly_map, dtype=np.float32)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        items = list(
            cast(
                Iterable[Union[str, np.ndarray]],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
            )
        )
        return np.stack([self.get_anomaly_map(item) for item in items])
