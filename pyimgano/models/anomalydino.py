from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from ._image_batch import _coerce_single_rgb_image
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .deep_io import safe_torch_load
from .knn_index import KNNIndex, build_knn_index
from .patchknn_core import AggregationMethod, aggregate_patch_scores, reshape_patch_scores
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."
PAPER_MODEL_NAME = "dinov2_vits14"
PAPER_IMAGE_SIZE = 448
PAPER_ROTATIONS = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
PAPER_GAUSSIAN_SIGMA = 4.0
PAPER_MASKED_CLASSES = frozenset(
    {
        "capsule",
        "hazelnut",
        "pill",
        "screw",
        "toothbrush",
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipefryum",
    }
)


def _normalize_rows(values: NDArray) -> NDArray[np.float32]:
    rows = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


class PatchEmbedder(Protocol):
    """Protocol for patch embedders used by :class:`VisionAnomalyDINO`."""

    def embed(
        self, image: Union[str, np.ndarray]
    ) -> Tuple[NDArray, Tuple[int, int], Tuple[int, int]]: ...


@dataclass
class _EmbeddedImage:
    patch_embeddings: NDArray
    grid_shape: Tuple[int, int]
    original_size: Tuple[int, int]
    patch_mask: NDArray[np.bool_]


def _embedder_to_checkpoint_payload(embedder: PatchEmbedder) -> dict[str, object]:
    if isinstance(embedder, TorchHubDinoV2Embedder):
        payload: dict[str, object] = {
            "type": "torchhub_dinov2",
            "config": {
                "model_name": str(embedder.model_name),
                "device": str(embedder.device),
                "image_size": int(embedder.image_size),
                "hub_repo": str(embedder.hub_repo),
            },
            "patch_size": (
                int(embedder._patch_size)
                if getattr(embedder, "_patch_size", None) is not None
                else None
            ),
        }
        model = getattr(embedder, "_model", None)
        state_dict = getattr(model, "state_dict", None)
        if model is not None and callable(state_dict):
            raw_state = state_dict()
            normalized_state: dict[str, object] = {}
            for key, value in dict(raw_state).items():
                detach = getattr(value, "detach", None)
                cpu = getattr(value, "cpu", None)
                if callable(detach) and callable(cpu):
                    normalized_state[str(key)] = detach().cpu()
                else:
                    normalized_state[str(key)] = value
            payload["model_state_dict"] = normalized_state
        return payload

    raise NotImplementedError(
        "VisionAnomalyDINO checkpointing only supports TorchHubDinoV2Embedder.\n"
        "Custom embedder pickle payloads are disabled because they are unsafe to deserialize."
    )


def _embedder_from_checkpoint_payload(payload: dict[str, object]) -> PatchEmbedder:
    payload_type = str(payload.get("type", ""))
    if payload_type == "pickle":
        raise ValueError(
            "VisionAnomalyDINO legacy pickle embedder payloads are disabled.\n"
            "Re-export the checkpoint with a TorchHubDinoV2Embedder-based detector."
        )

    if payload_type != "torchhub_dinov2":
        raise ValueError("Unsupported patch embedder checkpoint payload.")

    config = dict(cast(dict[str, object], payload.get("config", {})))
    embedder = TorchHubDinoV2Embedder(
        model_name=str(config.get("model_name", PAPER_MODEL_NAME)),
        device=str(config.get("device", "cpu")),
        image_size=int(config.get("image_size", PAPER_IMAGE_SIZE)),
        hub_repo=str(config.get("hub_repo", "facebookresearch/dinov2")),
    )

    model_state = payload.get("model_state_dict", None)
    if isinstance(model_state, dict):
        embedder._ensure_loaded()
        model = getattr(embedder, "_model", None)
        load_state_dict = getattr(model, "load_state_dict", None)
        if callable(load_state_dict):
            load_state_dict(dict(model_state), strict=False)

    patch_size = payload.get("patch_size", None)
    if patch_size is not None:
        embedder._patch_size = int(patch_size)

    return embedder


@register_model(
    "vision_anomalydino",
    tags=(
        "vision",
        "deep",
        "anomalydino",
        "knn",
        "dinov2",
        "few-shot",
        "numpy",
        "pixel_map",
        "neighbors",
        "wacv2025",
    ),
    metadata={
        "description": "Native AnomalyDINO DINOv2 patch-memory and preprocessing adaptation",
        "paper": "AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2",
        "paper_url": "https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html",
        "year": 2025,
        "conference": "WACV",
        "implementation_status": "native-paper-method-dinov2-adaptation",
        "paper_fidelity": "paper-adaptation",
        "type": "nearest-neighbor",
        "supervision": "few-shot",
        "supports_pixel_map": True,
        "requires_checkpoint": False,
        "weights_source": "DINOv2 dinov2_vits14 LVD-142M weights",
    },
)
class VisionAnomalyDINO:
    """AnomalyDINO paper adaptation for training-free few-shot detection.

    Notes
    -----
    - The embedder is injectable so unit tests can run without torch.
    - Paper rotations are enabled for the built-in DINOv2 embedder. Injected
      custom embedders retain one unmodified reference unless requested.
    """

    def __init__(
        self,
        *,
        embedder: Optional[PatchEmbedder] = None,
        contamination: float = 0.1,
        pretrained: bool = False,
        knn_backend: str = "sklearn",
        n_neighbors: int = 1,
        coreset_sampling_ratio: float = 1.0,
        random_seed: int = 0,
        aggregation_method: AggregationMethod = "topk_mean",
        aggregation_topk: float = 0.01,
        device: str = "cpu",
        image_size: int = PAPER_IMAGE_SIZE,
        dino_model_name: str = PAPER_MODEL_NAME,
        reference_rotations: Optional[Sequence[float]] = None,
        class_name: Optional[str] = None,
        masking: Optional[bool] = None,
        mask_reference_images: bool = False,
        gaussian_sigma: float = PAPER_GAUSSIAN_SIGMA,
    ) -> None:
        uses_builtin_embedder = embedder is None or isinstance(embedder, TorchHubDinoV2Embedder)
        if embedder is None:
            if bool(pretrained):
                if int(image_size) <= 0:
                    raise ValueError("image_size must be positive")
                embedder = TorchHubDinoV2Embedder(
                    model_name=dino_model_name,
                    device=device,
                    image_size=image_size,
                )
            else:
                raise ValueError(
                    "vision_anomalydino requires a patch embedder.\n"
                    "Pass embedder=... (recommended, offline) or set pretrained=True to allow "
                    "torch.hub to load DINOv2 weights (may download from the internet)."
                )

        self.embedder = embedder
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")
        self.pretrained = bool(pretrained)
        self.knn_backend = str(knn_backend)
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1. Got {n_neighbors}.")
        self.coreset_sampling_ratio = float(coreset_sampling_ratio)
        if not (0.0 < self.coreset_sampling_ratio <= 1.0):
            raise ValueError(
                f"coreset_sampling_ratio must be in (0, 1]. Got {self.coreset_sampling_ratio}."
            )
        self.random_seed = int(random_seed)
        self.aggregation_method = aggregation_method
        self.aggregation_topk = float(aggregation_topk)
        if not 0.0 < self.aggregation_topk <= 1.0:
            raise ValueError("aggregation_topk must be in (0, 1].")
        rotations = PAPER_ROTATIONS if uses_builtin_embedder else (0.0,)
        if reference_rotations is not None:
            rotations = tuple(float(angle) for angle in reference_rotations)
        if not rotations or not np.isfinite(rotations).all():
            raise ValueError("reference_rotations must contain finite angles")
        self.reference_rotations = tuple(rotations)
        self.class_name = None if class_name is None else str(class_name)
        normalized_class = (
            ""
            if self.class_name is None
            else self.class_name.lower().replace("_", "").replace(" ", "")
        )
        auto_masking = normalized_class in PAPER_MASKED_CLASSES and callable(
            getattr(self.embedder, "foreground_mask", None)
        )
        self.masking = auto_masking if masking is None else bool(masking)
        self.mask_reference_images = bool(mask_reference_images)
        self.gaussian_sigma = float(gaussian_sigma)
        if not np.isfinite(self.gaussian_sigma) or self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be finite and non-negative")

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

        self._memory_bank: Optional[NDArray] = None
        self._knn_index: Optional[KNNIndex] = None
        self._n_neighbors_fit: Optional[int] = None

    def save_checkpoint(self, path: str | Path) -> Path:
        if self._memory_bank is None or self._knn_index is None or self._n_neighbors_fit is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        if self.threshold_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        embedder_payload = _embedder_to_checkpoint_payload(self.embedder)

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="VisionAnomalyDINO checkpoint saving")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embedder": embedder_payload,
                "memory_bank": np.asarray(self._memory_bank, dtype=np.float32),
                "n_neighbors_fit": int(self._n_neighbors_fit),
                "decision_scores_": np.asarray(self.decision_scores_, dtype=np.float64),
                "threshold_": float(self.threshold_),
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        state = safe_torch_load(Path(path), map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Invalid VisionAnomalyDINO checkpoint payload.")

        embedder_payload = state.get("embedder", None)
        if not isinstance(embedder_payload, dict):
            raise ValueError("VisionAnomalyDINO checkpoint is missing embedder payload.")
        self.embedder = _embedder_from_checkpoint_payload(dict(embedder_payload))
        self._memory_bank = _normalize_rows(np.asarray(state["memory_bank"], dtype=np.float32))
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

    @property
    def memory_bank_size_(self) -> int:
        if self._memory_bank is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        return int(self._memory_bank.shape[0])

    def _embed(
        self,
        image: Union[str, np.ndarray],
        *,
        apply_mask: bool = False,
    ) -> _EmbeddedImage:
        patch_embeddings, grid_shape, original_size = self.embedder.embed(image)
        patch_embeddings_np = np.asarray(patch_embeddings, dtype=np.float32)
        if patch_embeddings_np.ndim != 2 or 0 in patch_embeddings_np.shape:
            raise ValueError(
                f"Expected non-empty 2D patch embeddings, got shape {patch_embeddings_np.shape}"
            )
        if not np.isfinite(patch_embeddings_np).all():
            raise ValueError("Patch embeddings must contain only finite values")

        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"Invalid grid_shape: {grid_shape}")
        if patch_embeddings_np.shape[0] != grid_h * grid_w:
            raise ValueError(
                "Patch embedding count does not match grid shape. "
                f"Got {patch_embeddings_np.shape[0]} patches for grid {grid_h}x{grid_w}."
            )

        original_h, original_w = int(original_size[0]), int(original_size[1])
        if original_h <= 0 or original_w <= 0:
            raise ValueError(f"Invalid original_size: {original_size}")

        patch_mask = np.ones(patch_embeddings_np.shape[0], dtype=bool)
        if apply_mask:
            foreground_mask = getattr(self.embedder, "foreground_mask", None)
            if not callable(foreground_mask):
                raise TypeError("masking=True requires an embedder with foreground_mask(...)")
            patch_mask = np.asarray(
                foreground_mask(patch_embeddings_np, (grid_h, grid_w)),
                dtype=bool,
            ).reshape(-1)
            if patch_mask.shape[0] != patch_embeddings_np.shape[0]:
                raise ValueError("foreground_mask must return one value per patch")
            if not patch_mask.any():
                raise ValueError("foreground_mask removed every patch")

        return _EmbeddedImage(
            patch_embeddings=patch_embeddings_np,
            grid_shape=(grid_h, grid_w),
            original_size=(original_h, original_w),
            patch_mask=patch_mask,
        )

    def _reference_variants(
        self,
        item: Union[str, np.ndarray],
    ) -> list[Union[str, np.ndarray]]:
        if self.reference_rotations == (0.0,):
            return [item]

        import cv2

        from pyimgano.preprocessing.augmentation import rotate_image

        image = np.ascontiguousarray(_coerce_single_rgb_image(item))
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        return [
            rotate_image(
                image,
                angle,
                center=center,
                border_mode=cv2.BORDER_DEFAULT,
            )
            for angle in self.reference_rotations
        ]

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

        embedded_train = [
            self._embed(
                variant,
                apply_mask=self.masking and self.mask_reference_images,
            )
            for item in items
            for variant in self._reference_variants(item)
        ]
        memory_bank = np.concatenate(
            [embedded.patch_embeddings[embedded.patch_mask] for embedded in embedded_train],
            axis=0,
        )
        memory_bank = _normalize_rows(memory_bank)

        if self.coreset_sampling_ratio < 1.0:
            rng = np.random.default_rng(self.random_seed)
            n_total = int(memory_bank.shape[0])
            n_keep = max(1, int(math.ceil(self.coreset_sampling_ratio * n_total)))
            n_keep = min(n_keep, n_total)
            keep_idx = rng.choice(n_total, size=n_keep, replace=False)
            memory_bank = memory_bank[keep_idx]

        self._memory_bank = memory_bank

        effective_k = min(max(1, self.n_neighbors), int(memory_bank.shape[0]))
        self._n_neighbors_fit = effective_k
        self._knn_index = build_knn_index(
            backend=self.knn_backend,
            n_neighbors=effective_k,
        )
        self._knn_index.fit(memory_bank)

        self.decision_scores_ = self.decision_function(items)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _patch_scores(self, embedded: _EmbeddedImage) -> NDArray:
        if self._knn_index is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        if self._n_neighbors_fit is None:
            raise RuntimeError("Internal error: missing fitted neighbor count.")
        if self._memory_bank is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        query = _normalize_rows(embedded.patch_embeddings[embedded.patch_mask])
        _distances, indices = self._knn_index.kneighbors(
            query,
            n_neighbors=self._n_neighbors_fit,
        )
        indices_np = np.asarray(indices, dtype=np.int64)
        if indices_np.ndim != 2:
            raise RuntimeError(f"Expected 2D kNN indices, got shape {indices_np.shape}")

        neighbors = self._memory_bank[indices_np]
        cosine = np.einsum("nd,nkd->nk", query, neighbors)
        both_zero = ~query.any(axis=1, keepdims=True) & ~neighbors.any(axis=2)
        cosine[both_zero] = 1.0
        selected_scores = (1.0 - np.clip(cosine, -1.0, 1.0)).mean(axis=1)
        patch_scores = np.zeros(embedded.patch_embeddings.shape[0], dtype=np.float32)
        patch_scores[embedded.patch_mask] = selected_scores.astype(np.float32, copy=False)
        return patch_scores

    def decision_function(self, x: object = MISSING, **kwargs: object) -> NDArray:
        items = list(
            cast(
                Iterable[Union[str, np.ndarray]],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        scores = np.zeros(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            embedded = self._embed(item, apply_mask=self.masking)
            patch_scores = self._patch_scores(embedded)
            scores[i] = aggregate_patch_scores(
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
        embedded = self._embed(image, apply_mask=self.masking)
        patch_scores = self._patch_scores(embedded)
        patch_grid = reshape_patch_scores(
            patch_scores,
            grid_h=embedded.grid_shape[0],
            grid_w=embedded.grid_shape[1],
        )

        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required to upsample anomaly maps.\n"
                "Install it via:\n  pip install 'opencv-python'\n"
                f"Original error: {exc}"
            ) from exc

        original_h, original_w = embedded.original_size
        upsampled = cv2.resize(
            np.asarray(patch_grid, dtype=np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.gaussian_sigma > 0:
            upsampled = gaussian_filter(upsampled, sigma=self.gaussian_sigma)
        return np.asarray(upsampled, dtype=np.float32)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        items = list(
            cast(
                Iterable[Union[str, np.ndarray]],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
            )
        )
        maps = [self.get_anomaly_map(item) for item in items]
        return np.stack(maps)


@dataclass
class TorchHubDinoV2Embedder:
    """Default patch embedder using DINOv2 via ``torch.hub``.

    This class is **lazy**: it avoids importing torch / downloading weights
    until the first call to :meth:`embed`.
    """

    model_name: str = PAPER_MODEL_NAME
    device: str = "cpu"
    image_size: int = PAPER_IMAGE_SIZE
    hub_repo: str = "facebookresearch/dinov2"

    _model: Any = None
    _transform: Any = None
    _patch_size: Optional[int] = None

    _legacy_attr_aliases = {"_Image": "_image_cls"}

    def __post_init__(self) -> None:
        if int(self.image_size) <= 0:
            raise ValueError("image_size must be positive")

    def __getattr__(self, name: str):
        alias = type(self)._legacy_attr_aliases.get(name)
        if alias is not None:
            return getattr(self, alias)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        alias = type(self)._legacy_attr_aliases.get(name)
        super().__setattr__(alias or name, value)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="DINOv2 embedder")

        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Pillow is required for image loading.\n"
                "Install it via:\n  pip install 'Pillow'\n"
                f"Original error: {exc}"
            ) from exc

        transforms = require("torchvision.transforms", extra="torch", purpose="DINOv2 embedder")

        self._image_cls = Image  # type: ignore[attr-defined]
        self._torch = torch  # type: ignore[attr-defined]

        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    int(self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        model = torch.hub.load(self.hub_repo, self.model_name)
        model.requires_grad_(False).eval().to(self.device)
        self._model = model

        patch_size = None
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
            ps = model.patch_embed.patch_size
            if isinstance(ps, tuple):
                patch_size = int(ps[0])
            else:
                patch_size = int(ps)
        self._patch_size = patch_size

    def embed(
        self, image: Union[str, np.ndarray]
    ) -> Tuple[NDArray, Tuple[int, int], Tuple[int, int]]:
        self._ensure_loaded()

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB image, got dtype={image.dtype}")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected shape (H,W,3), got {image.shape}")
            image = self._image_cls.fromarray(np.ascontiguousarray(image), mode="RGB")
        else:
            image = self._image_cls.open(str(image)).convert("RGB")
        original_w, original_h = image.size

        if not self._patch_size:
            raise RuntimeError("DINOv2 model does not expose its patch size")
        x = self._transform(image)
        height, width = x.shape[-2:]
        cropped_height = height - height % self._patch_size
        cropped_width = width - width % self._patch_size
        if cropped_height <= 0 or cropped_width <= 0:
            raise ValueError("Image is too small for the DINOv2 patch size")
        x = x[:, :cropped_height, :cropped_width].unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            if hasattr(self._model, "get_intermediate_layers"):
                outputs = self._model.get_intermediate_layers(x)
                patch_tokens = outputs[0] if outputs else None
            else:
                features = self._model.forward_features(x)
                patch_tokens = features.get("x_norm_patchtokens")
        if patch_tokens is None:
            raise RuntimeError(
                "Unable to extract patch tokens from DINOv2 output. "
                "Please provide a custom embedder via embedder=..."
            )
        patch_embeddings = patch_tokens[0].detach().cpu().numpy().astype(np.float32, copy=False)
        num_patches = int(patch_embeddings.shape[0])
        grid_h = cropped_height // self._patch_size
        grid_w = cropped_width // self._patch_size
        if grid_h * grid_w != num_patches:
            raise RuntimeError(
                f"Unable to infer patch grid shape from {num_patches} patches. "
                "Please provide a custom embedder."
            )

        return patch_embeddings, (grid_h, grid_w), (int(original_h), int(original_w))

    def foreground_mask(
        self,
        patch_embeddings: NDArray,
        grid_shape: Tuple[int, int],
        *,
        threshold: float = 10.0,
        kernel_size: int = 3,
        center_border: float = 0.2,
    ) -> NDArray[np.bool_]:
        """Paper PCA foreground mask with center check and morphology."""

        import cv2
        from sklearn.decomposition import PCA

        features = np.asarray(patch_embeddings, dtype=np.float32)
        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if features.shape[0] != grid_h * grid_w:
            raise ValueError("Patch count does not match grid_shape")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0 <= center_border < 0.5:
            raise ValueError("center_border must be in [0, 0.5)")

        component = PCA(n_components=1, svd_solver="randomized").fit_transform(features)
        component = component.reshape(grid_h, grid_w)
        mask = (component > float(threshold)).astype(np.uint8)
        y0, y1 = int(grid_h * center_border), int(grid_h * (1 - center_border))
        x0, x1 = int(grid_w * center_border), int(grid_w * (1 - center_border))
        center = mask[y0:y1, x0:x1]
        if center.size and center.sum() <= center.size * 0.35:
            mask = (-component > float(threshold)).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask.reshape(-1).astype(bool, copy=False)
