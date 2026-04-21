from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .deep_io import safe_torch_load
from .knn_index import KNNIndex, build_knn_index
from .patchknn_core import AggregationMethod, aggregate_patch_scores, reshape_patch_scores
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."


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
        model_name=str(config.get("model_name", "dinov2_vits14")),
        device=str(config.get("device", "cpu")),
        image_size=int(config.get("image_size", 518)),
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
    tags=("vision", "deep", "anomalydino", "knn", "dinov2", "numpy", "pixel_map", "neighbors"),
    metadata={
        "description": "AnomalyDINO-style DINOv2 patch-kNN detector (few-shot friendly)",
        "paper": "AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2",
        "year": 2025,
    },
)
class VisionAnomalyDINO:
    """DINOv2 patch-kNN anomaly detector (AnomalyDINO-style).

    Notes
    -----
    - The embedder is injectable so unit tests can run without torch.
    - This implementation is inference-first and calibrates a simple threshold
      on the training set.
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
        image_size: int = 518,
        dino_model_name: str = "dinov2_vits14",
    ) -> None:
        if embedder is None:
            if bool(pretrained):
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
        self.coreset_sampling_ratio = float(coreset_sampling_ratio)
        if not (0.0 < self.coreset_sampling_ratio <= 1.0):
            raise ValueError(
                f"coreset_sampling_ratio must be in (0, 1]. Got {self.coreset_sampling_ratio}."
            )
        self.random_seed = int(random_seed)
        self.aggregation_method = aggregation_method
        self.aggregation_topk = float(aggregation_topk)

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
        self._memory_bank = np.asarray(state["memory_bank"], dtype=np.float32)
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

    def _embed(self, image: Union[str, np.ndarray]) -> _EmbeddedImage:
        patch_embeddings, grid_shape, original_size = self.embedder.embed(image)
        patch_embeddings_np = np.asarray(patch_embeddings, dtype=np.float32)
        if patch_embeddings_np.ndim != 2:
            raise ValueError(f"Expected 2D patch embeddings, got shape {patch_embeddings_np.shape}")

        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if patch_embeddings_np.shape[0] != grid_h * grid_w:
            raise ValueError(
                "Patch embedding count does not match grid shape. "
                f"Got {patch_embeddings_np.shape[0]} patches for grid {grid_h}x{grid_w}."
            )

        original_h, original_w = int(original_size[0]), int(original_size[1])
        if original_h <= 0 or original_w <= 0:
            raise ValueError(f"Invalid original_size: {original_size}")

        return _EmbeddedImage(
            patch_embeddings=patch_embeddings_np,
            grid_shape=(grid_h, grid_w),
            original_size=(original_h, original_w),
        )

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

        embedded_train = [self._embed(item) for item in items]
        memory_bank = np.concatenate([e.patch_embeddings for e in embedded_train], axis=0)

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

        distances, _indices = self._knn_index.kneighbors(
            embedded.patch_embeddings,
            n_neighbors=self._n_neighbors_fit,
        )
        distances_np = np.asarray(distances, dtype=np.float32)
        if distances_np.ndim != 2:
            raise RuntimeError(f"Expected 2D kNN distances, got shape {distances_np.shape}")

        return distances_np.min(axis=1)

    def decision_function(self, x: object = MISSING, **kwargs: object) -> NDArray:
        items = list(
            cast(
                Iterable[Union[str, np.ndarray]],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        scores = np.zeros(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            embedded = self._embed(item)
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
        embedded = self._embed(image)
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

    model_name: str = "dinov2_vits14"
    device: str = "cpu"
    image_size: int = 518
    hub_repo: str = "facebookresearch/dinov2"

    _model: Any = None
    _transform: Any = None
    _patch_size: Optional[int] = None

    _legacy_attr_aliases = {"_Image": "_image_cls"}

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
                transforms.Resize((int(self.image_size), int(self.image_size))),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        model = torch.hub.load(self.hub_repo, self.model_name)
        model.eval()
        model.to(self.device)
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

        x = self._transform(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            features = self._model.forward_features(x)  # type: ignore[no-any-return]

        patch_tokens = None
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "x_norm_patch_tokens", "patch_tokens"):
                if key in features:
                    patch_tokens = features[key]
                    break
        if patch_tokens is None:
            raise RuntimeError(
                "Unable to extract patch tokens from DINOv2 output. "
                "Please provide a custom embedder via embedder=..."
            )

        patch_embeddings = patch_tokens[0].detach().cpu().numpy().astype(np.float32, copy=False)
        num_patches = int(patch_embeddings.shape[0])

        if self._patch_size:
            grid_h = int(self.image_size) // int(self._patch_size)
            grid_w = int(self.image_size) // int(self._patch_size)
            if grid_h * grid_w != num_patches:
                grid_h = int(round(math.sqrt(num_patches)))
                grid_w = grid_h
        else:
            grid_h = int(round(math.sqrt(num_patches)))
            grid_w = grid_h

        if grid_h * grid_w != num_patches:
            raise RuntimeError(
                f"Unable to infer patch grid shape from {num_patches} patches. "
                "Please provide a custom embedder."
            )

        return patch_embeddings, (grid_h, grid_w), (int(original_h), int(original_w))
