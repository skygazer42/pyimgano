from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from ._image_batch import _coerce_single_rgb_image
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .anomalydino import TorchHubDinoV2Embedder
from .patchknn_core import aggregate_patch_scores
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."

# The released headline scripts use ViT-Large and blocks 4..18. The paper's
# implementation-details baseline uses ViT-Base and blocks 2..9.
RELEASE_MODEL_NAME = "dinov2_vitl14_reg"
PAPER_BASE_MODEL_NAME = "dinov2_vitb14_reg"
RELEASE_LARGE_LAYERS = tuple(range(4, 19))
PAPER_BASE_LAYERS = tuple(range(2, 10))
PAPER_RESIZE_SIZE = 448
PAPER_CROP_SIZE = 392
PAPER_MAP_SIZE = 256
PAPER_TOPK = 0.01
PAPER_SUPPORT_VIEWS = ("identity", "rot90", "rot180", "rot270", "flip_y", "flip_x")
PAPER_QUERY_VIEWS = ("identity", "flip_y", "positive_clamp")


def _normalize_rows(values: NDArray[Any]) -> NDArray[np.float32]:
    rows = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


def _as_float_matrix(value: Any) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"Expected a non-empty 2D patch embedding array, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Patch embeddings must contain only finite values.")
    return array


def _as_rgb_uint8(image: Any) -> NDArray[np.uint8]:
    array = np.asarray(_coerce_single_rgb_image(image))
    if not np.isfinite(array).all():
        raise ValueError("VisionAD images must contain only finite values.")
    minimum, maximum = float(array.min()), float(array.max())
    if minimum < 0 or maximum > 255:
        raise ValueError("VisionAD image values must be in [0, 1] or [0, 255].")
    if np.issubdtype(array.dtype, np.floating) and maximum <= 1.0:
        array = array * 255.0
    return np.ascontiguousarray(np.rint(array).astype(np.uint8))


def _default_layers(model_name: str) -> tuple[int, ...]:
    name = str(model_name).lower()
    if "vitl14" in name:
        return RELEASE_LARGE_LAYERS
    if "vitb14" in name or "vits14" in name:
        return PAPER_BASE_LAYERS
    raise ValueError("interested_layers is required for an unsupported DINOv2 architecture.")


def _forward_fused_tokens(model: Any, images: Any, layers: Sequence[int]) -> Any:
    """Author DADE path: average raw tokens from the selected transformer blocks."""

    selected = tuple(int(layer) for layer in layers)
    if not selected or selected != tuple(sorted(set(selected))) or selected[0] < 0:
        raise ValueError("interested_layers must be a non-empty increasing sequence.")
    blocks = model.blocks
    if selected[-1] >= len(blocks):
        raise ValueError(
            f"interested_layers contains block {selected[-1]}, but the backbone has "
            f"{len(blocks)} blocks."
        )
    prepare = getattr(model, "prepare_tokens", None)
    if not callable(prepare):
        prepare = getattr(model, "prepare_tokens_with_masks", None)
    if not callable(prepare):
        raise TypeError("DINOv2 backbone must implement prepare_tokens(_with_masks).")

    tokens = prepare(images)
    outputs = []
    selected_set = set(selected)
    for index, block in enumerate(blocks):
        tokens = block(tokens)
        if index in selected_set:
            outputs.append(tokens)
        if index >= selected[-1]:
            break
    if len(outputs) == 1:
        return outputs[0]
    import torch

    return torch.stack(outputs, dim=1).mean(dim=1)


def _transform_view(torch: Any, tensor: Any, name: str) -> Any:
    if name == "identity":
        return tensor
    if name == "flip_y":
        return torch.flip(tensor, dims=(-2,))
    if name == "flip_x":
        return torch.flip(tensor, dims=(-1,))
    if name == "positive_clamp":
        return tensor.clamp_min(0)
    if name.startswith("rot"):
        return torch.rot90(tensor, k=int(name[3:]) // 90, dims=(-2, -1))
    raise ValueError(f"Unknown VisionAD view: {name!r}.")


@dataclass(frozen=True)
class _ArrayEmbedding:
    patches: NDArray[np.float32]
    global_feature: NDArray[np.float32]
    grid_shape: tuple[int, int] | None


def _call_embedder(embedder: Any, image: Any) -> _ArrayEmbedding:
    if embedder is None:
        raise ValueError("embedder is required for the injected VisionAD path.")
    output = embedder.embed(image) if hasattr(embedder, "embed") else embedder(image)

    grid_shape = None
    global_feature = None
    if isinstance(output, tuple):
        if len(output) == 4:
            patches, global_feature, grid, _original_size = output
            grid_shape = (int(grid[0]), int(grid[1]))
        elif len(output) == 3:
            patches, grid, _original_size = output
            grid_shape = (int(grid[0]), int(grid[1]))
        else:
            raise ValueError(
                "embedder tuple output must be (patches, grid, original_size) or "
                "(patches, global_feature, grid, original_size)."
            )
    else:
        patches = output

    patch_matrix = _as_float_matrix(patches)
    if grid_shape is not None and patch_matrix.shape[0] != grid_shape[0] * grid_shape[1]:
        raise ValueError("Patch embedding count does not match grid_shape.")
    if global_feature is None:
        global_vector = patch_matrix.mean(axis=0)
    else:
        global_vector = np.asarray(global_feature, dtype=np.float32).reshape(-1)
    if global_vector.shape[0] != patch_matrix.shape[1] or not np.isfinite(global_vector).all():
        raise ValueError("global_feature must be a finite vector with the patch embedding width.")
    return _ArrayEmbedding(patch_matrix, global_vector, grid_shape)


def _validate_labels(labels: Sequence[Any], count: int) -> tuple[Any, ...]:
    if len(labels) != count:
        raise ValueError(f"y must contain one category label per support image ({count}).")
    normalized = tuple(labels)
    for label in normalized:
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError("VisionAD category labels must be hashable.") from exc
    return normalized


class _InjectedVisionADBackend:
    """Offline test hook retaining the paper's cosine memory search semantics."""

    def __init__(self, embedder: Any, search_backend: Any, topk: float) -> None:
        self.embedder = embedder
        self.search_backend = search_backend
        self.topk = float(topk)
        self.memories: dict[Any, NDArray[np.float32]] = {}
        self.global_labels: list[Any] = []
        self.global_memory: NDArray[np.float32] | None = None

    def fit(self, items: Sequence[Any], labels: Sequence[Any]) -> None:
        embedded = [_call_embedder(self.embedder, item) for item in items]
        if self.search_backend is not None:
            if not hasattr(self.search_backend, "fit") or not hasattr(self.search_backend, "score"):
                raise TypeError(
                    "search_backend must implement .fit(train_patches) and .score(patches)."
                )
            self.search_backend.fit([entry.patches for entry in embedded])
            return

        first_global: dict[Any, NDArray[np.float32]] = {}
        grouped: dict[Any, list[NDArray[np.float32]]] = {}
        for label, entry in zip(labels, embedded):
            grouped.setdefault(label, []).append(entry.patches)
            first_global.setdefault(label, entry.global_feature)
        self.memories = {
            label: _normalize_rows(np.concatenate(patches, axis=0))
            for label, patches in grouped.items()
        }
        self.global_labels = list(first_global)
        self.global_memory = _normalize_rows(
            np.stack([first_global[label] for label in self.global_labels], axis=0)
        )

    def score(self, item: Any) -> tuple[float, NDArray[np.float32] | None]:
        embedded = _call_embedder(self.embedder, item)
        if self.search_backend is not None:
            image_score, patch_scores = self.search_backend.score(embedded.patches)
            scores = np.asarray(patch_scores, dtype=np.float32).reshape(-1)
            if scores.shape[0] != embedded.patches.shape[0]:
                raise ValueError("search_backend.score must return one score per patch.")
            anomaly_map = (
                None if embedded.grid_shape is None else scores.reshape(embedded.grid_shape)
            )
            return float(image_score), anomaly_map

        if self.global_memory is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        query_global = _normalize_rows(embedded.global_feature[None])[0]
        label = self.global_labels[int(np.argmax(self.global_memory @ query_global))]
        query = _normalize_rows(embedded.patches)
        similarities = query @ self.memories[label].T
        patch_scores = (1.0 - similarities.max(axis=1)).astype(np.float32, copy=False)
        image_score = aggregate_patch_scores(patch_scores, method="topk_mean", topk=self.topk)
        anomaly_map = (
            None if embedded.grid_shape is None else patch_scores.reshape(embedded.grid_shape)
        )
        return image_score, anomaly_map


class TorchVisionADBackend:
    """Frozen DINOv2-R implementation of the released VisionAD inference path."""

    def __init__(
        self,
        *,
        model_name: str = RELEASE_MODEL_NAME,
        interested_layers: Sequence[int] | None = None,
        device: str = "cpu",
        resize_size: int = PAPER_RESIZE_SIZE,
        crop_size: int = PAPER_CROP_SIZE,
        map_size: int = PAPER_MAP_SIZE,
        topk: float = PAPER_TOPK,
        batch_size: int = 8,
        memory_chunk_size: int = 16384,
        model: Any = None,
        preprocess: Any = None,
    ) -> None:
        self.model_name = str(model_name)
        self.interested_layers = tuple(
            _default_layers(self.model_name)
            if interested_layers is None
            else (int(layer) for layer in interested_layers)
        )
        self.device = str(device)
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.map_size = int(map_size)
        self.topk = float(topk)
        self.batch_size = int(batch_size)
        self.memory_chunk_size = int(memory_chunk_size)
        if min(self.resize_size, self.crop_size, self.map_size, self.batch_size) <= 0:
            raise ValueError("VisionAD image and batch sizes must be positive.")
        if self.crop_size > self.resize_size or self.crop_size % 14:
            raise ValueError("crop_size must not exceed resize_size and must be divisible by 14.")
        if not 0.0 < self.topk <= 1.0:
            raise ValueError("topk must be in (0, 1].")
        if self.memory_chunk_size <= 0:
            raise ValueError("memory_chunk_size must be positive.")

        self.model = model
        self.preprocess = preprocess
        self._torch: Any = None
        self._loader: TorchHubDinoV2Embedder | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._memories: dict[str, dict[Any, Any]] = {}
        self._global_labels: list[Any] = []
        self._global_memory: Any = None
        self.selected_category_: Any = None

    def _ensure_loaded(self) -> None:
        if self._torch is not None:
            return
        from pyimgano.utils.optional_deps import require

        self._torch = require("torch", extra="torch", purpose="VisionAD")
        if self.model is None:
            self._loader = TorchHubDinoV2Embedder(
                model_name=self.model_name,
                device=self.device,
                image_size=self.resize_size,
            )
            self._loader._ensure_loaded()
            self.model = self._loader._model
        self.model.requires_grad_(False).eval().to(self.device)
        if self.preprocess is None:
            transforms = require("torchvision.transforms", extra="torch", purpose="VisionAD")
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((self.resize_size, self.resize_size)),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

    def _prepare(self, image: Any) -> tuple[Any, tuple[int, int]]:
        self._ensure_loaded()
        from PIL import Image

        array = _as_rgb_uint8(image)
        tensor = self.preprocess(Image.fromarray(array, mode="RGB"))
        if tensor.ndim != 3 or tuple(tensor.shape) != (3, self.crop_size, self.crop_size):
            raise ValueError(
                "VisionAD preprocess must return a CHW tensor with shape "
                f"(3, {self.crop_size}, {self.crop_size})."
            )
        return tensor, (int(array.shape[0]), int(array.shape[1]))

    def _encode(self, tensors: Sequence[Any]) -> tuple[Any, Any, tuple[int, int]]:
        if not tensors:
            raise ValueError("VisionAD cannot encode an empty image batch.")
        patch_batches, global_batches = [], []
        register_count = int(getattr(self.model, "num_register_tokens", 0))
        with self._torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                batch = self._torch.stack(tensors[start : start + self.batch_size]).to(self.device)
                fused = _forward_fused_tokens(self.model, batch, self.interested_layers)
                global_batches.append(fused[:, 0])
                patch_batches.append(fused[:, 1 + register_count :])
        patches = self._torch.cat(patch_batches, dim=0)
        globals_ = self._torch.cat(global_batches, dim=0)
        side = math.isqrt(int(patches.shape[1]))
        if side * side != int(patches.shape[1]):
            raise RuntimeError("VisionAD expects a square DINOv2 patch grid.")
        return patches, globals_, (side, side)

    def fit(self, items: Sequence[Any], labels: Sequence[Any]) -> None:
        self._ensure_loaded()
        prepared = [self._prepare(item)[0] for item in items]
        _base_patches, base_globals, self._grid_shape = self._encode(prepared)

        first_indices: dict[Any, int] = {}
        for index, label in enumerate(labels):
            first_indices.setdefault(label, index)
        self._global_labels = list(first_indices)
        self._global_memory = self._torch.nn.functional.normalize(
            self._torch.stack(
                [base_globals[first_indices[label]] for label in self._global_labels], dim=0
            ),
            dim=-1,
        )

        self._memories = {}
        for query_view in PAPER_QUERY_VIEWS:
            transformed = [_transform_view(self._torch, tensor, query_view) for tensor in prepared]
            augmented = [
                _transform_view(self._torch, tensor, support_view)
                for support_view in PAPER_SUPPORT_VIEWS
                for tensor in transformed
            ]
            repeated_labels = tuple(labels) * len(PAPER_SUPPORT_VIEWS)
            patches, _globals, grid_shape = self._encode(augmented)
            if grid_shape != self._grid_shape:
                raise RuntimeError("VisionAD view transforms changed the patch grid.")
            grouped: dict[Any, list[Any]] = {}
            for label, patch_grid in zip(repeated_labels, patches):
                grouped.setdefault(label, []).append(patch_grid)
            self._memories[query_view] = {
                label: self._torch.nn.functional.normalize(self._torch.cat(values, dim=0), dim=-1)
                for label, values in grouped.items()
            }

    def _nearest_distance(self, query: Any, memory: Any) -> Any:
        query = self._torch.nn.functional.normalize(query, dim=-1)
        best = self._torch.full(
            (int(query.shape[0]),),
            -self._torch.inf,
            dtype=query.dtype,
            device=query.device,
        )
        for start in range(0, int(memory.shape[0]), self.memory_chunk_size):
            similarity = query @ memory[start : start + self.memory_chunk_size].T
            best = self._torch.maximum(best, similarity.amax(dim=1))
        return 1.0 - best

    def score(self, item: Any) -> tuple[float, NDArray[np.float32]]:
        if self._global_memory is None or self._grid_shape is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        tensor, _original_size = self._prepare(item)
        base_patches, base_global, grid_shape = self._encode([tensor])
        query_global = self._torch.nn.functional.normalize(base_global, dim=-1)
        category_index = int((query_global @ self._global_memory.T).argmax(dim=1).item())
        category = self._global_labels[category_index]
        self.selected_category_ = category

        grids = []
        for query_view in PAPER_QUERY_VIEWS:
            if query_view == "identity":
                patches = base_patches[0]
            else:
                patches = self._encode([_transform_view(self._torch, tensor, query_view)])[0][0]
            distances = self._nearest_distance(patches, self._memories[query_view][category])
            grid = distances.reshape(grid_shape)
            if query_view == "flip_y":
                grid = self._torch.flip(grid, dims=(-2,))
            grids.append(grid)
        anomaly_grid = self._torch.stack(grids, dim=0).sum(dim=0)
        anomaly_map = self._torch.nn.functional.interpolate(
            anomaly_grid[None, None],
            size=(self.map_size, self.map_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        flat_scores = anomaly_map.flatten()
        topk_count = max(1, int(flat_scores.numel() * self.topk))
        image_score = float(self._torch.topk(flat_scores, topk_count).values.mean().item())
        anomaly_map_np = anomaly_map.detach().cpu().numpy().astype(np.float32, copy=False)
        return image_score, anomaly_map_np


@register_model(
    "vision_visionad",
    tags=(
        "vision",
        "deep",
        "neighbors",
        "few-shot",
        "dinov2",
        "memory_bank",
        "pixel_map",
        "visionad",
    ),
    metadata={
        "description": "Native VisionAD DINOv2-R multi-layer cosine-memory adaptation",
        "paper": "Search is All You Need for Few-shot Anomaly Detection",
        "paper_url": "https://arxiv.org/abs/2504.11895",
        "year": 2025,
        "implementation_status": "native-paper-method-dinov2-adaptation",
        "paper_fidelity": "paper-adaptation",
        "supervision": "few-shot",
        "supports_pixel_map": True,
        "requires_checkpoint": True,
        "weights_source": "DINOv2-Register ViT-L/14 LVD-142M weights",
    },
)
class VisionVisionAD:
    """Training-free VisionAD with released support/query augmentation and search."""

    def __init__(
        self,
        *,
        backend: Any = None,
        embedder: Any = None,
        search_backend: Any = None,
        contamination: float = 0.1,
        pretrained: bool = False,
        device: str = "cpu",
        dino_model_name: str = RELEASE_MODEL_NAME,
        interested_layers: Sequence[int] | None = None,
        resize_size: int = PAPER_RESIZE_SIZE,
        crop_size: int = PAPER_CROP_SIZE,
        map_size: int = PAPER_MAP_SIZE,
        aggregation_topk: float = PAPER_TOPK,
        batch_size: int = 8,
        memory_chunk_size: int = 16384,
    ) -> None:
        self.contamination = float(contamination)
        if not 0.0 < self.contamination < 0.5:
            raise ValueError(f"contamination must be in (0, 0.5), got {contamination}.")
        self.pretrained = bool(pretrained)
        self.aggregation_topk = float(aggregation_topk)
        if not 0.0 < self.aggregation_topk <= 1.0:
            raise ValueError("aggregation_topk must be in (0, 1].")
        if backend is not None and (embedder is not None or search_backend is not None):
            raise ValueError("backend cannot be combined with embedder/search_backend.")
        if backend is None:
            if embedder is not None:
                backend = _InjectedVisionADBackend(embedder, search_backend, self.aggregation_topk)
            elif self.pretrained:
                backend = TorchVisionADBackend(
                    model_name=dino_model_name,
                    interested_layers=interested_layers,
                    device=device,
                    resize_size=resize_size,
                    crop_size=crop_size,
                    map_size=map_size,
                    topk=self.aggregation_topk,
                    batch_size=batch_size,
                    memory_chunk_size=memory_chunk_size,
                )
            else:
                raise ValueError(
                    "vision_visionad requires backend=.../embedder=... for offline use, or "
                    "pretrained=True to load the released DINOv2-R backbone."
                )
        if not hasattr(backend, "fit") or not hasattr(backend, "score"):
            raise TypeError("backend must implement .fit(items, labels) and .score(item).")
        self.backend = backend
        self.decision_scores_: NDArray[np.float64] | None = None
        self.threshold_: float | None = None

    def fit(self, x: object = MISSING, y: Any = None, **kwargs: object):
        items = list(cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not items:
            raise ValueError("X must contain at least one normal support image.")
        labels = _validate_labels(tuple(0 for _ in items) if y is None else tuple(y), len(items))
        self.backend.fit(items, labels)
        self.decision_scores_ = self.decision_function(items)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _score_item(self, item: Any) -> tuple[float, NDArray[np.float32] | None]:
        score, anomaly_map = self.backend.score(item)
        if anomaly_map is None:
            return float(score), None
        array = np.asarray(anomaly_map, dtype=np.float32)
        if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
            raise ValueError("backend.score must return a finite, non-empty 2D anomaly map.")
        return float(score), array

    def decision_function(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float64]:
        items = list(
            cast(
                Iterable[Any],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        return np.asarray([self._score_item(item)[0] for item in items], dtype=np.float64)

    def get_anomaly_map(self, image: Any) -> NDArray[np.float32]:
        _score, anomaly_map = self._score_item(image)
        if anomaly_map is None:
            raise ValueError("The injected embedder must provide grid metadata for anomaly maps.")
        return anomaly_map

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        items = list(
            cast(
                Iterable[Any],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
            )
        )
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        return np.stack([self.get_anomaly_map(item) for item in items], axis=0)

    def predict(self, x: object = MISSING, **kwargs: object) -> NDArray[np.int64]:
        if self.threshold_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        scores = self.decision_function(
            cast(Iterable[Any], resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        )
        return (scores > self.threshold_).astype(np.int64)
