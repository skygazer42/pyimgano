from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from .knn_index import KNNIndex, build_knn_index
from .registry import register_model


_AggregationMethod = Literal["topk_mean", "max", "mean"]


def _aggregate_patch_scores(
    patch_scores: NDArray,
    *,
    method: _AggregationMethod = "topk_mean",
    topk: float = 0.01,
) -> float:
    scores = np.asarray(patch_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("patch_scores must be non-empty")

    method_lower = str(method).lower()
    if method_lower == "max":
        return float(np.max(scores))
    if method_lower == "mean":
        return float(np.mean(scores))
    if method_lower == "topk_mean":
        topk_float = float(topk)
        if not (0.0 < topk_float <= 1.0):
            raise ValueError("topk must be a fraction in (0, 1].")

        k = max(1, int(math.ceil(topk_float * scores.size)))
        k = min(k, scores.size)

        top_scores = np.partition(scores, -k)[-k:]
        return float(np.mean(top_scores))

    raise ValueError(f"Unknown aggregation method: {method}. Choose from: topk_mean, max, mean")


def _reshape_patch_scores(
    patch_scores: NDArray,
    *,
    grid_h: int,
    grid_w: int,
) -> NDArray:
    scores = np.asarray(patch_scores)
    if scores.ndim != 1:
        scores = scores.reshape(-1)

    grid_h_int = int(grid_h)
    grid_w_int = int(grid_w)
    expected = grid_h_int * grid_w_int
    if scores.size != expected:
        raise ValueError(
            f"Expected {expected} patch scores for grid {grid_h_int}x{grid_w_int}, got {scores.size}."
        )

    return scores.reshape(grid_h_int, grid_w_int)


class PatchEmbedder(Protocol):
    """Protocol for patch embedders used by :class:`VisionAnomalyDINO`."""

    def embed(self, image_path: str) -> Tuple[NDArray, Tuple[int, int], Tuple[int, int]]: ...


@dataclass
class _EmbeddedImage:
    patch_embeddings: NDArray
    grid_shape: Tuple[int, int]
    original_size: Tuple[int, int]


@register_model(
    "vision_anomalydino",
    tags=("vision", "deep", "anomalydino", "knn", "dinov2"),
    metadata={
        "description": "AnomalyDINO-style DINOv2 patch-kNN detector (few-shot friendly)",
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
        pretrained: bool = True,
        knn_backend: str = "sklearn",
        n_neighbors: int = 1,
        aggregation_method: _AggregationMethod = "topk_mean",
        aggregation_topk: float = 0.01,
        device: str = "cpu",
        image_size: int = 518,
        dino_model_name: str = "dinov2_vits14",
    ) -> None:
        if embedder is None:
            embedder = TorchHubDinoV2Embedder(
                model_name=dino_model_name,
                device=device,
                image_size=image_size,
            )

        self.embedder = embedder
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5). Got {self.contamination}."
            )
        self.pretrained = bool(pretrained)
        self.knn_backend = str(knn_backend)
        self.n_neighbors = int(n_neighbors)
        self.aggregation_method = aggregation_method
        self.aggregation_topk = float(aggregation_topk)

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

        self._memory_bank: Optional[NDArray] = None
        self._knn_index: Optional[KNNIndex] = None

    def _embed(self, image_path: str) -> _EmbeddedImage:
        patch_embeddings, grid_shape, original_size = self.embedder.embed(image_path)
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

    def fit(self, X: Iterable[str], y=None):
        paths = list(X)
        if not paths:
            raise ValueError("X must contain at least one training image path.")

        embedded_train = [self._embed(path) for path in paths]
        memory_bank = np.concatenate([e.patch_embeddings for e in embedded_train], axis=0)
        self._memory_bank = memory_bank

        self._knn_index = build_knn_index(
            backend=self.knn_backend,
            n_neighbors=max(1, self.n_neighbors),
        )
        self._knn_index.fit(memory_bank)

        self.decision_scores_ = self.decision_function(paths)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _patch_scores(self, embedded: _EmbeddedImage) -> NDArray:
        if self._knn_index is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        distances, _indices = self._knn_index.kneighbors(
            embedded.patch_embeddings,
            n_neighbors=max(1, self.n_neighbors),
        )
        distances_np = np.asarray(distances, dtype=np.float32)
        if distances_np.ndim != 2:
            raise RuntimeError(f"Expected 2D kNN distances, got shape {distances_np.shape}")

        return distances_np.min(axis=1)

    def decision_function(self, X: Iterable[str]) -> NDArray:
        paths = list(X)
        scores = np.zeros(len(paths), dtype=np.float64)
        for i, path in enumerate(paths):
            embedded = self._embed(path)
            patch_scores = self._patch_scores(embedded)
            scores[i] = _aggregate_patch_scores(
                patch_scores,
                method=self.aggregation_method,
                topk=self.aggregation_topk,
            )
        return scores

    def predict(self, X: Iterable[str]) -> NDArray:
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image_path: str) -> NDArray:
        embedded = self._embed(image_path)
        patch_scores = self._patch_scores(embedded)
        patch_grid = _reshape_patch_scores(patch_scores, grid_h=embedded.grid_shape[0], grid_w=embedded.grid_shape[1])

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

    def predict_anomaly_map(self, X: Iterable[str]) -> NDArray:
        paths = list(X)
        maps = [self.get_anomaly_map(path) for path in paths]
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

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "torch is required for the default DINOv2 embedder.\n"
                "Install it via:\n  pip install 'torch'\n"
                f"Original error: {exc}"
            ) from exc

        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Pillow is required for image loading.\n"
                "Install it via:\n  pip install 'Pillow'\n"
                f"Original error: {exc}"
            ) from exc

        try:
            from torchvision import transforms  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "torchvision is required for the default DINOv2 embedder.\n"
                "Install it via:\n  pip install 'torchvision'\n"
                f"Original error: {exc}"
            ) from exc

        self._Image = Image  # type: ignore[attr-defined]
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

    def embed(self, image_path: str) -> Tuple[NDArray, Tuple[int, int], Tuple[int, int]]:
        self._ensure_loaded()

        image = self._Image.open(image_path).convert("RGB")
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
