from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import require

from .registry import register_model


def _build_patchcore_inspection_model(
    *,
    checkpoint_path: str,
    device: str,
    faiss_num_workers: int,
):
    require("patchcore", extra="patchcore_inspection", purpose="patchcore-inspection backend detectors")

    from patchcore.patchcore import PatchCore  # type: ignore[import-not-found]
    import patchcore.common  # type: ignore[import-not-found]

    faiss_on_gpu = "cuda" in str(device)
    nn_method = patchcore.common.FaissNN(faiss_on_gpu, int(faiss_num_workers))

    model = PatchCore(device)
    model.load_from_path(
        load_path=str(checkpoint_path),
        device=str(device),
        nn_method=nn_method,
        prepend="",
    )
    return model


def _build_transform(*, resize: int, imagesize: int):
    # Keep imports local to reduce import-time overhead for users that don't
    # use this backend.
    from torchvision import transforms  # type: ignore[import-not-found]

    # Match patchcore-inspection's default ImageNet normalization.
    return transforms.Compose(
        [
            transforms.Resize(int(resize)),
            transforms.CenterCrop(int(imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@dataclass(frozen=True)
class _PredictResult:
    scores: NDArray
    maps: Optional[NDArray]


@register_model(
    "vision_patchcore_inspection_checkpoint",
    tags=("vision", "deep", "backend", "patchcore_inspection", "patchcore"),
    metadata={
        "description": "PatchCore (amazon-science/patchcore-inspection) checkpoint wrapper (optional backend)",
        "backend": "patchcore_inspection",
        "requires_checkpoint": True,
    },
)
class VisionPatchCoreInspectionCheckpoint:
    """Inference wrapper for PatchCore models saved by patchcore-inspection.

    This wrapper is **inference-first**: training is expected to be done via
    `amazon-science/patchcore-inspection`, producing a saved model folder that
    can be loaded for benchmarking in PyImgAno.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str = "cpu",
        contamination: float = 0.1,
        resize: int = 256,
        imagesize: int = 224,
        batch_size: int = 8,
        faiss_num_workers: int = 4,
        inferencer=None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5). Got {self.contamination}."
            )

        self.resize = int(resize)
        self.imagesize = int(imagesize)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1. Got {self.batch_size}.")

        self._transform = _build_transform(resize=self.resize, imagesize=self.imagesize)
        self._inferencer = (
            inferencer
            if inferencer is not None
            else _build_patchcore_inspection_model(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                faiss_num_workers=int(faiss_num_workers),
            )
        )

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

    def _load_images(self, paths: Sequence[str]):
        from PIL import Image
        import torch

        tensors = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            tensors.append(self._transform(img))
        return torch.stack(tensors, dim=0)

    def _predict(self, paths: Sequence[str], *, return_maps: bool) -> _PredictResult:
        import torch

        scores_list: list[float] = []
        maps_list: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                batch_paths = paths[i : i + self.batch_size]
                batch = self._load_images(batch_paths)

                batch_scores, batch_maps = self._inferencer.predict(batch)
                scores_list.extend([float(s) for s in batch_scores])

                if return_maps:
                    maps_list.extend([np.asarray(m, dtype=np.float32) for m in batch_maps])

        scores = np.asarray(scores_list, dtype=np.float64)
        maps = None
        if return_maps:
            if not maps_list:
                raise ValueError("patchcore-inspection prediction did not include anomaly maps")
            for m in maps_list:
                if m.ndim != 2:
                    raise ValueError(
                        "patchcore-inspection anomaly_map must be 2D. "
                        f"Got shape {tuple(m.shape)}"
                    )
            maps = np.stack(maps_list, axis=0).astype(np.float32, copy=False)

        return _PredictResult(scores=scores, maps=maps)

    def fit(self, X: Iterable[str], y=None):
        paths = list(X)
        if not paths:
            raise ValueError("X must contain at least one training image path.")

        self.decision_scores_ = self.decision_function(paths)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def decision_function(self, X: Iterable[str]) -> NDArray:
        paths = list(X)
        return self._predict(paths, return_maps=False).scores

    def predict(self, X: Iterable[str]) -> NDArray:
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image_path: str) -> NDArray:
        pred = self._predict([str(image_path)], return_maps=True)
        assert pred.maps is not None
        return pred.maps[0]

    def predict_anomaly_map(self, X: Iterable[str]) -> NDArray:
        paths: Sequence[str] = list(X)
        pred = self._predict(paths, return_maps=True)
        assert pred.maps is not None
        return pred.maps

