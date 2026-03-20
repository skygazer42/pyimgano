from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import require

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .registry import register_model

PATCHCORE_INSPECTION_BACKEND_DETECTORS = "patchcore-inspection backend detectors"
_PATCHCORE_INSPECTION_REQUIRED_FILES = (
    "nnscorer_search_index.faiss",
    "patchcore_params.pkl",
)


def audit_patchcore_inspection_saved_model(checkpoint_path: str) -> dict[str, object]:
    root = Path(str(checkpoint_path))
    present_files = [
        name for name in _PATCHCORE_INSPECTION_REQUIRED_FILES if (root / name).is_file()
    ]
    missing_files = [
        name for name in _PATCHCORE_INSPECTION_REQUIRED_FILES if name not in set(present_files)
    ]

    payload: dict[str, object] = {
        "path": str(root),
        "artifact_format": "patchcore-saved-model",
        "artifact_format_status": "missing",
        "checkpoint_version_sensitive": True,
        "required_files": list(_PATCHCORE_INSPECTION_REQUIRED_FILES),
        "present_files": present_files,
        "missing_files": missing_files,
        "errors": [],
    }

    if not root.exists():
        payload["errors"] = ["checkpoint_path_missing"]
        return payload

    if not root.is_dir():
        payload["artifact_format_status"] = "invalid"
        payload["errors"] = ["checkpoint_path_not_directory"]
        return payload

    if missing_files:
        payload["artifact_format_status"] = "partial"
        payload["errors"] = ["missing_required_patchcore_saved_model_files"]
        return payload

    payload["artifact_format_status"] = "recognized"
    return payload


def _build_patchcore_inspection_model(
    *,
    checkpoint_path: str,
    device: str,
    faiss_num_workers: int,
):
    # `patchcore-inspection` is an optional backend installed separately (VCS).
    # We still want missing dependency errors to be actionable.
    require("torch", extra="torch", purpose=PATCHCORE_INSPECTION_BACKEND_DETECTORS)
    require("faiss", extra="faiss", purpose=PATCHCORE_INSPECTION_BACKEND_DETECTORS)
    require("patchcore", purpose=PATCHCORE_INSPECTION_BACKEND_DETECTORS)

    import patchcore.common  # type: ignore[import-not-found]
    from patchcore.patchcore import PatchCore  # type: ignore[import-not-found]

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
    transforms = require(
        "torchvision.transforms",
        extra="torch",
        purpose="patchcore-inspection backend transforms",
    )

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
    tags=("vision", "deep", "backend", "patchcore_inspection", "patchcore", "memory_bank"),
    metadata={
        "description": "PatchCore (amazon-science/patchcore-inspection) checkpoint wrapper (optional backend)",
        "backend": "patchcore_inspection",
        "requires_checkpoint": True,
        "paper": "Towards Total Recall in Industrial Anomaly Detection",
        "year": 2022,
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
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

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
        torch = require("torch", extra="torch", purpose="patchcore-inspection backend inference")
        from PIL import Image

        tensors = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            tensors.append(self._transform(img))
        return torch.stack(tensors, dim=0)

    def _predict(self, paths: Sequence[str], *, return_maps: bool) -> _PredictResult:
        torch = require("torch", extra="torch", purpose="patchcore-inspection backend inference")

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

    def fit(self, x: object = MISSING, y=None, **kwargs: object):
        del y
        paths = list(cast(Iterable[str], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not paths:
            raise ValueError("X must contain at least one training image path.")

        self.decision_scores_ = self.decision_function(paths)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def decision_function(self, x: object = MISSING, **kwargs: object) -> NDArray:
        paths = list(
            cast(
                Iterable[str],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        return self._predict(paths, return_maps=False).scores

    def predict(self, x: object = MISSING, **kwargs: object) -> NDArray:
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(
            cast(Iterable[str], resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        )
        return (scores > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image_path: str) -> NDArray:
        pred = self._predict([str(image_path)], return_maps=True)
        assert pred.maps is not None
        return pred.maps[0]

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        paths: Sequence[str] = list(
            cast(
                Iterable[str],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
            )
        )
        pred = self._predict(paths, return_maps=True)
        assert pred.maps is not None
        return pred.maps
