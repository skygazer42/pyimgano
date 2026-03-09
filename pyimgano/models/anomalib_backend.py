from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import require

from .registry import register_model


def _build_torch_inferencer(*, checkpoint_path: str, device: str):
    require("anomalib", extra="anomalib", purpose="anomalib backend detectors")

    # Support multiple anomalib versions with different import paths.
    try:
        from anomalib.deploy import TorchInferencer  # type: ignore
    except Exception:  # pragma: no cover
        from anomalib.deploy.inferencers import TorchInferencer  # type: ignore

    return TorchInferencer(path=checkpoint_path, device=device)


@dataclass
class _AnomalibPredictResult:
    score: float
    anomaly_map: Optional[NDArray] = None


def _extract_score_and_map(result) -> _AnomalibPredictResult:
    # Newer anomalib uses structured objects; older versions may return dict-like.
    if isinstance(result, dict):
        score = result.get("anomaly_score", None)
        if score is None:
            score = result.get("pred_score", None)
        if score is None:
            score = result.get("score", None)
        if score is None:
            raise ValueError("Unable to extract anomaly score from anomalib prediction result")

        anomaly_map = result.get("anomaly_map", None)
        if anomaly_map is None:
            anomaly_map = result.get("segmentation_map", None)
        return _AnomalibPredictResult(score=float(score), anomaly_map=anomaly_map)

    score = getattr(result, "anomaly_score", None)
    if score is None:
        score = getattr(result, "pred_score", None)
    if score is None:
        score = getattr(result, "score", None)
    if score is None:
        raise ValueError("Unable to extract anomaly score from anomalib prediction result")

    anomaly_map = getattr(result, "anomaly_map", None)
    if anomaly_map is None:
        anomaly_map = getattr(result, "segmentation_map", None)

    return _AnomalibPredictResult(score=float(score), anomaly_map=anomaly_map)


def _to_numpy(value) -> NDArray:
    """Best-effort conversion for anomalib anomaly maps to numpy.

    anomalib may return numpy arrays, torch tensors, or other array-like types.
    We avoid importing torch directly and instead use duck-typing.
    """

    if isinstance(value, np.ndarray):
        return value

    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    numpy = getattr(value, "numpy", None)
    if callable(detach) and callable(cpu) and callable(numpy):
        try:
            return value.detach().cpu().numpy()
        except Exception:
            pass

    if callable(numpy):
        try:
            return value.numpy()
        except Exception:
            pass

    return np.asarray(value)


def _normalize_anomaly_map(anomaly_map) -> NDArray:
    """Normalize an anomalib anomaly map to a stable (H, W) float32 contract."""

    arr = np.asarray(_to_numpy(anomaly_map))

    # Common anomalib output shapes include (H, W), (1, H, W), (H, W, 1),
    # and sometimes (1, 1, H, W). Strip only leading/trailing singleton dims
    # to keep (H, W) intact.
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(
            "anomalib anomaly_map must be 2D after normalization. " f"Got shape {tuple(arr.shape)}"
        )

    return np.asarray(arr, dtype=np.float32)


@register_model(
    "vision_anomalib_checkpoint",
    tags=("vision", "deep", "backend", "anomalib"),
    metadata={
        "description": "Generic anomalib checkpoint inferencer wrapper (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
    },
)
class VisionAnomalibCheckpoint:
    """Generic inference wrapper for anomalib checkpoints.

    This is intentionally **inference-first**: training is expected to be done
    via anomalib, producing a checkpoint that this wrapper can load.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str = "cpu",
        contamination: float = 0.1,
        inferencer=None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5). Got {self.contamination}.")

        self._inferencer = (
            inferencer
            if inferencer is not None
            else _build_torch_inferencer(checkpoint_path=checkpoint_path, device=device)
        )

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

    def fit(self, X: Iterable[str], y=None):
        paths = list(X)
        if not paths:
            raise ValueError("X must contain at least one training image path.")

        self.decision_scores_ = self.decision_function(paths)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def decision_function(self, X: Iterable[str]) -> NDArray:
        paths = list(X)
        scores = np.zeros(len(paths), dtype=np.float64)
        for i, path in enumerate(paths):
            pred = self._inferencer.predict(path)
            scores[i] = _extract_score_and_map(pred).score
        return scores

    def predict(self, X: Iterable[str]) -> NDArray:
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image_path: str) -> NDArray:
        pred = self._inferencer.predict(image_path)
        extracted = _extract_score_and_map(pred)
        if extracted.anomaly_map is None:
            raise ValueError("anomalib prediction did not include an anomaly map")
        return _normalize_anomaly_map(extracted.anomaly_map)

    def predict_anomaly_map(self, X: Iterable[str]) -> NDArray:
        paths: Sequence[str] = list(X)
        maps = [self.get_anomaly_map(path) for path in paths]
        return np.stack(maps)


@register_model(
    "vision_patchcore_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "patchcore", "memory_bank"),
    metadata={
        "description": "PatchCore via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "patchcore",
        "paper": "Towards Total Recall in Industrial Anomaly Detection",
        "year": 2022,
    },
)
class VisionPatchCoreAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with PatchCore tags.

    The implementation is shared; only the registry entry differs.
    """


@register_model(
    "vision_padim_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "padim"),
    metadata={
        "description": "PaDiM via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "padim",
        "paper": "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization",
        "year": 2020,
    },
)
class VisionPadimAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with PaDiM tags."""


@register_model(
    "vision_stfpm_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "stfpm", "distillation"),
    metadata={
        "description": "STFPM via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "stfpm",
        "paper": "Student-Teacher Feature Pyramid Matching for Anomaly Detection",
        "year": 2021,
    },
)
class VisionStfpmAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with STFPM tags."""


@register_model(
    "vision_draem_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "draem", "reconstruction"),
    metadata={
        "description": "DRAEM via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "draem",
        "paper": "DRAEM: Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection",
        "year": 2021,
    },
)
class VisionDraemAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with DRAEM tags."""


@register_model(
    "vision_fastflow_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "fastflow", "flow"),
    metadata={
        "description": "FastFlow via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "fastflow",
        "paper": "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows",
        "year": 2021,
    },
)
class VisionFastflowAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with FastFlow tags."""


@register_model(
    "vision_reverse_distillation_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "reverse_distillation", "distillation"),
    metadata={
        "description": "Reverse Distillation via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "reverse_distillation",
        "paper": "Anomaly Detection via Reverse Distillation from One-Class Embedding",
        "year": 2022,
    },
)
class VisionReverseDistillationAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with Reverse Distillation tags."""


@register_model(
    "vision_dfm_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "dfm", "gaussian"),
    metadata={
        "description": "DFM via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "dfm",
        "paper": "Probabilistic Modeling of Deep Features for Out-of-Distribution and Adversarial Detection",
        "year": 2019,
    },
)
class VisionDfmAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with DFM tags."""


@register_model(
    "vision_cflow_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "cflow", "flow"),
    metadata={
        "description": "CFlow via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "cflow",
        "paper": "Real-Time Unsupervised Anomaly Detection with Localization",
        "year": 2022,
    },
)
class VisionCflowAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with CFlow tags."""


@register_model(
    "vision_efficientad_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "efficientad", "distillation"),
    metadata={
        "description": "EfficientAD via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "efficientad",
        "paper": "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies",
        "year": 2024,
    },
)
class VisionEfficientadAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with EfficientAD tags."""


@register_model(
    "vision_dinomaly_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "dinomaly", "reconstruction"),
    metadata={
        "description": "Dinomaly via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "dinomaly",
        "paper": "Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection",
        "year": 2025,
    },
)
class VisionDinomalyAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with Dinomaly tags."""


@register_model(
    "vision_cfa_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "cfa"),
    metadata={
        "description": "CFA via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "cfa",
        "paper": "CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization",
        "year": 2022,
    },
)
class VisionCfaAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with CFA tags."""


@register_model(
    "vision_csflow_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "csflow", "flow"),
    metadata={
        "description": "CS-Flow via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "csflow",
        "paper": "Fully Convolutional Cross-Scale-Flows for Image-Based Defect Detection",
        "year": 2022,
    },
)
class VisionCsflowAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with CS-Flow tags."""


@register_model(
    "vision_dfkde_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "dfkde", "density"),
    metadata={
        "description": "DFKDE via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "dfkde",
        "paper": "Deep Feature Kernel Density Estimation",
    },
)
class VisionDfkdeAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with DFKDE tags."""


@register_model(
    "vision_dsr_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "dsr", "reconstruction"),
    metadata={
        "description": "DSR via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "dsr",
        "paper": "DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection",
        "year": 2022,
    },
)
class VisionDsrAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with DSR tags."""


@register_model(
    "vision_ganomaly_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "ganomaly", "gan"),
    metadata={
        "description": "GANomaly via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "ganomaly",
        "paper": "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training",
        "year": 2018,
    },
)
class VisionGanomalyAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with GANomaly tags."""


@register_model(
    "vision_rkde_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "rkde", "density"),
    metadata={
        "description": "R-KDE via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "rkde",
        "paper": "Region Based Anomaly Detection With Real-Time Training and Analysis",
        "year": 2019,
    },
)
class VisionRkdeAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with R-KDE tags."""


@register_model(
    "vision_uflow_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "uflow", "flow"),
    metadata={
        "description": "U-Flow via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "uflow",
        "paper": "U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold",
        "year": 2022,
    },
)
class VisionUflowAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with U-Flow tags."""


@register_model(
    "vision_winclip_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "winclip", "clip"),
    metadata={
        "description": "WinCLIP via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "winclip",
        "paper": "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation",
        "year": 2023,
    },
)
class VisionWinclipAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with WinCLIP tags."""


@register_model(
    "vision_fre_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "fre", "reconstruction"),
    metadata={
        "description": "FRE via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "fre",
        "paper": "FRE: A Fast Method For Anomaly Detection And Segmentation",
        "year": 2023,
    },
)
class VisionFreAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with FRE tags."""


@register_model(
    "vision_supersimplenet_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "supersimplenet"),
    metadata={
        "description": "SuperSimpleNet via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "supersimplenet",
        "paper": "SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection",
        "year": 2024,
    },
)
class VisionSuperSimpleNetAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with SuperSimpleNet tags."""


@register_model(
    "vision_vlmad_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "vlmad"),
    metadata={
        "description": "VLM-AD via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
        "requires_checkpoint": True,
        "anomalib_model": "vlm-ad",
    },
)
class VisionVLMADAnomalib(VisionAnomalibCheckpoint):
    """Alias for ``vision_anomalib_checkpoint`` with VLM-AD tags."""
