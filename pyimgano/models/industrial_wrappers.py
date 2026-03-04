# -*- coding: utf-8 -*-
"""Industrial convenience wrappers built on our pipeline contracts.

These wrappers exist to provide "batteries-included" model names that:
- follow the `VisionFeaturePipeline` design (feature extractor + core detector)
- avoid heavy deep-model dependencies
- are safe by default (deterministic, no implicit downloads)

They intentionally stay small; for anything more complex, prefer
`vision_feature_pipeline` or `vision_embedding_core` directly.
"""

from __future__ import annotations

from typing import Any, Mapping

from pyimgano.models.registry import register_model
from pyimgano.pipelines.feature_pipeline import VisionFeaturePipeline
from pyimgano.models.vision_embedding_core import VisionEmbeddingCoreDetector


def _default_structural_extractor(*, max_size: int = 512) -> dict[str, Any]:
    return {"name": "structural", "kwargs": {"max_size": int(max_size)}}


def _default_torchvision_embedding_extractor(
    *,
    backbone: str = "resnet18",
    pretrained: bool = False,
    pool: str = "avg",
    device: str = "cpu",
    image_size: int = 224,
) -> dict[str, Any]:
    # Safe defaults: `pretrained=False` avoids implicit weight downloads.
    return {
        "name": "torchvision_backbone",
        "kwargs": {
            "backbone": str(backbone),
            "pretrained": bool(pretrained),
            "pool": str(pool),
            "device": str(device),
            "image_size": int(image_size),
        },
    }


def _default_torchscript_embedding_extractor(
    *,
    checkpoint_path: str,
    device: str = "cpu",
    batch_size: int = 16,
    image_size: int = 224,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "batch_size": int(batch_size),
        "image_size": int(image_size),
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return {"name": "torchscript_embed", "kwargs": kwargs}


def _default_onnx_embedding_extractor(
    *,
    checkpoint_path: str,
    device: str = "cpu",
    batch_size: int = 16,
    image_size: int = 224,
    cache_dir: str | None = None,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "batch_size": int(batch_size),
        "image_size": int(image_size),
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if providers is not None:
        kwargs["providers"] = [str(p) for p in providers]
    return {"name": "onnx_embed", "kwargs": kwargs}


@register_model(
    "vision_structural_ecod",
    tags=("vision", "classical", "pipeline", "industrial", "fast"),
    metadata={"description": "Industrial baseline: structural features + core_ecod"},
)
class VisionStructuralECOD(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_ecod",
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_copod",
    tags=("vision", "classical", "pipeline", "industrial", "fast", "parameter-free"),
    metadata={"description": "Industrial baseline: structural features + core_copod"},
)
class VisionStructuralCOPOD(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_copod",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_knn",
    tags=("vision", "classical", "pipeline", "industrial", "neighbors"),
    metadata={"description": "Industrial baseline: structural features + core_knn"},
)
class VisionStructuralKNN(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_knn",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_lof",
    tags=("vision", "classical", "pipeline", "industrial", "neighbors", "density"),
    metadata={"description": "Industrial baseline: structural features + core_lof"},
)
class VisionStructuralLOF(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_lof",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_extra_trees_density",
    tags=("vision", "classical", "pipeline", "industrial", "trees", "density"),
    metadata={"description": "Industrial baseline: structural features + core_extra_trees_density"},
)
class VisionStructuralExtraTreesDensity(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_extra_trees_density",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_mcd",
    tags=("vision", "classical", "pipeline", "industrial", "gaussian", "robust"),
    metadata={"description": "Industrial baseline: structural features + core_mcd (robust covariance)"},
)
class VisionStructuralMCD(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_mcd",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_pca_md",
    tags=("vision", "classical", "pipeline", "industrial", "pca", "distance"),
    metadata={"description": "Industrial baseline: structural features + core_pca_md (subspace MD)"},
)
class VisionStructuralPCAMD(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_pca_md",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_iforest",
    tags=("vision", "classical", "pipeline", "industrial", "baseline"),
    metadata={"description": "Industrial baseline: structural features + core_iforest"},
)
class VisionStructuralIForest(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_iforest",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_lid",
    tags=("vision", "classical", "pipeline", "industrial", "neighbors"),
    metadata={"description": "Structural features + core_lid (kNN distance statistic)"},
)
class VisionStructuralLID(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_lid",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_structural_mst_outlier",
    tags=("vision", "classical", "pipeline", "industrial", "graph"),
    metadata={"description": "Structural features + core_mst_outlier (MST baseline)"},
)
class VisionStructuralMSTOutlier(VisionFeaturePipeline):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor: Any = None,
        feature_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        fx = feature_extractor
        if fx is None:
            mk = dict(feature_kwargs or {})
            fx = _default_structural_extractor(max_size=int(mk.get("max_size", 512)))

        super().__init__(
            core_detector="core_mst_outlier",
            core_kwargs=dict(core_kwargs or {}),
            feature_extractor=fx,
            contamination=float(contamination),
        )


@register_model(
    "vision_resnet18_ecod",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "fast"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_ecod",
    },
)
class VisionResNet18ECOD(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_ecod",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_copod",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "fast", "parameter-free"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_copod",
    },
)
class VisionResNet18COPOD(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_copod",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_iforest",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "baseline"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_iforest",
    },
)
class VisionResNet18IForest(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_iforest",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_knn",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "neighbors"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_knn",
    },
)
class VisionResNet18KNN(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool="avg",
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_knn",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_ecod",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "fast"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_ecod",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptECOD(VisionEmbeddingCoreDetector):
    """TorchScript embeddings + ECOD (industrial deployment-friendly)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        # Advanced override knobs:
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_ecod")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_ecod",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_copod",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "fast", "parameter-free"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_copod",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptCOPOD(VisionEmbeddingCoreDetector):
    """TorchScript embeddings + COPOD (parameter-free, industrial deployment-friendly)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        # Advanced override knobs:
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_copod")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_copod",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_iforest",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "baseline"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_iforest",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptIForest(VisionEmbeddingCoreDetector):
    """TorchScript embeddings + Isolation Forest."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_iforest")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_iforest",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_knn_cosine",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "neighbors"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_knn_cosine",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptKNNCosine(VisionEmbeddingCoreDetector):
    """TorchScript embeddings + cosine kNN (embedding-friendly distance baseline)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_knn_cosine")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_knn_cosine",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_knn_cosine_calibrated",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "neighbors", "calibration"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_knn_cosine_calibrated",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptKNNCosineCalibrated(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required for vision_torchscript_knn_cosine_calibrated"
                )
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_knn_cosine_calibrated",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_cosine_mahalanobis",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "distance", "gaussian", "cosine"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_cosine_mahalanobis",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptCosineMahalanobis(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_cosine_mahalanobis")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_cosine_mahalanobis",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_lid",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "neighbors", "lid"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_lid",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptLID(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_lid")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_lid",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_lof",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "neighbors", "density", "lof"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_lof",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptLOF(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_lof")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_lof",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_mcd",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "gaussian", "robust", "mcd"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_mcd (robust covariance)",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptMCD(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_mcd")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_mcd",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_mst_outlier",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "graph", "mst"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_mst_outlier",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptMSTOutlier(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_mst_outlier")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_mst_outlier",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_pca_md",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "pca", "distance"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_pca_md",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptPCAMD(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_pca_md")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_pca_md",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_extra_trees_density",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "trees", "density"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_extra_trees_density",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptExtraTreesDensity(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required for vision_torchscript_extra_trees_density"
                )
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_extra_trees_density",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_torchscript_oddoneout",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "torchscript", "neighbors", "oddoneout"),
    metadata={
        "description": "Industrial baseline: TorchScript embeddings + core_oddoneout",
        "requires_checkpoint": True,
    },
)
class VisionTorchscriptOddOneOut(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        embedding_extractor: str | Any = "torchscript_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchscript_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_torchscript_oddoneout")
            embedding_kwargs = _default_torchscript_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_oddoneout",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_ecod",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "fast"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_ecod",
        "requires_checkpoint": True,
    },
)
class VisionONNXECOD(VisionEmbeddingCoreDetector):
    """ONNX Runtime embeddings + ECOD (industrial deployment-friendly)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        # Advanced override knobs:
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_ecod")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_ecod",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_copod",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "fast", "parameter-free"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_copod",
        "requires_checkpoint": True,
    },
)
class VisionONNXCOPOD(VisionEmbeddingCoreDetector):
    """ONNX Runtime embeddings + COPOD (parameter-free)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_copod")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_copod",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_iforest",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "baseline"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_iforest",
        "requires_checkpoint": True,
    },
)
class VisionONNXIForest(VisionEmbeddingCoreDetector):
    """ONNX Runtime embeddings + Isolation Forest."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_iforest")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_iforest",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_knn_cosine",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "neighbors"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_knn_cosine",
        "requires_checkpoint": True,
    },
)
class VisionONNXKNNCosine(VisionEmbeddingCoreDetector):
    """ONNX Runtime embeddings + cosine kNN (embedding-friendly distance baseline)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_knn_cosine")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_knn_cosine",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_knn_cosine_calibrated",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "neighbors", "calibration"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_knn_cosine_calibrated",
        "requires_checkpoint": True,
    },
)
class VisionONNXKNNCosineCalibrated(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required for vision_onnx_knn_cosine_calibrated"
                )
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_knn_cosine_calibrated",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_cosine_mahalanobis",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "distance", "gaussian", "cosine"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_cosine_mahalanobis",
        "requires_checkpoint": True,
    },
)
class VisionONNXCosineMahalanobis(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_cosine_mahalanobis")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_cosine_mahalanobis",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_lid",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "neighbors", "lid"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_lid",
        "requires_checkpoint": True,
    },
)
class VisionONNXLID(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_lid")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_lid",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_lof",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "neighbors", "density", "lof"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_lof",
        "requires_checkpoint": True,
    },
)
class VisionONNXLOF(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_lof")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_lof",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_mcd",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "gaussian", "robust", "mcd"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_mcd",
        "requires_checkpoint": True,
    },
)
class VisionONNXMCD(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_mcd")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_mcd",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_mst_outlier",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "graph", "mst"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_mst_outlier",
        "requires_checkpoint": True,
    },
)
class VisionONNXMSTOutlier(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_mst_outlier")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_mst_outlier",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_pca_md",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "pca", "distance"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_pca_md",
        "requires_checkpoint": True,
    },
)
class VisionONNXPcaMD(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_pca_md")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_pca_md",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_extra_trees_density",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "trees", "density"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_extra_trees_density",
        "requires_checkpoint": True,
    },
)
class VisionONNXExtraTreesDensity(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required for vision_onnx_extra_trees_density"
                )
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_extra_trees_density",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_onnx_oddoneout",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "onnx", "neighbors", "oddoneout"),
    metadata={
        "description": "Industrial baseline: ONNX Runtime embeddings + core_oddoneout",
        "requires_checkpoint": True,
    },
)
class VisionONNXOddOneOut(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: str | None = None,
        providers: list[str] | None = None,
        embedding_extractor: str | Any = "onnx_embed",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "onnx_embed":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for vision_onnx_oddoneout")
            embedding_kwargs = _default_onnx_embedding_extractor(
                checkpoint_path=str(checkpoint_path),
                device=str(device),
                batch_size=int(batch_size),
                image_size=int(image_size),
                cache_dir=cache_dir,
                providers=providers,
            )["kwargs"]

        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_oddoneout",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_knn_cosine",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "neighbors", "cosine"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_knn_cosine",
    },
)
class VisionResNet18KNNCosine(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_knn_cosine",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_knn_cosine_calibrated",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "neighbors", "cosine", "calibration"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_knn_cosine_calibrated",
    },
)
class VisionResNet18KNNCosineCalibrated(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_knn_cosine_calibrated",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_cosine_mahalanobis",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "distance", "gaussian", "cosine"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_cosine_mahalanobis",
    },
)
class VisionResNet18CosineMahalanobis(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_cosine_mahalanobis",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_lid",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "neighbors", "lid"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_lid",
    },
)
class VisionResNet18LID(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_lid",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_lof",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "neighbors", "density", "lof"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_lof",
    },
)
class VisionResNet18LOF(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_lof",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_mcd",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "gaussian", "robust", "mcd"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_mcd (robust covariance)",
    },
)
class VisionResNet18MCD(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_mcd",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_mst_outlier",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "graph", "mst"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_mst_outlier",
    },
)
class VisionResNet18MSTOutlier(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_mst_outlier",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_pca_md",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "pca", "distance"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_pca_md",
    },
)
class VisionResNet18PCAMD(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_pca_md",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_extra_trees_density",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "trees", "density"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_extra_trees_density",
    },
)
class VisionResNet18ExtraTreesDensity(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_extra_trees_density",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_oddoneout",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "neighbors", "oddoneout"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_oddoneout",
    },
)
class VisionResNet18OddOneOut(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_oddoneout",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_mahalanobis_shrinkage",
    tags=("vision", "classical", "pipeline", "industrial", "embeddings", "distance", "gaussian", "shrinkage"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_mahalanobis_shrinkage",
    },
)
class VisionResNet18MahalanobisShrinkage(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        pool: str = "avg",
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool=str(pool),
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_mahalanobis_shrinkage",
            core_kwargs=dict(core_kwargs or {}),
        )


@register_model(
    "vision_resnet18_torch_ae",
    tags=("vision", "deep", "pipeline", "industrial", "embeddings", "reconstruction"),
    metadata={
        "description": "Industrial baseline: resnet18 embeddings (safe) + core_torch_autoencoder",
    },
)
class VisionResNet18TorchAE(VisionEmbeddingCoreDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        embedding_extractor: str | Any = "torchvision_backbone",
        embedding_kwargs: Mapping[str, Any] | None = None,
        core_kwargs: Mapping[str, Any] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        if embedding_kwargs is None and str(embedding_extractor) == "torchvision_backbone":
            embedding_kwargs = _default_torchvision_embedding_extractor(
                backbone=str(backbone),
                pretrained=bool(pretrained),
                pool="avg",
                device=str(device),
                image_size=int(image_size),
            )["kwargs"]
        super().__init__(
            contamination=float(contamination),
            embedding_extractor=embedding_extractor,
            embedding_kwargs=embedding_kwargs,
            core_detector="core_torch_autoencoder",
            core_kwargs=dict(core_kwargs or {}),
        )


__all__ = [
    "VisionStructuralECOD",
    "VisionStructuralCOPOD",
    "VisionStructuralKNN",
    "VisionStructuralLOF",
    "VisionStructuralExtraTreesDensity",
    "VisionStructuralMCD",
    "VisionStructuralPCAMD",
    "VisionStructuralIForest",
    "VisionStructuralLID",
    "VisionStructuralMSTOutlier",
    "VisionResNet18ECOD",
    "VisionResNet18COPOD",
    "VisionResNet18IForest",
    "VisionResNet18KNN",
    "VisionResNet18KNNCosine",
    "VisionResNet18KNNCosineCalibrated",
    "VisionResNet18CosineMahalanobis",
    "VisionResNet18LID",
    "VisionResNet18LOF",
    "VisionResNet18MCD",
    "VisionResNet18MSTOutlier",
    "VisionResNet18PCAMD",
    "VisionResNet18ExtraTreesDensity",
    "VisionResNet18OddOneOut",
    "VisionResNet18MahalanobisShrinkage",
    "VisionResNet18TorchAE",
    "VisionTorchscriptECOD",
    "VisionTorchscriptCOPOD",
    "VisionTorchscriptIForest",
    "VisionTorchscriptKNNCosine",
    "VisionTorchscriptKNNCosineCalibrated",
    "VisionTorchscriptCosineMahalanobis",
    "VisionTorchscriptLID",
    "VisionTorchscriptLOF",
    "VisionTorchscriptMCD",
    "VisionTorchscriptMSTOutlier",
    "VisionTorchscriptPCAMD",
    "VisionTorchscriptExtraTreesDensity",
    "VisionTorchscriptOddOneOut",
]
