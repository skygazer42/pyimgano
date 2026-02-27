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
    device: str = "cpu",
    image_size: int = 224,
) -> dict[str, Any]:
    # Safe defaults: `pretrained=False` avoids implicit weight downloads.
    return {
        "name": "torchvision_backbone",
        "kwargs": {
            "backbone": str(backbone),
            "pretrained": bool(pretrained),
            "device": str(device),
            "image_size": int(image_size),
        },
    }


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
    "VisionStructuralIForest",
    "VisionStructuralLID",
    "VisionStructuralMSTOutlier",
    "VisionResNet18ECOD",
    "VisionResNet18IForest",
    "VisionResNet18KNN",
    "VisionResNet18TorchAE",
]
