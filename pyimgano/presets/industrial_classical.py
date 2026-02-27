"""Industrial classical presets (JSON-ready).

These presets are intended to be copied into config files or used from the CLI.

Design goals:
- deterministic by default
- no implicit network downloads
- works well on small/medium industrial datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelPreset:
    """A JSON-friendly model preset.

    - `model` must be a registered model name.
    - `kwargs` must be JSON-serializable (dict/lists/scalars).
    """

    name: str
    model: str
    kwargs: Mapping[str, Any]
    description: str
    optional: bool = False


def _structural_feature_extractor(*, max_size: int = 512) -> dict[str, Any]:
    return {"name": "structural", "kwargs": {"max_size": int(max_size)}}


def _torchvision_embedding_extractor(
    *,
    backbone: str = "resnet18",
    pretrained: bool = False,
    pool: str = "avg",
) -> dict[str, Any]:
    # `pretrained=False` is critical to avoid implicit weight downloads.
    return {
        "name": "torchvision_backbone",
        "kwargs": {
            "backbone": str(backbone),
            "pretrained": bool(pretrained),
            "pool": str(pool),
        },
    }


INDUSTRIAL_CLASSICAL_PRESETS: dict[str, ModelPreset] = {
    "industrial-structural-ecod": ModelPreset(
        name="industrial-structural-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _structural_feature_extractor(max_size=512),
            "core_detector": "core_ecod",
        },
        description="Fast industrial baseline: structural features -> ECOD.",
    ),
    "industrial-structural-iforest": ModelPreset(
        name="industrial-structural-iforest",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _structural_feature_extractor(max_size=512),
            "core_detector": "core_iforest",
            "core_kwargs": {"n_estimators": 200, "n_jobs": 1},
        },
        description="Robust baseline: structural features -> IsolationForest.",
    ),
    "industrial-structural-mst": ModelPreset(
        name="industrial-structural-mst",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _structural_feature_extractor(max_size=512),
            "core_detector": "core_mst_outlier",
            "core_kwargs": {"metric": "euclidean", "score_mode": "max"},
        },
        description="Graph baseline: structural features -> MST outlier score.",
    ),
    "industrial-embed-ecod": ModelPreset(
        name="industrial-embed-ecod",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "torchvision_backbone",
            "embedding_kwargs": _torchvision_embedding_extractor(backbone="resnet18")["kwargs"],
            "core_detector": "core_ecod",
        },
        description="Embeddings route: torchvision backbone embeddings -> ECOD.",
    ),
    "industrial-embed-lid": ModelPreset(
        name="industrial-embed-lid",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "torchvision_backbone",
            "embedding_kwargs": _torchvision_embedding_extractor(backbone="resnet18")["kwargs"],
            "core_detector": "core_lid",
            "core_kwargs": {"n_neighbors": 20},
        },
        description="Embeddings route: torchvision backbone embeddings -> LID (kNN statistic).",
    ),
    "industrial-openclip-knn": ModelPreset(
        name="industrial-openclip-knn",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "openclip_embed",
            # `pretrained=None` to avoid implicit downloads.
            "embedding_kwargs": {"model_name": "ViT-B-32", "pretrained": None, "device": "cpu"},
            "core_detector": "core_knn",
            "core_kwargs": {"method": "largest", "n_neighbors": 5},
        },
        description="Optional: OpenCLIP embeddings -> kNN distance (requires open_clip_torch).",
        optional=True,
    ),
}


def list_industrial_classical_presets() -> list[str]:
    return sorted(INDUSTRIAL_CLASSICAL_PRESETS.keys())


def get_industrial_classical_preset(name: str) -> ModelPreset:
    key = str(name).strip()
    if key not in INDUSTRIAL_CLASSICAL_PRESETS:
        available = ", ".join(list_industrial_classical_presets()) or "<none>"
        raise KeyError(f"Unknown preset {name!r}. Available: {available}")
    return INDUSTRIAL_CLASSICAL_PRESETS[key]


__all__ = [
    "ModelPreset",
    "INDUSTRIAL_CLASSICAL_PRESETS",
    "list_industrial_classical_presets",
    "get_industrial_classical_preset",
]

