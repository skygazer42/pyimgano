"""Unified discovery helpers for models, families, and preprocessing schemes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ModelFamily:
    name: str
    description: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ModelType:
    name: str
    description: str
    tags: tuple[str, ...]


_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        name="neighbors",
        description="Nearest-neighbor and locality-based anomaly detectors.",
        tags=("neighbors",),
    ),
    ModelFamily(
        name="density",
        description="Density, support-estimation, and occupancy-style detectors.",
        tags=("density",),
    ),
    ModelFamily(
        name="graph",
        description="Graph and topology-aware anomaly detectors.",
        tags=("graph",),
    ),
    ModelFamily(
        name="clustering",
        description="Clustering and prototype-style anomaly detectors.",
        tags=("clustering",),
    ),
    ModelFamily(
        name="ensemble",
        description="Ensemble-style model combinations and selector methods.",
        tags=("ensemble",),
    ),
    ModelFamily(
        name="gaussian",
        description="Gaussian, covariance, and Mahalanobis-oriented families.",
        tags=("gaussian",),
    ),
    ModelFamily(
        name="reconstruction",
        description="Reconstruction-based anomaly detectors.",
        tags=("reconstruction",),
    ),
    ModelFamily(
        name="distillation",
        description="Teacher-student and distillation-oriented detectors.",
        tags=("distillation",),
    ),
    ModelFamily(
        name="template",
        description="Template, correlation, and direct image-comparison baselines.",
        tags=("template",),
    ),
    ModelFamily(
        name="patchcore",
        description="PatchCore-style memory bank and patch embedding detectors.",
        tags=("patchcore",),
    ),
    ModelFamily(
        name="memory_bank",
        description="Memory-bank style embedding retrieval methods.",
        tags=("memory_bank",),
    ),
    ModelFamily(
        name="autoencoder",
        description="Autoencoder and latent-reconstruction detector families.",
        tags=("autoencoder",),
    ),
    ModelFamily(
        name="clip",
        description="CLIP/OpenCLIP and multimodal foundation-model variants.",
        tags=("clip",),
    ),
    ModelFamily(
        name="backend",
        description="Optional backend wrappers and external integration families.",
        tags=("backend",),
    ),
    ModelFamily(
        name="pipeline",
        description="Composed pipelines and higher-level wrapper detectors.",
        tags=("pipeline",),
    ),
)

_MODEL_TYPES: tuple[ModelType, ...] = (
    ModelType(
        name="classical-core",
        description="Classical core detectors operating on feature matrices or tabular vectors.",
        tags=("classical", "core"),
    ),
    ModelType(
        name="classical-vision",
        description="Classical image-oriented detectors and wrappers working directly on visual inputs.",
        tags=("classical", "vision"),
    ),
    ModelType(
        name="deep-vision",
        description="Deep-learning vision detectors, including reconstruction and representation models.",
        tags=("deep", "vision"),
    ),
    ModelType(
        name="pixel-map",
        description="Models that produce pixel-level anomaly maps for localization and defects export.",
        tags=("pixel_map",),
    ),
    ModelType(
        name="embedding-models",
        description="Embedding-centric models and pipelines built around learned representations.",
        tags=("embeddings",),
    ),
    ModelType(
        name="industrial-pipelines",
        description="Industrial wrappers and pipeline-style models targeting deployable inspection flows.",
        tags=("industrial", "pipeline"),
    ),
    ModelType(
        name="self-supervised",
        description="Self-supervised anomaly detectors trained from pretext or representation tasks.",
        tags=("self-supervised",),
    ),
    ModelType(
        name="weakly-supervised",
        description="Weakly supervised detectors that use limited anomaly labels or ranking signals.",
        tags=("weakly-supervised",),
    ),
    ModelType(
        name="reconstruction",
        description="Reconstruction-oriented models such as autoencoders and decoder-based methods.",
        tags=("reconstruction",),
    ),
    ModelType(
        name="flow-based",
        description="Normalizing-flow and density-transform based anomaly detectors.",
        tags=("flow",),
    ),
    ModelType(
        name="gan-based",
        description="GAN and adversarially trained anomaly detectors.",
        tags=("gan",),
    ),
    ModelType(
        name="distillation",
        description="Student-teacher and distillation-based anomaly detectors.",
        tags=("distillation",),
    ),
    ModelType(
        name="one-class-svm",
        description="One-class SVM style detectors and wrappers.",
        tags=("svm", "one-class"),
    ),
    ModelType(
        name="neighbor-based",
        description="Nearest-neighbor, local-neighborhood, and kNN anomaly detectors.",
        tags=("neighbors",),
    ),
    ModelType(
        name="graph-based",
        description="Graph, MST, and random-walk oriented anomaly detectors.",
        tags=("graph",),
    ),
    ModelType(
        name="clustering-based",
        description="Clustering and cluster-deviation anomaly detectors.",
        tags=("clustering",),
    ),
    ModelType(
        name="gaussian-distance",
        description="Gaussian, covariance, and Mahalanobis-style distance detectors.",
        tags=("gaussian",),
    ),
    ModelType(
        name="density-estimation",
        description="Density estimation methods such as KDE and related region-density detectors.",
        tags=("density",),
    ),
    ModelType(
        name="template-matching",
        description="Template, SSIM, NCC, and reference-comparison detectors.",
        tags=("template",),
    ),
    ModelType(
        name="memory-bank",
        description="Memory-bank retrieval methods such as PatchCore-style pipelines.",
        tags=("memory_bank",),
    ),
    ModelType(
        name="multimodal-clip",
        description="CLIP/OpenCLIP and multimodal foundation-model based detectors.",
        tags=("clip",),
    ),
    ModelType(
        name="backend-wrappers",
        description="External backend wrappers and registry shims for optional runtimes or integrations.",
        tags=("backend",),
    ),
)

_TIMELINE_START_YEAR = 2001


def _parse_tag_values(items: Optional[Iterable[str]]) -> list[str]:
    out: list[str] = []
    if items is None:
        return out
    for item in items:
        for piece in str(item).split(","):
            tag = piece.strip()
            if tag:
                out.append(tag)
    return out


def _ensure_models_loaded() -> None:
    import pyimgano.models  # noqa: F401


def _ensure_features_loaded() -> None:
    import pyimgano.features  # noqa: F401


def _normalize_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _all_registry_tags() -> set[str]:
    _ensure_models_loaded()
    from pyimgano.models.registry import MODEL_REGISTRY

    return {
        str(tag).strip().lower()
        for model_name in MODEL_REGISTRY.available()
        for tag in MODEL_REGISTRY.info(model_name).tags
    }


def _coerce_metadata_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_year_filter(value: str | int) -> str | int:
    """Resolve a year selector into a canonical timeline bucket or exact year."""

    if isinstance(value, bool):
        raise KeyError(f"Unknown model year filter: {value!r}")
    if isinstance(value, int):
        return int(value)

    raw = str(value).strip().lower().replace("_", "-")
    if not raw:
        raise KeyError(f"Unknown model year filter: {value!r}")
    if raw in {"pre-2001", "pre2001", "before-2001", "<=2000"}:
        return "pre-2001"
    if raw in {"unknown", "unannotated", "missing"}:
        return "unknown"
    try:
        return int(raw)
    except ValueError as exc:
        raise KeyError(f"Unknown model year filter: {value!r}") from exc


def _matches_year_filter(metadata_year: Any, selector: str | int) -> bool:
    year = _coerce_metadata_year(metadata_year)
    if selector == "unknown":
        return year is None
    if selector == "pre-2001":
        return year is not None and year < _TIMELINE_START_YEAR
    return year == int(selector)


def list_model_names(
    *,
    tags: Optional[Iterable[str]] = None,
    family: str | None = None,
    algorithm_type: str | None = None,
    year: str | int | None = None,
) -> list[str]:
    """List model names with optional tag/family filtering."""

    from pyimgano.models.registry import MODEL_REGISTRY, list_models

    _ensure_models_loaded()
    merged_tags = _parse_tag_values(tags)
    if family:
        merged_tags.extend(resolve_family_tags(family))
    if algorithm_type:
        merged_tags.extend(resolve_type_tags(algorithm_type))
    if not merged_tags:
        names = list_models()
    else:
        names = list_models(tags=merged_tags)
    if year is None:
        return names

    selector = resolve_year_filter(year)
    return [
        name for name in names if _matches_year_filter(MODEL_REGISTRY.info(name).metadata.get("year"), selector)
    ]


def list_feature_names(*, tags: Optional[Iterable[str]] = None) -> list[str]:
    """List feature extractor names with optional tag filtering."""

    from pyimgano.features import list_feature_extractors

    _ensure_features_loaded()
    parsed = _parse_tag_values(tags)
    if not parsed:
        return list_feature_extractors()
    return list_feature_extractors(tags=parsed)


def list_model_families() -> list[dict[str, Any]]:
    """Return curated model family summaries backed by registry tags."""

    from pyimgano.models.registry import MODEL_REGISTRY

    _ensure_models_loaded()
    out: list[dict[str, Any]] = []
    for family in _MODEL_FAMILIES:
        names = MODEL_REGISTRY.available(tags=family.tags)
        if not names:
            continue
        out.append(
            {
                "name": family.name,
                "description": family.description,
                "tags": list(family.tags),
                "model_count": len(names),
                "sample_models": names[:8],
            }
        )
    return sorted(out, key=lambda item: (-int(item["model_count"]), str(item["name"])))


def list_model_types() -> list[dict[str, Any]]:
    """Return curated high-level model type summaries backed by registry tags."""

    from pyimgano.models.registry import MODEL_REGISTRY

    _ensure_models_loaded()
    out: list[dict[str, Any]] = []
    for model_type in _MODEL_TYPES:
        names = MODEL_REGISTRY.available(tags=model_type.tags)
        if not names:
            continue
        out.append(
            {
                "name": model_type.name,
                "description": model_type.description,
                "tags": list(model_type.tags),
                "model_count": len(names),
                "sample_models": names[:8],
            }
        )
    return sorted(out, key=lambda item: (-int(item["model_count"]), str(item["name"])))


def list_model_years() -> list[dict[str, Any]]:
    """Return a publication-year timeline for models with year metadata."""

    from pyimgano.models.registry import MODEL_REGISTRY

    _ensure_models_loaded()
    names_by_year: dict[int, list[str]] = {}
    unknown: list[str] = []
    for model_name in MODEL_REGISTRY.available():
        year = _coerce_metadata_year(MODEL_REGISTRY.info(model_name).metadata.get("year"))
        if year is None:
            unknown.append(model_name)
            continue
        names_by_year.setdefault(int(year), []).append(model_name)

    out: list[dict[str, Any]] = []
    early = sorted(
        model_name
        for year, names in names_by_year.items()
        if int(year) < _TIMELINE_START_YEAR
        for model_name in names
    )
    if early:
        out.append(
            {
                "name": "pre-2001",
                "label": "Pre-2001",
                "description": "Models annotated with publication years earlier than 2001.",
                "year_start": None,
                "year_end": _TIMELINE_START_YEAR - 1,
                "model_count": len(early),
                "sample_models": early[:8],
            }
        )

    timeline_end = max(date.today().year, max(names_by_year, default=_TIMELINE_START_YEAR))
    for year in range(_TIMELINE_START_YEAR, timeline_end + 1):
        names = sorted(names_by_year.get(year, []))
        out.append(
            {
                "name": str(year),
                "label": str(year),
                "description": f"Models annotated with publication year {year}.",
                "year": int(year),
                "model_count": len(names),
                "sample_models": names[:8],
            }
        )

    if unknown:
        out.append(
            {
                "name": "unknown",
                "label": "Unknown",
                "description": "Models without an annotated publication year in registry metadata.",
                "year": None,
                "model_count": len(unknown),
                "sample_models": sorted(unknown)[:8],
            }
        )
    return out


def resolve_family_tags(name: str) -> tuple[str, ...]:
    """Resolve a family name into the tag constraints used for filtering."""

    key = _normalize_key(name)
    for family in _MODEL_FAMILIES:
        if family.name == key:
            return family.tags

    # Allow power users to treat a raw registry tag as a family filter.
    all_tags = _all_registry_tags()
    raw = key.replace("_", "-")
    if key in all_tags:
        return (key,)
    if raw in all_tags:
        return (raw,)
    raise KeyError(f"Unknown model family/tag: {name!r}")


def resolve_type_tags(name: str) -> tuple[str, ...]:
    """Resolve a high-level model type into the tag constraints used for filtering."""

    key = _normalize_key(name)
    for model_type in _MODEL_TYPES:
        if _normalize_key(model_type.name) == key:
            return model_type.tags

    all_tags = _all_registry_tags()
    raw = key.replace("_", "-")
    if key in all_tags:
        return (key,)
    if raw in all_tags:
        return (raw,)
    raise KeyError(f"Unknown model type/tag: {name!r}")


def list_preprocessing_schemes(*, deployable_only: bool = False) -> list[dict[str, Any]]:
    """Return JSON-friendly preprocessing scheme summaries."""

    from pyimgano.preprocessing.catalog import list_preprocessing_schemes as _list_schemes

    out: list[dict[str, Any]] = []
    for item in _list_schemes(deployable_only=deployable_only):
        payload = asdict(item)
        if payload.get("payload", None) is None:
            payload.pop("payload", None)
        if payload.get("config_key", None) is None:
            payload.pop("config_key", None)
        if payload.get("entrypoint", None) is None:
            payload.pop("entrypoint", None)
        out.append(payload)
    return out


__all__ = [
    "ModelFamily",
    "ModelType",
    "list_feature_names",
    "list_model_families",
    "list_model_names",
    "list_model_types",
    "list_model_years",
    "list_preprocessing_schemes",
    "resolve_family_tags",
    "resolve_type_tags",
    "resolve_year_filter",
]
