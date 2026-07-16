from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Protocol, Sequence

from pyimgano.models.capabilities import compute_model_capabilities

_VALID_SUPERVISION_VALUES = {
    "unsupervised",
    "self-supervised",
    "weakly-supervised",
    "supervised",
    "few-shot",
    "zero-shot",
    "one-class",
    "backend-defined",
}

_VALID_PAPER_FIDELITY_VALUES = {
    "core-aligned",
    "paper-adaptation",
    "partial",
    "inspired",
    "external-backend",
    "not-applicable",
}

_SUPERVISION_TAG_PRIORITY: tuple[tuple[str, str], ...] = (
    ("self-supervised", "self-supervised"),
    ("weakly-supervised", "weakly-supervised"),
    ("zero-shot", "zero-shot"),
    ("few-shot", "few-shot"),
    ("one-class", "one-class"),
    ("supervised", "supervised"),
)


class _ModelEntryLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def constructor(self) -> Any: ...

    @property
    def tags(self) -> Sequence[str]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...


class _RegistryLike(Protocol):
    def available(self, *, tags: Iterable[str] | None = None) -> list[str]: ...

    def info(self, name: str) -> _ModelEntryLike: ...


@dataclass(frozen=True)
class MetadataFieldSpec:
    name: str
    source: str
    requirement: str
    description: str
    value_type: str
    required_when: str | None = None


_METADATA_CONTRACT: tuple[MetadataFieldSpec, ...] = (
    MetadataFieldSpec(
        name="paper_fidelity",
        source="registry_metadata",
        requirement="conditional",
        description=(
            "Relationship between a deep-model entry and the cited paper: core-aligned, "
            "paper-adaptation, partial, inspired, external-backend, or not-applicable."
        ),
        value_type="string enum",
        required_when="required for entries tagged deep",
    ),
    MetadataFieldSpec(
        name="implementation_status",
        source="registry_metadata",
        requirement="conditional",
        description="Concise implementation status shown to operators and auditors.",
        value_type="non-empty string",
        required_when="required for entries tagged deep",
    ),
    MetadataFieldSpec(
        name="paper",
        source="registry_metadata",
        requirement="conditional",
        description="Canonical paper title only when the entry implements or delegates that method.",
        value_type="non-empty string",
        required_when=(
            "required for core-aligned, paper-adaptation, and paper-specific external backends"
        ),
    ),
    MetadataFieldSpec(
        name="related_paper",
        source="registry_metadata",
        requirement="conditional",
        description=(
            "Paper that inspired a partial/proxy implementation; this deliberately does not "
            "claim a paper reproduction."
        ),
        value_type="non-empty string",
        required_when="required for partial and inspired deep-model entries",
    ),
    MetadataFieldSpec(
        name="year",
        source="registry_metadata",
        requirement="recommended",
        description="Publication year for the paper or algorithm family backing the model entry.",
        value_type="integer year",
    ),
    MetadataFieldSpec(
        name="family",
        source="derived_from_tags",
        requirement="required",
        description="One or more curated algorithm families resolved from registry tags.",
        value_type="list[str]",
    ),
    MetadataFieldSpec(
        name="type",
        source="derived_from_tags",
        requirement="required",
        description="One or more high-level model types resolved from registry tags.",
        value_type="list[str]",
    ),
    MetadataFieldSpec(
        name="supervision",
        source="registry_metadata",
        requirement="recommended",
        description="Training supervision regime for discovery and recommendation layers.",
        value_type="string enum",
    ),
    MetadataFieldSpec(
        name="supports_pixel_map",
        source="derived_from_capabilities",
        requirement="required",
        description="Whether the model exposes pixel-level anomaly map outputs.",
        value_type="boolean",
    ),
    MetadataFieldSpec(
        name="requires_checkpoint",
        source="derived_from_capabilities",
        requirement="required",
        description="Whether the model requires an external checkpoint or model artifact to run.",
        value_type="boolean",
    ),
    MetadataFieldSpec(
        name="weights_source",
        source="registry_metadata",
        requirement="conditional",
        description="Where recommended weights/checkpoints come from (official, upstream, local-only, etc.).",
        value_type="non-empty string",
        required_when="required when requires_checkpoint is true",
    ),
)


def metadata_contract_fields() -> list[dict[str, Any]]:
    """Return the structured metadata contract for registry models."""

    return [asdict(item) for item in _METADATA_CONTRACT]


def _normalize_key(text: str) -> str:
    return str(text).strip().lower().replace("-", "_")


def _coerce_year(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
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


def _match_family_names(tags: Sequence[str]) -> list[str]:
    from pyimgano.discovery import _MODEL_FAMILIES

    tag_set = {_normalize_key(tag) for tag in tags}
    out: list[str] = []
    for family in _MODEL_FAMILIES:
        if {_normalize_key(tag) for tag in family.tags}.issubset(tag_set):
            out.append(str(family.name))
    return out


def _match_type_names(tags: Sequence[str]) -> list[str]:
    from pyimgano.discovery import _MODEL_TYPES

    tag_set = {_normalize_key(tag) for tag in tags}
    out: list[str] = []
    for model_type in _MODEL_TYPES:
        if {_normalize_key(tag) for tag in model_type.tags}.issubset(tag_set):
            out.append(str(model_type.name))
    return out


def _infer_supervision(meta: Mapping[str, Any], tags: Sequence[str]) -> str | None:
    raw = meta.get("supervision", None)
    if raw is not None and str(raw).strip():
        return str(raw).strip()

    tag_set = {_normalize_key(tag) for tag in tags}
    for tag_name, resolved in _SUPERVISION_TAG_PRIORITY:
        if _normalize_key(tag_name) in tag_set:
            return resolved
    if "winclip" in tag_set or "prompt" in tag_set:
        return "zero-shot"
    if _normalize_key(str(meta.get("paper_fidelity", ""))) == "external_backend":
        return "backend-defined"
    if "deep" in tag_set:
        # The remaining native deep detectors expose the normal-only training
        # contract used throughout the registry.
        return "one-class"
    return None


def resolve_metadata_contract_payload(entry: _ModelEntryLike) -> dict[str, Any]:
    """Resolve raw + derived metadata fields for one registry entry."""

    meta = dict(entry.metadata)
    caps = compute_model_capabilities(entry)
    return {
        "paper_fidelity": meta.get("paper_fidelity"),
        "implementation_status": meta.get("implementation_status"),
        "paper": meta.get("paper"),
        "related_paper": meta.get("related_paper"),
        "year": _coerce_year(meta.get("year")),
        "family": _match_family_names(entry.tags),
        "type": _match_type_names(entry.tags),
        "supervision": _infer_supervision(meta, entry.tags),
        "supports_pixel_map": bool(caps.supports_pixel_map),
        "requires_checkpoint": bool(caps.requires_checkpoint),
        "weights_source": meta.get("weights_source"),
    }


def audit_metadata_contract(
    registry: _RegistryLike,
    *,
    names: Sequence[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Audit registry entries against the model metadata contract."""

    selected = list(names) if names is not None else list(registry.available())
    if limit is not None:
        selected = selected[: int(limit)]

    required_missing_by_model: dict[str, list[str]] = {}
    recommended_missing_by_model: dict[str, list[str]] = {}
    invalid_fields_by_model: dict[str, list[dict[str, Any]]] = {}

    current_year = int(date.today().year)

    for name in selected:
        entry = registry.info(name)
        payload = resolve_metadata_contract_payload(entry)
        tag_set = {_normalize_key(tag) for tag in entry.tags}
        is_deep = "deep" in tag_set

        required_missing: list[str] = []
        recommended_missing: list[str] = []
        invalid: list[dict[str, Any]] = []

        if not payload["family"]:
            required_missing.append("family")
        if not payload["type"]:
            required_missing.append("type")

        fidelity = payload["paper_fidelity"]
        implementation_status = payload["implementation_status"]
        fidelity_key = _normalize_key(str(fidelity)) if fidelity is not None else None

        if is_deep:
            if fidelity is None or not str(fidelity).strip():
                required_missing.append("paper_fidelity")
            elif fidelity_key not in {
                _normalize_key(item) for item in _VALID_PAPER_FIDELITY_VALUES
            }:
                invalid.append(
                    {
                        "field": "paper_fidelity",
                        "reason": "paper_fidelity must be one of the documented contract values",
                        "value": fidelity,
                    }
                )

            if implementation_status is None or not str(implementation_status).strip():
                required_missing.append("implementation_status")
            elif not isinstance(implementation_status, str):
                invalid.append(
                    {
                        "field": "implementation_status",
                        "reason": "implementation_status must be a non-empty string",
                    }
                )

        paper = payload["paper"]
        related_paper = payload["related_paper"]
        if paper is not None and not isinstance(paper, str):
            invalid.append({"field": "paper", "reason": "paper must be a non-empty string"})
        elif paper is not None and not str(paper).strip():
            invalid.append({"field": "paper", "reason": "paper must be a non-empty string"})

        if related_paper is not None and not isinstance(related_paper, str):
            invalid.append(
                {"field": "related_paper", "reason": "related_paper must be a non-empty string"}
            )
        elif related_paper is not None and not str(related_paper).strip():
            invalid.append(
                {"field": "related_paper", "reason": "related_paper must be a non-empty string"}
            )

        if is_deep and fidelity_key in {"core_aligned", "paper_adaptation"}:
            if paper is None or not str(paper).strip():
                required_missing.append("paper")
        elif is_deep and fidelity_key in {"partial", "inspired"}:
            if paper is not None:
                invalid.append(
                    {
                        "field": "paper",
                        "reason": (
                            "partial/inspired entries must use related_paper instead of claiming "
                            "a paper implementation"
                        ),
                    }
                )
            if related_paper is None or not str(related_paper).strip():
                required_missing.append("related_paper")
        elif is_deep and fidelity_key == "not_applicable":
            if paper is not None or related_paper is not None:
                invalid.append(
                    {
                        "field": "paper_fidelity",
                        "reason": "not-applicable entries cannot claim paper or related_paper",
                    }
                )

        year = payload["year"]
        if year is not None and not (1900 <= int(year) <= current_year):
            invalid.append(
                {
                    "field": "year",
                    "reason": f"year must be between 1900 and {current_year}",
                    "value": year,
                }
            )
        if (
            is_deep
            and fidelity_key
            in {
                "core_aligned",
                "paper_adaptation",
                "partial",
                "inspired",
            }
            and year is None
        ):
            required_missing.append("year")

        supervision = payload["supervision"]
        if supervision is None or not str(supervision).strip():
            recommended_missing.append("supervision")
        else:
            key = _normalize_key(str(supervision))
            if key not in {_normalize_key(item) for item in _VALID_SUPERVISION_VALUES}:
                invalid.append(
                    {
                        "field": "supervision",
                        "reason": "supervision must be one of the documented contract values",
                        "value": supervision,
                    }
                )

        if not isinstance(payload["supports_pixel_map"], bool):
            invalid.append(
                {
                    "field": "supports_pixel_map",
                    "reason": "supports_pixel_map must resolve to a boolean",
                    "value": payload["supports_pixel_map"],
                }
            )

        if not isinstance(payload["requires_checkpoint"], bool):
            invalid.append(
                {
                    "field": "requires_checkpoint",
                    "reason": "requires_checkpoint must resolve to a boolean",
                    "value": payload["requires_checkpoint"],
                }
            )

        weights_source = payload["weights_source"]
        if bool(payload["requires_checkpoint"]):
            if weights_source is None or not str(weights_source).strip():
                required_missing.append("weights_source")
        elif weights_source is not None and not str(weights_source).strip():
            invalid.append(
                {
                    "field": "weights_source",
                    "reason": "weights_source must be a non-empty string when provided",
                    "value": weights_source,
                }
            )

        if required_missing:
            required_missing_by_model[str(name)] = required_missing
        if recommended_missing:
            recommended_missing_by_model[str(name)] = recommended_missing
        if invalid:
            invalid_fields_by_model[str(name)] = invalid

    summary = {
        "total_models": len(selected),
        "models_with_required_issues": len(required_missing_by_model),
        "models_with_recommended_issues": len(recommended_missing_by_model),
        "models_with_invalid_fields": len(invalid_fields_by_model),
    }
    return {
        "summary": summary,
        "contract_fields": metadata_contract_fields(),
        "required_missing_by_model": required_missing_by_model,
        "recommended_missing_by_model": recommended_missing_by_model,
        "invalid_fields_by_model": invalid_fields_by_model,
    }
