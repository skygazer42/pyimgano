from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pyimgano.pyim_list_spec import ALL_PAYLOAD_FIELDS, CORE_PAYLOAD_FIELDS, get_pyim_list_kind_spec


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in list(value)]


@dataclass(frozen=True)
class PyimModelFacetSummary:
    name: str
    description: str
    model_count: int
    tags: list[str] = field(default_factory=list)
    sample_models: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PyimModelFacetSummary":
        return cls(
            name=str(payload["name"]),
            description=str(payload["description"]),
            model_count=int(payload["model_count"]),
            tags=_list_of_str(payload.get("tags", [])),
            sample_models=_list_of_str(payload.get("sample_models", [])),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "model_count": int(self.model_count),
            "sample_models": list(self.sample_models),
        }


@dataclass(frozen=True)
class PyimYearSummary:
    name: str
    label: str
    description: str
    model_count: int
    sample_models: list[str] = field(default_factory=list)
    year: int | None = None
    year_start: int | None = None
    year_end: int | None = None
    _has_year: bool = field(default=False, repr=False, compare=False)
    _has_year_start: bool = field(default=False, repr=False, compare=False)
    _has_year_end: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PyimYearSummary":
        raw_year = payload.get("year")
        raw_year_start = payload.get("year_start")
        raw_year_end = payload.get("year_end")
        return cls(
            name=str(payload["name"]),
            label=str(payload["label"]),
            description=str(payload["description"]),
            model_count=int(payload["model_count"]),
            sample_models=_list_of_str(payload.get("sample_models", [])),
            year=(None if raw_year is None else int(raw_year)),
            year_start=(None if raw_year_start is None else int(raw_year_start)),
            year_end=(None if raw_year_end is None else int(raw_year_end)),
            _has_year=("year" in payload),
            _has_year_start=("year_start" in payload),
            _has_year_end=("year_end" in payload),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "model_count": int(self.model_count),
            "sample_models": list(self.sample_models),
        }
        if self._has_year:
            payload["year"] = self.year
        if self._has_year_start:
            payload["year_start"] = self.year_start
        if self._has_year_end:
            payload["year_end"] = self.year_end
        return payload


@dataclass(frozen=True)
class PyimMetadataContractField:
    name: str
    source: str
    requirement: str
    description: str
    value_type: str
    required_when: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PyimMetadataContractField":
        return cls(
            name=str(payload["name"]),
            source=str(payload["source"]),
            requirement=str(payload["requirement"]),
            description=str(payload["description"]),
            value_type=str(payload["value_type"]),
            required_when=(
                None if payload.get("required_when") is None else str(payload.get("required_when"))
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "requirement": self.requirement,
            "description": self.description,
            "value_type": self.value_type,
            "required_when": self.required_when,
        }


@dataclass(frozen=True)
class PyimPreprocessingSchemeSummary:
    name: str
    description: str
    deployable: bool
    tags: list[str] = field(default_factory=list)
    config_key: str | None = None
    payload: dict[str, Any] | None = None
    entrypoint: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PyimPreprocessingSchemeSummary":
        raw_payload = payload.get("payload")
        return cls(
            name=str(payload["name"]),
            description=str(payload["description"]),
            deployable=bool(payload.get("deployable", False)),
            tags=_list_of_str(payload.get("tags", [])),
            config_key=(None if payload.get("config_key") is None else str(payload.get("config_key"))),
            payload=(None if raw_payload is None else dict(raw_payload)),
            entrypoint=(None if payload.get("entrypoint") is None else str(payload.get("entrypoint"))),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "deployable": bool(self.deployable),
            "tags": list(self.tags),
        }
        if self.config_key is not None:
            payload["config_key"] = self.config_key
        if self.payload is not None:
            payload["payload"] = dict(self.payload)
        if self.entrypoint is not None:
            payload["entrypoint"] = self.entrypoint
        return payload


@dataclass(frozen=True)
class PyimDatasetSummary:
    name: str
    description: str
    requires_category: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PyimDatasetSummary":
        return cls(
            name=str(payload["name"]),
            description=str(payload["description"]),
            requires_category=bool(payload.get("requires_category", False)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "requires_category": bool(self.requires_category),
        }


def _coerce_model_facet(value: PyimModelFacetSummary | Mapping[str, Any]) -> PyimModelFacetSummary:
    if isinstance(value, PyimModelFacetSummary):
        return value
    return PyimModelFacetSummary.from_mapping(value)


def _coerce_year_summary(value: PyimYearSummary | Mapping[str, Any]) -> PyimYearSummary:
    if isinstance(value, PyimYearSummary):
        return value
    return PyimYearSummary.from_mapping(value)


def _coerce_metadata_field(
    value: PyimMetadataContractField | Mapping[str, Any],
) -> PyimMetadataContractField:
    if isinstance(value, PyimMetadataContractField):
        return value
    return PyimMetadataContractField.from_mapping(value)


def _coerce_preprocessing_scheme(
    value: PyimPreprocessingSchemeSummary | Mapping[str, Any],
) -> PyimPreprocessingSchemeSummary:
    if isinstance(value, PyimPreprocessingSchemeSummary):
        return value
    return PyimPreprocessingSchemeSummary.from_mapping(value)


def _coerce_dataset_summary(value: PyimDatasetSummary | Mapping[str, Any]) -> PyimDatasetSummary:
    if isinstance(value, PyimDatasetSummary):
        return value
    return PyimDatasetSummary.from_mapping(value)


def _serialize_json_section_value(value: Any) -> Any:
    if isinstance(value, list):
        return [item.to_payload() if hasattr(item, "to_payload") else item for item in value]
    return value


def _resolve_request_flag(
    explicit_value: bool | None,
    *,
    legacy_default: bool,
    list_kind_default: bool | None,
) -> bool:
    if explicit_value is not None:
        return bool(explicit_value)
    if list_kind_default is not None:
        return bool(list_kind_default)
    return bool(legacy_default)


@dataclass(frozen=True)
class PyimListRequest:
    tags: list[str] | None = None
    family: str | None = None
    algorithm_type: str | None = None
    year: str | int | None = None
    deployable_only: bool = False
    list_kind: str | None = None
    include_core_sections: bool | None = None
    include_recipes: bool | None = None
    include_datasets: bool | None = None

    def __post_init__(self) -> None:
        raw_include_core_sections = self.include_core_sections
        raw_include_recipes = self.include_recipes
        raw_include_datasets = self.include_datasets
        list_kind_value = None if self.list_kind is None else str(self.list_kind)
        kind_spec = None if list_kind_value is None else get_pyim_list_kind_spec(list_kind_value)

        object.__setattr__(self, "list_kind", list_kind_value)
        object.__setattr__(
            self,
            "include_core_sections",
            _resolve_request_flag(
                raw_include_core_sections,
                legacy_default=True,
                list_kind_default=(None if kind_spec is None else kind_spec.include_core_sections),
            ),
        )
        object.__setattr__(
            self,
            "include_recipes",
            _resolve_request_flag(
                raw_include_recipes,
                legacy_default=False,
                list_kind_default=(None if kind_spec is None else kind_spec.include_recipes),
            ),
        )
        object.__setattr__(
            self,
            "include_datasets",
            _resolve_request_flag(
                raw_include_datasets,
                legacy_default=False,
                list_kind_default=(None if kind_spec is None else kind_spec.include_datasets),
            ),
        )

    def requested_payload_fields(self) -> tuple[str, ...]:
        requested_fields: set[str] = set()

        if self.list_kind is None:
            if bool(self.include_core_sections):
                requested_fields.update(CORE_PAYLOAD_FIELDS)
            if bool(self.include_recipes):
                requested_fields.add("recipes")
            if bool(self.include_datasets):
                requested_fields.add("datasets")
            return tuple(field_name for field_name in ALL_PAYLOAD_FIELDS if field_name in requested_fields)

        requested_fields.update(get_pyim_list_kind_spec(self.list_kind).request_fields)
        kind_spec = get_pyim_list_kind_spec(self.list_kind)
        if not kind_spec.include_core_sections and bool(self.include_core_sections):
            requested_fields.update(CORE_PAYLOAD_FIELDS)
        if not kind_spec.include_recipes and bool(self.include_recipes):
            requested_fields.add("recipes")
        if not kind_spec.include_datasets and bool(self.include_datasets):
            requested_fields.add("datasets")

        return tuple(field_name for field_name in ALL_PAYLOAD_FIELDS if field_name in requested_fields)


@dataclass(frozen=True)
class PyimListPayload:
    models: list[str] = field(default_factory=list)
    families: list[PyimModelFacetSummary | Mapping[str, Any]] = field(default_factory=list)
    types: list[PyimModelFacetSummary | Mapping[str, Any]] = field(default_factory=list)
    years: list[PyimYearSummary | Mapping[str, Any]] = field(default_factory=list)
    metadata_contract: list[PyimMetadataContractField | Mapping[str, Any]] = field(
        default_factory=list
    )
    preprocessing: list[PyimPreprocessingSchemeSummary | Mapping[str, Any]] = field(
        default_factory=list
    )
    features: list[str] = field(default_factory=list)
    model_presets: list[str] = field(default_factory=list)
    model_preset_infos: list[dict[str, Any]] = field(default_factory=list)
    defects_presets: list[str] = field(default_factory=list)
    recipes: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[PyimDatasetSummary | Mapping[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "families", [_coerce_model_facet(item) for item in self.families])
        object.__setattr__(self, "types", [_coerce_model_facet(item) for item in self.types])
        object.__setattr__(self, "years", [_coerce_year_summary(item) for item in self.years])
        object.__setattr__(
            self,
            "metadata_contract",
            [_coerce_metadata_field(item) for item in self.metadata_contract],
        )
        object.__setattr__(
            self,
            "preprocessing",
            [_coerce_preprocessing_scheme(item) for item in self.preprocessing],
        )
        object.__setattr__(self, "datasets", [_coerce_dataset_summary(item) for item in self.datasets])

    def get_section_value(self, list_kind: str) -> Any:
        spec = get_pyim_list_kind_spec(list_kind)
        if spec.text_field is None:
            raise ValueError(f"Pyim list kind does not expose a single text section: {spec.name}")
        return getattr(self, spec.text_field)

    def to_json_payload(self, list_kind: str) -> Any:
        spec = get_pyim_list_kind_spec(list_kind)
        if spec.name == "all":
            return self.to_all_json_payload()
        if spec.json_field is None:
            raise ValueError(f"Pyim list kind does not expose a single JSON section: {spec.name}")
        return _serialize_json_section_value(getattr(self, spec.json_field))

    def to_all_json_payload(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "families": [item.to_payload() for item in self.families],
            "types": [item.to_payload() for item in self.types],
            "years": [item.to_payload() for item in self.years],
            "metadata_contract": [item.to_payload() for item in self.metadata_contract],
            "preprocessing": [item.to_payload() for item in self.preprocessing],
            "features": self.features,
            "model_presets": self.model_presets,
            "defects_presets": self.defects_presets,
            "recipes": self.recipes,
            "datasets": [item.to_payload() for item in self.datasets],
        }


__all__ = [
    "PyimDatasetSummary",
    "PyimListPayload",
    "PyimListRequest",
    "PyimMetadataContractField",
    "PyimModelFacetSummary",
    "PyimPreprocessingSchemeSummary",
    "PyimYearSummary",
]
