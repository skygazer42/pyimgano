from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Union

from pyimgano.pyim_contracts import PyimListPayload
from pyimgano.pyim_list_spec import PYIM_ALL_TEXT_LIST_KINDS, get_pyim_list_kind_spec

PyimListPayloadLike = Union[PyimListPayload, Mapping[str, Any]]


@dataclass(frozen=True)
class PyimTextSectionView:
    list_kind: str
    title: str
    render_kind: str
    value: Any


def _coerce_payload(payload: PyimListPayloadLike) -> PyimListPayload:
    if isinstance(payload, PyimListPayload):
        return payload
    return PyimListPayload(
        models=list(payload.get("models", [])),
        families=list(payload.get("families", [])),
        types=list(payload.get("types", [])),
        years=list(payload.get("years", [])),
        metadata_contract=list(payload.get("metadata_contract", [])),
        preprocessing=list(payload.get("preprocessing", [])),
        features=list(payload.get("features", [])),
        model_presets=list(payload.get("model_presets", [])),
        model_preset_infos=list(payload.get("model_preset_infos", [])),
        defects_presets=list(payload.get("defects_presets", [])),
        recipes=list(payload.get("recipes", [])),
        datasets=list(payload.get("datasets", [])),
    )


def resolve_pyim_json_payload(payload: PyimListPayloadLike, list_kind: str) -> Any:
    return _coerce_payload(payload).to_json_payload(str(list_kind))


def resolve_pyim_text_section_view(
    payload: PyimListPayloadLike,
    list_kind: str,
) -> PyimTextSectionView:
    payload_value = _coerce_payload(payload)
    spec = get_pyim_list_kind_spec(str(list_kind))
    if spec.text_title is None or spec.text_render_kind is None:
        raise ValueError(f"Pyim list kind does not expose a text section: {spec.name}")

    return PyimTextSectionView(
        list_kind=spec.name,
        title=spec.text_title,
        render_kind=spec.text_render_kind,
        value=payload_value.get_section_value(spec.name),
    )


def iter_pyim_all_text_section_views(
    payload: PyimListPayloadLike,
) -> Iterator[PyimTextSectionView]:
    payload_value = _coerce_payload(payload)

    for list_kind in PYIM_ALL_TEXT_LIST_KINDS:
        spec = get_pyim_list_kind_spec(list_kind)
        section_value = payload_value.get_section_value(spec.name)
        if not spec.render_when_empty_in_all and not section_value:
            continue

        yield PyimTextSectionView(
            list_kind=spec.name,
            title=str(spec.text_title),
            render_kind=str(spec.text_render_kind),
            value=section_value,
        )


__all__ = [
    "PyimListPayloadLike",
    "PyimTextSectionView",
    "iter_pyim_all_text_section_views",
    "resolve_pyim_json_payload",
    "resolve_pyim_text_section_view",
]
