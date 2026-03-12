from __future__ import annotations

from typing import Any, Iterable, Mapping

import pyimgano.cli_output as cli_output
import pyimgano.pyim_section_views as pyim_section_views


def _emit_json(payload: Any) -> int:
    return cli_output.emit_json(payload)


def _print_named_block(title: str, lines: Iterable[str]) -> None:
    print(title)
    for line in lines:
        print(line)
    print()


def _render_named_items(title: str, items: Iterable[Any]) -> None:
    _print_named_block(title, [str(item) for item in items])


def _render_counted_sections(
    title: str,
    items: Iterable[Any],
) -> None:
    _print_named_block(
        title,
        [f"{item.name} ({item.model_count}): {item.description}" for item in items],
    )


def _render_metadata_contract(title: str, items: Iterable[Any]) -> None:
    lines = []
    for item in items:
        rule = str(item.requirement)
        if item.required_when:
            rule = f"{rule}; {item.required_when}"
        lines.append(f"{item.name} [{item.source} / {rule}]: {item.description}")
    _print_named_block(title, lines)


def _render_preprocessing(title: str, items: Iterable[Any]) -> None:
    lines = []
    for item in items:
        suffix = " [deployable]" if bool(item.deployable) else ""
        lines.append(f"{item.name}{suffix}: {item.description}")
    _print_named_block(title, lines)


def _render_recipes(title: str, items: Iterable[Mapping[str, Any]]) -> None:
    lines = []
    for info in items:
        meta = info.get("metadata", {}) or {}
        desc = str(meta.get("description", "")).strip()
        suffix = f": {desc}" if desc else ""
        lines.append(f"{info.get('name')}{suffix}")
    _print_named_block(title, lines)


def _render_datasets(title: str, items: Iterable[Any]) -> None:
    lines = []
    for item in items:
        req = " (category required)" if bool(item.requires_category) else ""
        lines.append(f"{item.name}{req}: {item.description}")
    _print_named_block(title, lines)


def _render_named_items_section(title: str, items: Iterable[Any]) -> None:
    _render_named_items(title, items)


def _render_counted_section(title: str, items: Iterable[Any]) -> None:
    _render_counted_sections(title, items)


_TEXT_RENDERERS = {
    "named-items": _render_named_items_section,
    "counted-sections": _render_counted_section,
    "metadata-contract": _render_metadata_contract,
    "preprocessing": _render_preprocessing,
    "recipes": _render_recipes,
    "datasets": _render_datasets,
}


def _render_text_section(section: pyim_section_views.PyimTextSectionView) -> None:
    _TEXT_RENDERERS[section.render_kind](section.title, section.value)


def _render_all_sections(payload: pyim_section_views.PyimListPayloadLike) -> None:
    for section in pyim_section_views.iter_pyim_all_text_section_views(payload):
        _render_text_section(section)


def emit_pyim_list_payload(
    payload: pyim_section_views.PyimListPayloadLike,
    *,
    list_kind: str,
    json_output: bool,
) -> int:
    list_kind_value = str(list_kind)

    if bool(json_output):
        return _emit_json(pyim_section_views.resolve_pyim_json_payload(payload, list_kind_value))

    if list_kind_value == "all":
        _render_all_sections(payload)
        return 0

    try:
        _render_text_section(pyim_section_views.resolve_pyim_text_section_view(payload, list_kind_value))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported pyim list kind: {list_kind_value}") from exc
    return 0


__all__ = ["emit_pyim_list_payload"]
