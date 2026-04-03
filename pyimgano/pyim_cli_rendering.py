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


def _render_selection_context(selection_payload: Mapping[str, Any]) -> None:
    context = dict(selection_payload.get("selection_context", {}) or {})
    if not context:
        return

    lines = [
        f"objective={context.get('objective')}",
        f"selection_profile={context.get('selection_profile')}",
        f"topk={context.get('topk')}",
    ]
    summary = dict(selection_payload.get("selection_profile_summary", {}) or {})
    description = str(summary.get("description", "")).strip()
    if description:
        lines.append(f"description={description}")
    _print_named_block("Selection Context", lines)


def _render_starter_picks(selection_payload: Mapping[str, Any]) -> None:
    picks = list(selection_payload.get("starter_picks", []) or [])
    if not picks:
        return

    lines: list[str] = []
    for item in picks:
        if not isinstance(item, Mapping):
            continue
        line = f"{item.get('name')}: {item.get('summary')}"
        details: list[str] = []
        runtime = str(item.get("tested_runtime", "")).strip()
        if runtime:
            details.append(f"runtime={runtime}")
        if "supports_pixel_map" in item:
            details.append(f"pixel_map={'yes' if bool(item.get('supports_pixel_map')) else 'no'}")
        family = [str(entry) for entry in item.get("deployment_family", []) or [] if str(entry)]
        if family:
            details.append(f"family={','.join(family)}")
        required = list(item.get("required_extras", []) or [])
        if details:
            line += f" [{' | '.join(details)}]"
        if required:
            line += f" [extras: {', '.join(str(extra) for extra in required)}]"
        raw_install_hint = item.get("install_hint", None)
        install_hint = "" if raw_install_hint is None else str(raw_install_hint).strip()
        if install_hint:
            line += f" [install: {install_hint}]"
        lines.append(line)
    if lines:
        _print_named_block("Starter Picks", lines)


def _render_suggested_commands(selection_payload: Mapping[str, Any]) -> None:
    commands = [str(item) for item in selection_payload.get("suggested_commands", []) or [] if str(item).strip()]
    if commands:
        _print_named_block("Suggested Commands", commands)


def _render_goal_context(goal_payload: Mapping[str, Any]) -> None:
    context = dict(goal_payload.get("goal_context", {}) or {})
    if not context:
        return

    lines = [f"goal={context.get('goal')}"]
    for key in ("title", "summary", "objective", "selection_profile", "topk"):
        value = context.get(key)
        if value is not None:
            lines.append(f"{key}={value}")
    _print_named_block("Goal Context", lines)


def _render_goal_picks(goal_payload: Mapping[str, Any]) -> None:
    picks = dict(goal_payload.get("goal_picks", {}) or {})
    if not picks:
        return

    lines: list[str] = []
    for item in picks.get("models", []) or []:
        if not isinstance(item, Mapping):
            continue
        line = f"model={item.get('name')}"
        summary = str(item.get("summary", "")).strip()
        if summary:
            line += f" summary={summary}"
        why = str(item.get("why_this_pick", "")).strip()
        if why:
            line += f" why={why}"
        raw_install_hint = item.get("install_hint", None)
        install_hint = "" if raw_install_hint is None else str(raw_install_hint).strip()
        if install_hint:
            line += f" install={install_hint}"
        lines.append(line)
    for item in picks.get("recipes", []) or []:
        if not isinstance(item, Mapping):
            continue
        line = f"recipe={item.get('name')}"
        summary = str(item.get("summary", "")).strip()
        if summary:
            line += f" summary={summary}"
        config_path = str(item.get("config_path", "")).strip()
        if config_path:
            line += f" config={config_path}"
        runtime_profile = str(item.get("runtime_profile", "")).strip()
        if runtime_profile:
            line += f" profile={runtime_profile}"
        install_hint = str(item.get("install_hint", "")).strip()
        if install_hint:
            line += f" install={install_hint}"
        recipe_info_command = str(item.get("recipe_info_command", "")).strip()
        if recipe_info_command:
            line += f" inspect={recipe_info_command}"
        dry_run_command = str(item.get("dry_run_command", "")).strip()
        if dry_run_command:
            line += f" dry_run={dry_run_command}"
        preflight_command = str(item.get("preflight_command", "")).strip()
        if preflight_command:
            line += f" preflight={preflight_command}"
        recipe_run_command = str(item.get("recipe_run_command", "")).strip()
        if recipe_run_command:
            line += f" run={recipe_run_command}"
        lines.append(line)
    for item in picks.get("datasets", []) or []:
        if not isinstance(item, Mapping):
            continue
        line = f"dataset={item.get('name')}"
        summary = str(item.get("summary", "")).strip()
        if summary:
            line += f" summary={summary}"
        lines.append(line)
    if lines:
        _print_named_block("Goal Picks", lines)


def _render_all_sections(payload: pyim_section_views.PyimListPayloadLike) -> None:
    for section in pyim_section_views.iter_pyim_all_text_section_views(payload):
        _render_text_section(section)


def emit_pyim_list_payload(
    payload: pyim_section_views.PyimListPayloadLike,
    *,
    list_kind: str,
    json_output: bool,
    selection_payload: Mapping[str, Any] | None = None,
    goal_payload: Mapping[str, Any] | None = None,
) -> int:
    list_kind_value = str(list_kind)

    if bool(json_output):
        json_payload = pyim_section_views.resolve_pyim_json_payload(payload, list_kind_value)
        if goal_payload is not None and isinstance(json_payload, dict):
            return _emit_json({**json_payload, **dict(goal_payload)})
        if selection_payload is not None and list_kind_value == "models":
            return _emit_json({"items": json_payload, **dict(selection_payload)})
        return _emit_json(json_payload)

    if list_kind_value == "all":
        if goal_payload is not None:
            _render_goal_context(goal_payload)
            _render_goal_picks(goal_payload)
            _render_suggested_commands(goal_payload)
        _render_all_sections(payload)
        return 0

    try:
        if selection_payload is not None and list_kind_value == "models":
            _render_selection_context(selection_payload)
            _render_starter_picks(selection_payload)
            _render_suggested_commands(selection_payload)
        _render_text_section(
            pyim_section_views.resolve_pyim_text_section_view(payload, list_kind_value)
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported pyim list kind: {list_kind_value}") from exc
    return 0


__all__ = ["emit_pyim_list_payload"]
