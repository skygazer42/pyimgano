from __future__ import annotations


def format_suite_check_line(*, suite_name: str, info: dict[str, object]) -> str:
    summary = dict(info.get("summary", {}))
    total = summary.get("total", None)
    runnable = summary.get("runnable", None)
    missing = list(summary.get("missing_extras", []) or [])
    suffix = ""
    if missing:
        suffix = f" (missing extras: {', '.join(str(item) for item in missing)})"
    return f"- {suite_name}: runnable {runnable}/{total}{suffix}"


def format_require_extras_line(req: dict[str, object]) -> str | None:
    required = list(req.get("required", []) or [])
    if not required:
        return None

    missing = list(req.get("missing", []) or [])
    if missing:
        hint = req.get("install_hint")
        msg = f"require_extras: MISSING ({', '.join(str(x) for x in missing)})"
        if hint:
            msg += f" -> {hint}"
        return msg
    return "require_extras: OK"


def format_readiness_lines(readiness: dict[str, object]) -> list[str]:
    lines = [
        "readiness:",
        f"- target_kind: {readiness.get('target_kind')}",
        f"- path: {readiness.get('path')}",
        f"- status: {readiness.get('status')}",
    ]
    issues = list(readiness.get("issues", []) or [])
    if issues:
        lines.append(f"- issues: {', '.join(str(item) for item in issues)}")
    return lines


def format_extra_recommendation_lines(recommendation: dict[str, object]) -> list[str]:
    lines = [
        "extras_recommendation:",
        f"- target_kind: {recommendation.get('target_kind')}",
        f"- target: {recommendation.get('target')}",
    ]
    workflow_stage = recommendation.get("workflow_stage")
    if workflow_stage:
        lines.append(f"- workflow_stage: {workflow_stage}")
    if "supports_pixel_map" in recommendation:
        lines.append(f"- supports_pixel_map: {recommendation.get('supports_pixel_map')}")
    tested_runtime = recommendation.get("tested_runtime")
    if tested_runtime:
        lines.append(f"- tested_runtime: {tested_runtime}")
    model_info_command = recommendation.get("model_info_command")
    if model_info_command:
        lines.append(f"- model_info_command: {model_info_command}")
    required = list(recommendation.get("required_extras", []) or [])
    if required:
        lines.append(f"- required_extras: {', '.join(str(item) for item in required)}")
    recommended = list(recommendation.get("recommended_extras", []) or [])
    if recommended:
        lines.append(f"- recommended_extras: {', '.join(str(item) for item in recommended)}")
    optional_baseline_count = recommendation.get("optional_baseline_count")
    if optional_baseline_count is not None:
        lines.append(f"- optional_baseline_count: {optional_baseline_count}")
    starter_configs = list(recommendation.get("starter_configs", []) or [])
    if starter_configs:
        lines.append(f"- starter_configs: {', '.join(str(item) for item in starter_configs)}")
    starter_list_command = recommendation.get("starter_list_command")
    if starter_list_command:
        lines.append(f"- starter_list_command: {starter_list_command}")
    starter_info_command = recommendation.get("starter_info_command")
    if starter_info_command:
        lines.append(f"- starter_info_command: {starter_info_command}")
    starter_run_command = recommendation.get("starter_run_command")
    if starter_run_command:
        lines.append(f"- starter_run_command: {starter_run_command}")
    recipe_list_command = recommendation.get("recipe_list_command")
    if recipe_list_command:
        lines.append(f"- recipe_list_command: {recipe_list_command}")
    recipe_info_command = recommendation.get("recipe_info_command")
    if recipe_info_command:
        lines.append(f"- recipe_info_command: {recipe_info_command}")
    recipe_run_command = recommendation.get("recipe_run_command")
    if recipe_run_command:
        lines.append(f"- recipe_run_command: {recipe_run_command}")
    suggested_commands = list(recommendation.get("suggested_commands", []) or [])
    if suggested_commands:
        lines.append(f"- suggested_commands: {'; '.join(str(item) for item in suggested_commands)}")
    next_step_commands = list(recommendation.get("next_step_commands", []) or [])
    if next_step_commands:
        lines.append(f"- next_step_commands: {'; '.join(str(item) for item in next_step_commands)}")
    artifact_hints = list(recommendation.get("artifact_hints", []) or [])
    if artifact_hints:
        lines.append(f"- artifact_hints: {'; '.join(str(item) for item in artifact_hints)}")
    missing = list(recommendation.get("missing_extras", []) or [])
    if missing:
        lines.append(f"- missing_extras: {', '.join(str(item) for item in missing)}")
    install_hint = recommendation.get("install_hint")
    if install_hint:
        lines.append(f"- install_hint: {install_hint}")
    notes = list(recommendation.get("notes", []) or [])
    if notes:
        lines.append(f"- notes: {'; '.join(str(item) for item in notes)}")
    return lines


def _format_workflow_profile_lines(profile: dict[str, object]) -> list[str]:
    lines = [
        "workflow_profile:",
        f"- profile: {profile.get('profile')}",
        f"- status: {profile.get('status')}",
    ]
    summary = profile.get("summary")
    if summary:
        lines.append(f"- summary: {summary}")
    required_modules = list(profile.get("required_modules", []) or [])
    if required_modules:
        lines.append(f"- required_modules: {', '.join(str(item) for item in required_modules)}")
    missing_modules = list(profile.get("missing_modules", []) or [])
    if missing_modules:
        lines.append(f"- missing_modules: {', '.join(str(item) for item in missing_modules)}")
    required_extras = list(profile.get("required_extras", []) or [])
    if required_extras:
        lines.append(f"- required_extras: {', '.join(str(item) for item in required_extras)}")
    recommended_extras = list(profile.get("recommended_extras", []) or [])
    if recommended_extras:
        lines.append(
            f"- recommended_extras: {', '.join(str(item) for item in recommended_extras)}"
        )
    starter_config = profile.get("starter_config")
    if starter_config:
        lines.append(f"- starter_config: {starter_config}")
    target_source = profile.get("target_source")
    if target_source:
        lines.append(f"- target_source: {target_source}")
    target_path = profile.get("target_path")
    if target_path:
        lines.append(f"- target_path: {target_path}")
    dataset_target = profile.get("dataset_target")
    if dataset_target:
        lines.append(f"- dataset_target: {dataset_target}")
    starter_commands = list(profile.get("starter_commands", []) or [])
    if starter_commands:
        lines.append(f"- starter_commands: {'; '.join(str(item) for item in starter_commands)}")
    next_step_commands = list(profile.get("next_step_commands", []) or [])
    if next_step_commands:
        lines.append(f"- next_step_commands: {'; '.join(str(item) for item in next_step_commands)}")
    artifact_hints = list(profile.get("artifact_hints", []) or [])
    if artifact_hints:
        lines.append(f"- artifact_hints: {'; '.join(str(item) for item in artifact_hints)}")
    issues = list(profile.get("issues", []) or [])
    if issues:
        lines.append(f"- issues: {', '.join(str(item) for item in issues)}")
    return lines


__all__ = [
    "format_extra_recommendation_lines",
    "format_readiness_lines",
    "format_require_extras_line",
    "format_suite_check_line",
]
