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


__all__ = [
    "format_readiness_lines",
    "format_require_extras_line",
    "format_suite_check_line",
]
