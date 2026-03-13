from __future__ import annotations

from typing import Any, Mapping

from pyimgano.workbench.preflight_types import IssueSeverity, PreflightIssue


def build_preflight_issue(
    code: Any,
    severity: IssueSeverity,
    message: Any,
    *,
    context: Mapping[str, Any] | None = None,
) -> PreflightIssue:
    return PreflightIssue(code=str(code), severity=severity, message=str(message), context=context)


__all__ = ["build_preflight_issue"]
