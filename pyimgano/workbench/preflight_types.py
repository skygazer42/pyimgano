from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

IssueSeverity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    severity: IssueSeverity
    message: str
    context: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PreflightReport:
    dataset: str
    category: str
    summary: dict[str, Any]
    issues: list[PreflightIssue]


__all__ = ["IssueSeverity", "PreflightIssue", "PreflightReport"]
