from __future__ import annotations

from typing import Any, Mapping

import pyimgano.cli_output as cli_output


def _audit_has_issues(payload: Mapping[str, Any]) -> bool:
    summary = payload["summary"]
    return bool(
        summary["models_with_required_issues"]
        or summary["models_with_recommended_issues"]
        or summary["models_with_invalid_fields"]
    )


def emit_pyim_audit_payload(payload: Mapping[str, Any], *, json_output: bool) -> int:
    has_issues = _audit_has_issues(payload)

    if bool(json_output):
        return cli_output.emit_json(payload) if not has_issues else (cli_output.emit_json(payload) or 1)

    print("Metadata Audit")
    print(
        f"required={payload['summary']['models_with_required_issues']} "
        f"recommended={payload['summary']['models_with_recommended_issues']} "
        f"invalid={payload['summary']['models_with_invalid_fields']}"
    )
    return 1 if has_issues else 0


__all__ = ["emit_pyim_audit_payload"]
