"""Application facade for pyim command orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pyimgano.pyim_audit_rendering as pyim_audit_rendering
import pyimgano.pyim_cli_options as pyim_cli_options
import pyimgano.pyim_cli_rendering as pyim_cli_rendering
import pyimgano.services.pyim_audit_service as pyim_audit_service
import pyimgano.services.pyim_service as pyim_service


@dataclass(frozen=True)
class PyimCommand:
    list_kind: str | None = None
    tags: list[str] | None = None
    family: str | None = None
    algorithm_type: str | None = None
    year: str | None = None
    deployable_only: bool = False
    objective: str | None = None
    selection_profile: str | None = None
    topk: int | None = None
    audit_metadata: bool = False
    json_output: bool = False


def _run_pyim_audit(command: PyimCommand) -> int:
    payload = pyim_audit_service.collect_pyim_audit_payload()
    return pyim_audit_rendering.emit_pyim_audit_payload(
        payload,
        json_output=bool(command.json_output),
    )


def _run_pyim_listing(command: PyimCommand) -> int:
    list_options = pyim_cli_options.resolve_pyim_list_options(
        list_kind=command.list_kind,
        tags=command.tags,
        family=command.family,
        algorithm_type=command.algorithm_type,
        year=command.year,
        deployable_only=bool(command.deployable_only),
        objective=command.objective,
        selection_profile=command.selection_profile,
        topk=command.topk,
    )
    request = list_options.to_request()
    payload = pyim_service.collect_pyim_listing_payload(request)
    selection_payload = None
    if (
        list_options.list_kind == "models"
        and (
            list_options.objective is not None
            or list_options.selection_profile is not None
            or list_options.topk is not None
        )
    ):
        selection_payload = pyim_service.collect_pyim_model_selection_payload(request)
    return pyim_cli_rendering.emit_pyim_list_payload(
        payload,
        list_kind=list_options.list_kind,
        json_output=bool(command.json_output),
        selection_payload=selection_payload,
    )


def run_pyim_command(command: PyimCommand) -> int:
    if bool(command.audit_metadata):
        return _run_pyim_audit(command)
    return _run_pyim_listing(command)


__all__ = [
    "PyimCommand",
    "run_pyim_command",
]
