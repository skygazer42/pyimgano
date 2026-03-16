from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO


@dataclass(frozen=True)
class InferOutputTargetsRequest:
    save_jsonl: str | None = None
    defects_enabled: bool = False
    defects_regions_jsonl: str | None = None


@dataclass(frozen=True)
class InferOutputTargets:
    output_file: TextIO | None = None
    regions_file: TextIO | None = None


@dataclass(frozen=True)
class InferOutputWriteRequest:
    record: dict[str, Any]
    regions_payload: dict[str, Any] | None = None
    output_file: TextIO | None = None
    regions_file: TextIO | None = None
    flush_every: int = 0
    output_written: int = 0
    regions_written: int = 0


@dataclass(frozen=True)
class InferOutputWriteResult:
    output_written: int
    regions_written: int


@dataclass(frozen=True)
class InferErrorRecordRequest:
    index: int
    input_path: str
    exc: object
    stage: str


def _error_type_name(exc: object) -> str:
    return type(exc).__name__


def _error_message(exc: object) -> str:
    return str(exc)


def open_infer_output_targets(request: InferOutputTargetsRequest) -> InferOutputTargets:
    output_file: TextIO | None = None
    if request.save_jsonl is not None:
        output_path = Path(request.save_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", encoding="utf-8")

    regions_file: TextIO | None = None
    if request.defects_regions_jsonl is not None:
        if not bool(request.defects_enabled):
            raise ValueError("--defects-regions-jsonl requires --defects")
        regions_path = Path(request.defects_regions_jsonl)
        regions_path.parent.mkdir(parents=True, exist_ok=True)
        regions_file = regions_path.open("w", encoding="utf-8")

    return InferOutputTargets(
        output_file=output_file,
        regions_file=regions_file,
    )


def write_infer_output_payloads(
    request: InferOutputWriteRequest,
    *,
    print_fn: Callable[[str], None] | None = None,
) -> InferOutputWriteResult:
    output_written = int(request.output_written)
    regions_written = int(request.regions_written)
    flush_every = int(request.flush_every)

    if request.regions_payload is not None and request.regions_file is not None:
        request.regions_file.write(json.dumps(request.regions_payload, sort_keys=True))
        request.regions_file.write("\n")
        regions_written += 1
        if flush_every > 0 and (regions_written % flush_every == 0):
            request.regions_file.flush()

    line = json.dumps(request.record, sort_keys=True)
    if request.output_file is not None:
        request.output_file.write(line)
        request.output_file.write("\n")
        output_written += 1
        if flush_every > 0 and (output_written % flush_every == 0):
            request.output_file.flush()
    else:
        printer = print if print_fn is None else print_fn
        printer(line)

    return InferOutputWriteResult(
        output_written=output_written,
        regions_written=regions_written,
    )


def build_infer_error_record(request: InferErrorRecordRequest) -> dict[str, Any]:
    return {
        "status": "error",
        "index": int(request.index),
        "input": str(request.input_path),
        "error": {
            "type": _error_type_name(request.exc),
            "message": _error_message(request.exc),
            "stage": str(request.stage),
        },
    }


__all__ = [
    "InferErrorRecordRequest",
    "InferOutputTargets",
    "InferOutputTargetsRequest",
    "InferOutputWriteRequest",
    "InferOutputWriteResult",
    "build_infer_error_record",
    "open_infer_output_targets",
    "write_infer_output_payloads",
]
