from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ContinueOnErrorInferRequest:
    detector: Any
    inputs: Sequence[str]
    include_maps: bool = False
    postprocess: Any | None = None
    batch_size: int | None = None
    amp: bool = False
    max_errors: int = 0


@dataclass(frozen=True)
class ContinueOnErrorInferResult:
    processed: int
    errors: int
    timing_seconds: float
    stop_early: bool


def _has_reached_max_errors(*, errors: int, max_errors: int) -> bool:
    return int(max_errors) > 0 and int(errors) >= int(max_errors)


def _process_chunk_records(
    *,
    start: int,
    chunk: list[str],
    chunk_results: list[Any],
    process_ok_result: Callable[..., None],
    handle_error: Callable[..., None],
    errors: int,
    max_errors: int,
) -> tuple[int, int, bool]:
    processed = 0
    stop_early = False
    for j, result in enumerate(chunk_results):
        idx = int(start + j)
        try:
            process_ok_result(
                index=idx,
                input_path=str(chunk[j]),
                result=result,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort mode
            errors += 1
            handle_error(
                index=idx,
                input_path=str(chunk[j]),
                exc=exc,
                stage="artifacts",
            )
            stop_early = _has_reached_max_errors(errors=errors, max_errors=max_errors)
        processed += 1
        if stop_early:
            break
    return processed, errors, stop_early


def _run_single_input_fallback(
    *,
    start: int,
    chunk: list[str],
    detector: Any,
    include_maps: bool,
    postprocess: Any | None,
    amp: bool,
    max_errors: int,
    run_inference_fn: Callable[..., Any],
    process_ok_result: Callable[..., None],
    handle_error: Callable[..., None],
    errors: int,
) -> tuple[int, int, float, bool]:
    processed = 0
    timing_seconds = 0.0
    stop_early = False

    for j, input_path in enumerate(chunk):
        idx = int(start + j)
        try:
            one_run = run_inference_fn(
                detector=detector,
                inputs=[input_path],
                include_maps=bool(include_maps),
                postprocess=postprocess,
                batch_size=None,
                amp=bool(amp),
            )
            timing_seconds += float(one_run.timing_seconds)
            one = list(one_run.records)
            if len(one) != 1:
                raise RuntimeError("Internal error: expected 1 result for 1 input")
            process_ok_result(
                index=idx,
                input_path=str(input_path),
                result=one[0],
            )
        except Exception as exc:  # noqa: BLE001 - best-effort mode
            errors += 1
            handle_error(
                index=idx,
                input_path=str(input_path),
                exc=exc,
                stage="infer",
            )
            stop_early = _has_reached_max_errors(errors=errors, max_errors=max_errors)

        processed += 1
        if stop_early:
            break

    return processed, errors, timing_seconds, stop_early


def run_continue_on_error_inference(
    request: ContinueOnErrorInferRequest,
    *,
    process_ok_result: Callable[..., None],
    handle_error: Callable[..., None],
    run_inference_impl: Callable[..., Any] | None = None,
) -> ContinueOnErrorInferResult:
    from pyimgano.services.inference_service import run_inference

    run_inference_fn = run_inference if run_inference_impl is None else run_inference_impl

    chunk_size = int(request.batch_size) if request.batch_size is not None else 1
    processed = 0
    errors = 0
    timing_seconds = 0.0
    stop_early = False
    inputs = [str(item) for item in request.inputs]

    for start in range(0, len(inputs), int(chunk_size)):
        chunk = inputs[start : start + int(chunk_size)]
        try:
            chunk_run = run_inference_fn(
                detector=request.detector,
                inputs=chunk,
                include_maps=bool(request.include_maps),
                postprocess=request.postprocess,
                batch_size=None,
                amp=bool(request.amp),
            )
            chunk_results = list(chunk_run.records)
            timing_seconds += float(chunk_run.timing_seconds)
            chunk_processed, errors, stop_early = _process_chunk_records(
                start=int(start),
                chunk=chunk,
                chunk_results=chunk_results,
                process_ok_result=process_ok_result,
                handle_error=handle_error,
                errors=int(errors),
                max_errors=int(request.max_errors),
            )
            processed += int(chunk_processed)
        except Exception:
            # Fallback: isolate per-input failures when a chunk run fails.
            (
                chunk_processed,
                errors,
                chunk_timing,
                stop_early,
            ) = _run_single_input_fallback(
                start=int(start),
                chunk=chunk,
                detector=request.detector,
                include_maps=bool(request.include_maps),
                postprocess=request.postprocess,
                amp=bool(request.amp),
                max_errors=int(request.max_errors),
                run_inference_fn=run_inference_fn,
                process_ok_result=process_ok_result,
                handle_error=handle_error,
                errors=int(errors),
            )
            processed += int(chunk_processed)
            timing_seconds += float(chunk_timing)
        if bool(stop_early):
            break

    return ContinueOnErrorInferResult(
        processed=int(processed),
        errors=int(errors),
        timing_seconds=float(timing_seconds),
        stop_early=bool(stop_early),
    )


__all__ = [
    "ContinueOnErrorInferRequest",
    "ContinueOnErrorInferResult",
    "run_continue_on_error_inference",
]
