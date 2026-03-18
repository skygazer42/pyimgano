from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ContinueOnErrorInferRequest:
    detector: Any
    inputs: Sequence[str]
    include_maps: bool = False
    include_confidence: bool = False
    reject_confidence_below: float | None = None
    reject_label: int | None = None
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
    triage_summary: dict[str, Any] | None = None


def _build_triage_summary(
    *,
    processed: int,
    errors: int,
    stop_early: bool,
    inputs_total: int,
    error_stages: dict[str, int],
    fallback_used: bool,
) -> dict[str, Any]:
    return {
        "ok": max(0, int(processed) - int(errors)),
        "remaining": max(0, int(inputs_total) - int(processed)),
        "error_stages": {str(k): int(v) for k, v in sorted(error_stages.items())},
        "fallback_used": bool(fallback_used),
        "stop_reason": ("max_errors" if bool(stop_early) else "completed"),
    }


def _run_inference_chunk(
    *,
    run_inference_fn: Callable[..., Any],
    request: ContinueOnErrorInferRequest,
    inputs: Sequence[str],
) -> Any:
    return run_inference_fn(
        detector=request.detector,
        inputs=inputs,
        include_maps=bool(request.include_maps),
        include_confidence=bool(
            request.include_confidence or request.reject_confidence_below is not None
        ),
        reject_confidence_below=(
            float(request.reject_confidence_below)
            if request.reject_confidence_below is not None
            else None
        ),
        reject_label=(int(request.reject_label) if request.reject_label is not None else None),
        postprocess=request.postprocess,
        batch_size=None,
        amp=bool(request.amp),
    )


def _process_chunk_records(
    *,
    start: int,
    chunk: Sequence[str],
    chunk_records: Sequence[Any],
    process_ok_result: Callable[..., None],
    handle_error: Callable[..., None],
) -> tuple[int, int]:
    processed = 0
    errors = 0
    for j, result in enumerate(chunk_records):
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
        processed += 1
    return processed, errors


def _run_chunk_with_fallback(
    *,
    start: int,
    chunk: Sequence[str],
    request: ContinueOnErrorInferRequest,
    run_inference_fn: Callable[..., Any],
    process_ok_result: Callable[..., None],
    handle_error: Callable[..., None],
    timing_seconds: float,
    processed: int,
    errors: int,
) -> tuple[float, int, int, bool]:
    stop_early = False
    for j, input_path in enumerate(chunk):
        idx = int(start + j)
        try:
            one_run = _run_inference_chunk(
                run_inference_fn=run_inference_fn,
                request=request,
                inputs=[input_path],
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
        processed += 1

        if int(request.max_errors) > 0 and int(errors) >= int(request.max_errors):
            stop_early = True
            break
    return timing_seconds, processed, errors, stop_early


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
    error_stages: dict[str, int] = {"artifacts": 0, "infer": 0}
    fallback_used = False

    def _handle_error_with_summary(
        *,
        index: int,
        input_path: str,
        exc: Exception,
        stage: str,
    ) -> None:
        stage_key = str(stage)
        error_stages[stage_key] = int(error_stages.get(stage_key, 0)) + 1
        handle_error(
            index=int(index),
            input_path=str(input_path),
            exc=exc,
            stage=stage_key,
        )

    for start in range(0, len(inputs), int(chunk_size)):
        chunk = inputs[start : start + int(chunk_size)]
        try:
            chunk_run = _run_inference_chunk(
                run_inference_fn=run_inference_fn,
                request=request,
                inputs=chunk,
            )
            chunk_results = list(chunk_run.records)
            timing_seconds += float(chunk_run.timing_seconds)
            chunk_processed, chunk_errors = _process_chunk_records(
                start=start,
                chunk=chunk,
                chunk_records=chunk_results,
                process_ok_result=process_ok_result,
                handle_error=_handle_error_with_summary,
            )
            processed += int(chunk_processed)
            errors += int(chunk_errors)
        except Exception:
            # Fallback: isolate per-input failures when a chunk run fails.
            fallback_used = True
            timing_seconds, processed, errors, stop_early = _run_chunk_with_fallback(
                start=start,
                chunk=chunk,
                request=request,
                run_inference_fn=run_inference_fn,
                process_ok_result=process_ok_result,
                handle_error=_handle_error_with_summary,
                timing_seconds=timing_seconds,
                processed=processed,
                errors=errors,
            )
        if bool(stop_early):
            break

    return ContinueOnErrorInferResult(
        processed=int(processed),
        errors=int(errors),
        timing_seconds=float(timing_seconds),
        stop_early=bool(stop_early),
        triage_summary=_build_triage_summary(
            processed=int(processed),
            errors=int(errors),
            stop_early=bool(stop_early),
            inputs_total=int(len(inputs)),
            error_stages=error_stages,
            fallback_used=bool(fallback_used),
        ),
    )


__all__ = [
    "ContinueOnErrorInferRequest",
    "ContinueOnErrorInferResult",
    "run_continue_on_error_inference",
]
