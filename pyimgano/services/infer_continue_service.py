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
                processed += 1
        except Exception:
            # Fallback: isolate per-input failures when a chunk run fails.
            for j, input_path in enumerate(chunk):
                idx = int(start + j)
                try:
                    one_run = run_inference_fn(
                        detector=request.detector,
                        inputs=[input_path],
                        include_maps=bool(request.include_maps),
                        postprocess=request.postprocess,
                        batch_size=None,
                        amp=bool(request.amp),
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
