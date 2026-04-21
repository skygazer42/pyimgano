from __future__ import annotations

import logging
import time
import tracemalloc
from typing import Any, Iterable, Mapping, Protocol


class TrainingCallback(Protocol):
    def on_train_start(self, *, context: Mapping[str, Any]) -> None: ...

    def on_epoch_end(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        context: Mapping[str, Any],
    ) -> None: ...

    def on_train_end(self, *, report: Mapping[str, Any], context: Mapping[str, Any]) -> None: ...

    def on_exception(self, *, error: Exception, context: Mapping[str, Any]) -> None: ...


def run_callback_hook(
    callbacks: Iterable[Any],
    *,
    hook: str,
    **kwargs: Any,
) -> list[str]:
    warnings: list[str] = []
    for callback in callbacks:
        method = getattr(callback, hook, None)
        if method is None:
            continue
        try:
            method(**kwargs)
        except Exception as exc:  # noqa: BLE001 - callbacks are user extensibility points
            callback_name = type(callback).__name__
            warnings.append(f"{callback_name}.{hook}: {type(exc).__name__}: {exc}")
    return warnings


class MetricsLoggingCallback:
    def __init__(self, *, logger_name: str = "pyimgano.training") -> None:
        self._logger = logging.getLogger(logger_name)

    def on_train_start(self, *, context: Mapping[str, Any]) -> None:
        self._logger.info(
            "training started: train_count=%s validation_count=%s seed=%s",
            context.get("train_count"),
            context.get("validation_count"),
            context.get("seed"),
        )

    def on_epoch_end(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        context: Mapping[str, Any],  # noqa: ARG002 - interface parity
    ) -> None:
        self._logger.info("epoch=%s metrics=%s", int(epoch), dict(metrics))

    def on_train_end(
        self, *, report: Mapping[str, Any], context: Mapping[str, Any]
    ) -> None:  # noqa: ARG002 - interface parity
        timing = report.get("timing", {})
        self._logger.info(
            "training finished: fit_s=%s total_s=%s",
            timing.get("fit_s"),
            timing.get("total_s"),
        )

    def on_exception(
        self, *, error: Exception, context: Mapping[str, Any]
    ) -> None:  # noqa: ARG002 - interface parity
        self._logger.exception("training failed: %s", error)


class ResourceProfilingCallback:
    """Collect lightweight runtime and memory profiling for a training run."""

    def __init__(self, *, enable_cuda: bool = True) -> None:
        self._enable_cuda = bool(enable_cuda)
        self._start_s: float | None = None
        self._tracemalloc_started_here = False

    def on_train_start(
        self, *, context: Mapping[str, Any]
    ) -> None:  # noqa: ARG002 - interface parity
        self._start_s = time.perf_counter()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started_here = True
        else:
            self._tracemalloc_started_here = False

    def _stop_tracemalloc_if_needed(self) -> None:
        if self._tracemalloc_started_here and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._tracemalloc_started_here = False

    def _collect_cuda_profile(self) -> dict[str, Any]:
        if not self._enable_cuda:
            return {"enabled": False, "reason": "disabled_by_config"}

        try:
            import torch
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            return {"enabled": False, "reason": f"torch_unavailable: {exc}"}

        if not torch.cuda.is_available():
            return {"enabled": False, "reason": "cuda_unavailable"}

        device_index = int(torch.cuda.current_device())
        stats = torch.cuda.memory_stats(device_index)
        return {
            "enabled": True,
            "device_index": device_index,
            "device_name": torch.cuda.get_device_name(device_index),
            "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
            "active_bytes_peak": int(stats.get("active_bytes.all.peak", 0)),
        }

    def on_epoch_end(
        self,
        *,
        epoch: int,  # noqa: ARG002 - interface parity
        metrics: Mapping[str, float],  # noqa: ARG002 - interface parity
        context: Mapping[str, Any],  # noqa: ARG002 - interface parity
    ) -> None:
        return

    def on_train_end(self, *, report: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        current_bytes = 0
        peak_bytes = 0
        if tracemalloc.is_tracing():
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()

        duration_s = 0.0
        if self._start_s is not None:
            duration_s = float(max(0.0, time.perf_counter() - float(self._start_s)))

        profile: dict[str, Any] = {
            "train_count": int(context.get("train_count", 0)),
            "validation_count": int(context.get("validation_count", 0)),
            "duration_s": duration_s,
            "memory": {
                "current_bytes": int(current_bytes),
                "peak_bytes": int(peak_bytes),
            },
            "cuda": self._collect_cuda_profile(),
        }

        if isinstance(report, dict):
            report["resource_profile"] = profile

        self._stop_tracemalloc_if_needed()

    def on_exception(
        self, *, error: Exception, context: Mapping[str, Any]
    ) -> None:  # noqa: ARG002 - interface parity
        self._stop_tracemalloc_if_needed()


__all__ = [
    "TrainingCallback",
    "MetricsLoggingCallback",
    "ResourceProfilingCallback",
    "run_callback_hook",
]
