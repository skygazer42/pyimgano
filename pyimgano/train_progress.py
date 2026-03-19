from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping


class TrainProgressReporter:
    def on_run_start(self, *, config: Any, request: Any) -> None:
        return

    def on_run_context(self, *, run_dir: str | None) -> None:
        return

    def on_category_start(
        self,
        *,
        category: str,
        index: int | None = None,
        total: int | None = None,
    ) -> None:
        return

    def on_dataset_loaded(
        self,
        *,
        category: str,
        train_count: int,
        calibration_count: int,
        test_count: int,
        anomaly_count: int,
        pixel_metrics_enabled: bool | None,
        pixel_metrics_reason: str | None = None,
    ) -> None:
        return

    def on_training_start(
        self,
        *,
        category: str,
        enabled: bool,
        fit_kwargs: Mapping[str, Any] | None = None,
        tracker_backend: str | None = None,
        callback_names: list[str] | None = None,
    ) -> None:
        return

    def on_training_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int | None,
        metrics: Mapping[str, Any],
        live: bool = False,
    ) -> None:
        return

    def on_training_end(
        self,
        *,
        category: str,
        report: Mapping[str, Any] | None,
        checkpoint_meta: Mapping[str, Any] | None = None,
    ) -> None:
        return

    def on_calibration_end(
        self,
        *,
        category: str,
        threshold: float,
        quantile: float,
        source: str,
        score_summary: Mapping[str, Any] | None = None,
    ) -> None:
        return

    def on_evaluation_end(
        self,
        *,
        category: str,
        results: Mapping[str, Any],
        dataset_summary: Mapping[str, Any] | None = None,
    ) -> None:
        return

    def on_artifact_written(self, *, kind: str, path: str) -> None:
        return

    def on_run_end(self, *, report: Mapping[str, Any]) -> None:
        return

    def on_error(self, *, error: BaseException) -> None:
        return


_NULL_REPORTER = TrainProgressReporter()
_ACTIVE_REPORTER: ContextVar[TrainProgressReporter] = ContextVar(
    "pyimgano_active_train_progress_reporter",
    default=_NULL_REPORTER,
)


def get_active_train_progress_reporter() -> TrainProgressReporter:
    return _ACTIVE_REPORTER.get()


@contextmanager
def use_train_progress_reporter(
    reporter: TrainProgressReporter | None,
) -> Iterator[TrainProgressReporter]:
    active = _NULL_REPORTER if reporter is None else reporter
    token = _ACTIVE_REPORTER.set(active)
    try:
        yield active
    finally:
        _ACTIVE_REPORTER.reset(token)


__all__ = [
    "TrainProgressReporter",
    "get_active_train_progress_reporter",
    "use_train_progress_reporter",
]
