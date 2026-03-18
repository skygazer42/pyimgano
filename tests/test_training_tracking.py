from __future__ import annotations

import json
import sys
import types

import pytest

from pyimgano.training.tracking import JsonlTracker, NullTracker, create_training_tracker


def test_jsonl_tracker_persists_params_and_metrics(tmp_path) -> None:
    tracker = JsonlTracker(tmp_path)
    tracker.log_params({"model": "vision_ecod", "epochs": 2})
    tracker.log_metrics({"loss": 0.15, "lr": 0.001}, step=2)
    tracker.close()

    params = json.loads((tmp_path / "params.json").read_text(encoding="utf-8"))
    assert params == {"model": "vision_ecod", "epochs": 2}

    metrics_lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metrics_lines) == 1
    metrics_payload = json.loads(metrics_lines[0])
    assert metrics_payload["step"] == 2
    assert metrics_payload["metrics"] == {"loss": 0.15, "lr": 0.001}


def test_create_training_tracker_returns_expected_backends(tmp_path) -> None:
    null_tracker = create_training_tracker("none")
    assert isinstance(null_tracker, NullTracker)

    jsonl_tracker = create_training_tracker("jsonl", log_dir=tmp_path)
    assert isinstance(jsonl_tracker, JsonlTracker)


def test_create_training_tracker_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported training tracker backend"):
        create_training_tracker("bogus")


def test_create_training_tracker_supports_mlflow_backend(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, object]] = []

    module = types.ModuleType("mlflow")

    def _set_tracking_uri(uri):  # noqa: ANN001
        calls.append(("set_tracking_uri", uri))

    def _set_experiment(name):  # noqa: ANN001
        calls.append(("set_experiment", name))

    def _start_run(run_name=None):  # noqa: ANN001
        calls.append(("start_run", run_name))
        return object()

    def _log_params(params):  # noqa: ANN001
        calls.append(("log_params", dict(params)))

    def _log_metrics(metrics, step=None):  # noqa: ANN001
        calls.append(("log_metrics", {"metrics": dict(metrics), "step": step}))

    def _log_artifact(path):  # noqa: ANN001
        calls.append(("log_artifact", str(path)))

    def _end_run():  # noqa: ANN001
        calls.append(("end_run", None))

    module.set_tracking_uri = _set_tracking_uri  # type: ignore[attr-defined]
    module.set_experiment = _set_experiment  # type: ignore[attr-defined]
    module.start_run = _start_run  # type: ignore[attr-defined]
    module.log_params = _log_params  # type: ignore[attr-defined]
    module.log_metrics = _log_metrics  # type: ignore[attr-defined]
    module.log_artifact = _log_artifact  # type: ignore[attr-defined]
    module.end_run = _end_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", module)

    tracker = create_training_tracker(
        "mlflow",
        log_dir=tmp_path / "mlruns",
        project="pyimgano-test",
        run_name="run-1",
    )
    tracker.log_params({"epochs": 2, "optimizer": "adamw"})
    tracker.log_metrics({"loss": 0.12}, step=3)
    tracker.log_artifact("training_report.json", {"fit_s": 1.2})
    tracker.close()

    assert calls[0][0] == "set_tracking_uri"
    assert calls[1] == ("set_experiment", "pyimgano-test")
    assert calls[2] == ("start_run", "run-1")
    assert calls[3] == ("log_params", {"epochs": "2", "optimizer": "adamw"})
    assert calls[4] == ("log_metrics", {"metrics": {"loss": 0.12}, "step": 3})
    assert calls[5][0] == "log_artifact"
    assert calls[6] == ("end_run", None)
