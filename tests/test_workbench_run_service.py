from __future__ import annotations


def test_workbench_run_service_exports_expected_boundary() -> None:
    import pyimgano.services.workbench_run_service as workbench_run_service

    assert workbench_run_service.__all__ == [
        "extract_threshold",
        "load_checkpoint_into_detector",
        "load_report_from_run",
        "load_workbench_config_from_run",
        "resolve_checkpoint_path",
        "select_category_report",
    ]


def test_workbench_run_service_delegates_to_workbench_load_run(monkeypatch) -> None:
    import pyimgano.services.workbench_run_service as workbench_run_service
    import pyimgano.workbench.load_run as load_run

    calls: list[tuple[str, object, object]] = []

    monkeypatch.setattr(
        load_run,
        "load_workbench_config_from_run",
        lambda run_dir: calls.append(("config", run_dir, None)) or {"kind": "config"},
    )
    monkeypatch.setattr(
        load_run,
        "load_checkpoint_into_detector",
        lambda detector, checkpoint_path: calls.append(("checkpoint", detector, checkpoint_path)),
    )

    detector = object()

    result = workbench_run_service.load_workbench_config_from_run("/tmp/run")
    workbench_run_service.load_checkpoint_into_detector(detector, "/tmp/model.pt")

    assert result == {"kind": "config"}
    assert calls == [
        ("config", "/tmp/run", None),
        ("checkpoint", detector, "/tmp/model.pt"),
    ]
