from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_train_request_calls_canonical_post_run_exporter(monkeypatch, tmp_path):
    import pyimgano.services.export_service as export_service
    import pyimgano.services.train_service as train_service
    from pyimgano.recipes.registry import RECIPE_REGISTRY

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = []

    def _recipe(_cfg):
        events.append("recipe")
        return {"run_dir": str(run_dir), "threshold": 0.25}

    RECIPE_REGISTRY.register("test_artifact_export_recipe", _recipe, overwrite=True)
    cfg = SimpleNamespace(
        recipe="test_artifact_export_recipe",
        output=SimpleNamespace(save_run=True),
        model=SimpleNamespace(name="test-model"),
    )
    monkeypatch.setattr(train_service, "load_train_config", lambda _request: cfg)
    monkeypatch.setattr(train_service, "_validate_export_request", lambda *_args: None)
    monkeypatch.setattr(export_service, "preflight_train_export", lambda **_kwargs: None)

    calls = []

    def _export_from_run(**kwargs):
        events.append("export")
        calls.append(dict(kwargs))
        return {
            "status": "ok",
            "artifacts": [{"format": "native", "path": str(run_dir / "artifact")}],
        }

    monkeypatch.setattr(export_service, "export_from_run", _export_from_run)

    report = train_service.run_train_request(
        train_service.TrainRunRequest(
            config_path="ignored.json",
            export_formats=("native",),
            export_dir=str(tmp_path / "exports"),
            export_verification_level="end_to_end",
            export_strict=True,
            export_trust_checkpoint=True,
        )
    )

    assert events == ["recipe", "export"]
    assert calls == [
        {
            "run_dir": str(run_dir),
            "formats": ("native",),
            "out_dir": str(tmp_path / "exports"),
            "category": None,
            "verification_level": "end_to_end",
            "strict": True,
            "trust_checkpoint": True,
            "overwrite": False,
        }
    ]
    assert report["artifact_export"]["status"] == "ok"


def test_train_export_preflight_runs_before_recipe(monkeypatch, tmp_path):
    import pyimgano.services.export_service as export_service
    import pyimgano.services.train_service as train_service
    from pyimgano.recipes.registry import RECIPE_REGISTRY

    events = []

    def _recipe(_cfg):
        events.append("recipe")
        return {"run_dir": str(tmp_path / "run")}

    RECIPE_REGISTRY.register("test_artifact_preflight_recipe", _recipe, overwrite=True)
    cfg = SimpleNamespace(
        recipe="test_artifact_preflight_recipe",
        output=SimpleNamespace(save_run=True),
        model=SimpleNamespace(name="unsupported-model", model_kwargs={}),
    )
    monkeypatch.setattr(train_service, "load_train_config", lambda _request: cfg)
    monkeypatch.setattr(train_service, "_validate_export_request", lambda *_args: None)

    def _preflight(**_kwargs):
        events.append("preflight")
        raise ValueError("no_export_adapter")

    monkeypatch.setattr(export_service, "preflight_train_export", _preflight)

    request = train_service.TrainRunRequest(
        config_path="ignored.json",
        export_formats=("onnx",),
    )
    try:
        train_service.run_train_request(request)
    except ValueError as exc:
        assert "no_export_adapter" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("preflight should fail")
    assert events == ["preflight"]


def test_train_cli_builds_artifact_export_request():
    from pyimgano.train_cli import _build_parser, _build_train_request

    args = _build_parser().parse_args(
        [
            "--config",
            "cfg.json",
            "--export-format",
            "native",
            "--export-format",
            "onnx",
            "--export-dir",
            "artifacts",
            "--export-verification-level",
            "end-to-end",
            "--export-non-strict",
            "--export-trust-checkpoint",
            "--export-overwrite",
        ]
    )
    request = _build_train_request(args)

    assert request.export_formats == ("native", "onnx")
    assert request.export_dir == "artifacts"
    assert request.export_verification_level == "end_to_end"
    assert request.export_strict is False
    assert request.export_trust_checkpoint is True
    assert request.export_overwrite is True


def test_train_export_failure_is_persisted_after_successful_training(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.services.export_service as export_service
    import pyimgano.services.train_service as train_service
    from pyimgano.recipes.registry import RECIPE_REGISTRY

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    RECIPE_REGISTRY.register(
        "test_artifact_export_failure_recipe",
        lambda _cfg: {"run_dir": str(run_dir), "training": {"status": "ok"}},
        overwrite=True,
    )
    cfg = SimpleNamespace(
        recipe="test_artifact_export_failure_recipe",
        output=SimpleNamespace(save_run=True),
        model=SimpleNamespace(name="test-model"),
    )
    monkeypatch.setattr(train_service, "load_train_config", lambda _request: cfg)
    monkeypatch.setattr(train_service, "_validate_export_request", lambda *_args: None)
    monkeypatch.setattr(export_service, "preflight_train_export", lambda **_kwargs: None)
    monkeypatch.setattr(
        export_service,
        "export_from_run",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("parity failed")),
    )

    with pytest.raises(RuntimeError, match="parity failed"):
        train_service.run_train_request(
            train_service.TrainRunRequest(
                config_path="ignored.json",
                export_formats=("native",),
            )
        )

    persisted = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert persisted["training"] == {"status": "ok"}
    assert persisted["artifact_export"]["status"] == "failed"
    assert "parity failed" in persisted["artifact_export"]["failures"][0]["reason"]


def test_non_strict_train_export_still_fails_when_no_format_succeeds(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.services.export_service as export_service
    import pyimgano.services.train_service as train_service
    from pyimgano.recipes.registry import RECIPE_REGISTRY

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    RECIPE_REGISTRY.register(
        "test_artifact_export_empty_recipe",
        lambda _cfg: {"run_dir": str(run_dir)},
        overwrite=True,
    )
    cfg = SimpleNamespace(
        recipe="test_artifact_export_empty_recipe",
        output=SimpleNamespace(save_run=True),
        model=SimpleNamespace(name="test-model"),
    )
    monkeypatch.setattr(train_service, "load_train_config", lambda _request: cfg)
    monkeypatch.setattr(train_service, "_validate_export_request", lambda *_args: None)
    monkeypatch.setattr(export_service, "preflight_train_export", lambda **_kwargs: None)
    monkeypatch.setattr(
        export_service,
        "export_from_run",
        lambda **_kwargs: {
            "status": "failed",
            "artifacts": [],
            "failures": [{"format": "onnx", "reason": "unsupported"}],
        },
    )

    with pytest.raises(export_service.ExportServiceError, match="no deployable result"):
        train_service.run_train_request(
            train_service.TrainRunRequest(
                config_path="ignored.json",
                export_formats=("onnx",),
                export_strict=False,
            )
        )

    persisted = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert persisted["artifact_export"]["status"] == "failed"
