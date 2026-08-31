from __future__ import annotations

import json


def test_export_cli_delegates_to_post_run_service(monkeypatch, tmp_path, capsys):
    import pyimgano.export_cli as export_cli

    calls = []

    def _export_from_run(**kwargs):
        calls.append(dict(kwargs))
        return {
            "status": "ok",
            "run_dir": kwargs["run_dir"],
            "artifacts": [{"format": "native", "path": str(tmp_path / "artifact")}],
        }

    monkeypatch.setattr(export_cli.export_service, "export_from_run", _export_from_run)

    rc = export_cli.main(
        [
            "--from-run",
            str(tmp_path / "run"),
            "--format",
            "native",
            "--format",
            "onnx",
            "--out",
            str(tmp_path / "out"),
            "--category",
            "bottle",
            "--verification-level",
            "end-to-end",
            "--trust-checkpoint",
            "--json",
        ]
    )

    assert rc == 0
    assert calls == [
        {
            "run_dir": str(tmp_path / "run"),
            "formats": ("native", "onnx"),
            "out_dir": str(tmp_path / "out"),
            "category": "bottle",
            "verification_level": "end_to_end",
            "strict": True,
            "trust_checkpoint": True,
            "overwrite": False,
        }
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"


def test_export_cli_defaults_to_mandatory_reference_parity(monkeypatch, tmp_path):
    import pyimgano.export_cli as export_cli

    calls = []
    monkeypatch.setattr(
        export_cli.export_service,
        "export_from_run",
        lambda **kwargs: calls.append(dict(kwargs)) or {"status": "ok", "artifacts": []},
    )

    assert export_cli.main(["--from-run", str(tmp_path / "run")]) == 0
    assert calls[0]["formats"] == ("native",)
    assert calls[0]["verification_level"] == "reference_parity"
    assert calls[0]["strict"] is True


def test_export_cli_rejects_duplicate_formats(tmp_path, capsys):
    from pyimgano.export_cli import main

    rc = main(
        [
            "--from-run",
            str(tmp_path / "run"),
            "--format",
            "onnx",
            "--format",
            "onnx",
        ]
    )

    assert rc == 2
    assert "duplicate" in capsys.readouterr().err.lower()


def test_root_cli_exposes_export_and_artifact_commands(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []
    monkeypatch.setattr(
        root_cli,
        "_dispatch_command",
        lambda name, argv: calls.append((name, list(argv))) or 0,
    )

    assert root_cli.main(["export", "--from-run", "run"]) == 0
    assert root_cli.main(["artifact", "import", "--format", "onnx"]) == 0
    assert calls == [
        ("export", ["--from-run", "run"]),
        ("artifact", ["import", "--format", "onnx"]),
    ]
