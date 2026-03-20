from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_doctor_cli_outputs_json(capsys) -> None:
    from pyimgano.doctor_cli import main as doctor_main

    rc = doctor_main(["--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload.get("tool") == "pyimgano-doctor"
    assert isinstance(payload.get("pyimgano_version"), str)

    python = payload.get("python")
    assert isinstance(python, dict)
    assert isinstance(python.get("version"), str)

    platform = payload.get("platform")
    assert isinstance(platform, dict)
    assert isinstance(platform.get("system"), str)

    optional_modules = payload.get("optional_modules")
    assert isinstance(optional_modules, list)
    assert optional_modules, "expected at least one optional module check"
    for item in optional_modules:
        assert isinstance(item, dict)
        assert isinstance(item.get("module"), str)
        assert isinstance(item.get("available"), bool)

    baselines = payload.get("baselines")
    assert isinstance(baselines, dict)
    assert "industrial-v4" in set(baselines.get("suites", []))
    assert "industrial-feature-small" in set(baselines.get("sweeps", []))


def test_doctor_cli_json_delegates_to_doctor_service(monkeypatch, capsys) -> None:
    import pyimgano.services.doctor_service as doctor_service
    from pyimgano.doctor_cli import main

    monkeypatch.setattr(
        doctor_service,
        "collect_doctor_payload",
        lambda **_kwargs: {
            "tool": "delegated-doctor",
            "python": {},
            "platform": {},
            "optional_modules": [],
            "baselines": {},
        },
    )

    rc = main(["--json"])
    assert rc == 0
    assert "delegated-doctor" in capsys.readouterr().out


def test_doctor_cli_json_uses_cli_output_helper(monkeypatch) -> None:
    import pyimgano.doctor_cli as doctor_cli

    monkeypatch.setattr(
        doctor_cli.doctor_service,
        "collect_doctor_payload",
        lambda **_kwargs: {"tool": "delegated-doctor"},
    )

    calls = []
    monkeypatch.setattr(
        doctor_cli,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_json": staticmethod(
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 17
                ),
                "print_cli_error": staticmethod(lambda exc, **kwargs: None),
            },
        ),
        raising=False,
    )

    rc = doctor_cli.main(["--json"])
    assert rc == 17
    assert calls == [({"tool": "delegated-doctor"}, {"indent": None})]


def test_doctor_cli_outputs_text(capsys) -> None:
    from pyimgano.doctor_cli import main as doctor_main

    rc = doctor_main([])
    assert rc == 0

    out = capsys.readouterr().out
    assert "pyimgano-doctor" in out.lower()
    assert "pyimgano" in out.lower()


def test_doctor_cli_suite_check_outputs_json(capsys) -> None:
    from pyimgano.doctor_cli import main as doctor_main

    rc = doctor_main(["--json", "--suite", "industrial-v4"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    suite_checks = payload.get("suite_checks")
    assert isinstance(suite_checks, dict)

    v4 = suite_checks.get("industrial-v4")
    assert isinstance(v4, dict)
    assert v4.get("suite") == "industrial-v4"

    baselines = v4.get("baselines")
    assert isinstance(baselines, list)
    assert baselines, "expected at least one baseline entry"

    b0 = baselines[0]
    assert isinstance(b0, dict)
    assert isinstance(b0.get("name"), str)
    assert isinstance(b0.get("optional"), bool)
    assert isinstance(b0.get("requires_extras"), list)
    assert isinstance(b0.get("missing_extras"), list)
    assert isinstance(b0.get("runnable"), bool)


def test_doctor_cli_require_extras_missing_exits_nonzero(capsys) -> None:
    from pyimgano.doctor_cli import main as doctor_main

    rc = doctor_main(["--json", "--require-extras", "definitely_missing_extra"])
    assert rc == 1

    payload = json.loads(capsys.readouterr().out)
    req = payload.get("require_extras")
    assert isinstance(req, dict)
    assert req.get("ok") is False
    assert "definitely_missing_extra" in set(req.get("missing", []))


def test_doctor_cli_require_extras_import_error_exits_nonzero(tmp_path, capsys) -> None:
    """If a module exists but fails to import, --require-extras should fail.

    This is important for CI/deploy gates: `find_spec()` is not enough when native
    wheels are present but broken (missing shared libs, ABI mismatch, etc.).
    """

    from pathlib import Path

    from pyimgano.doctor_cli import main as doctor_main

    broken_root = tmp_path / "pyimgano_broken_extra_root"
    pkg_dir = broken_root / "pyimgano__broken_extra__"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("raise RuntimeError('broken import')\n", encoding="utf-8")

    import sys

    sys.path.insert(0, str(broken_root))
    try:
        rc = doctor_main(["--json", "--require-extras", "pyimgano__broken_extra__"])
        assert rc == 1

        payload = json.loads(capsys.readouterr().out)
        req = payload.get("require_extras")
        assert isinstance(req, dict)
        assert req.get("ok") is False
        assert "pyimgano__broken_extra__" in set(req.get("missing", []))
    finally:
        # Avoid polluting other tests (order-independent).
        sys.path = [p for p in sys.path if str(p) != str(broken_root)]


def test_doctor_cli_error_uses_cli_output_helper(monkeypatch) -> None:
    import pyimgano.doctor_cli as doctor_cli

    def _boom(**_kwargs):
        raise RuntimeError("broken-doctor")

    monkeypatch.setattr(doctor_cli.doctor_service, "collect_doctor_payload", _boom)

    calls = []
    monkeypatch.setattr(
        doctor_cli,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_json": staticmethod(lambda payload, **kwargs: 0),
                "print_cli_error": staticmethod(
                    lambda exc, **kwargs: calls.append((str(exc), kwargs))
                ),
            },
        ),
        raising=False,
    )

    rc = doctor_cli.main([])
    assert rc == 1
    assert calls == [("broken-doctor", {})]


def test_doctor_cli_run_dir_readiness_outputs_json(tmp_path: Path, capsys) -> None:
    from pyimgano.doctor_cli import main as doctor_main

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"run_dir": str(run_dir)})
    _write_json(run_dir / "config.json", {"recipe": "industrial-adapt"})
    _write_json(run_dir / "environment.json", {"python": "3.10"})
    _write_json(
        run_dir / "artifacts" / "infer_config.json",
        {
            "model": {"name": "vision_patchcore", "model_kwargs": {}},
            "defects": {"mask_format": "png"},
        },
    )

    rc = doctor_main(["--json", "--run-dir", str(run_dir)])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    readiness = payload.get("readiness")
    assert isinstance(readiness, dict)
    assert readiness.get("target_kind") == "run"
    assert readiness.get("path") == str(run_dir)
    assert readiness.get("status") == "warning"
    acceptance = readiness.get("acceptance")
    assert isinstance(acceptance, dict)
    assert acceptance.get("ready") is False
    assert "insufficient_quality_status" in set(readiness.get("issues", []))


def test_doctor_cli_deploy_bundle_readiness_exits_nonzero_on_invalid_bundle(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.doctor_cli import main as doctor_main

    bundle_dir = tmp_path / "deploy_bundle"
    _write_json(
        bundle_dir / "infer_config.json",
        {
            "model": {"name": "vision_patchcore", "model_kwargs": {}},
            "artifact_quality": {
                "status": "deployable",
                "threshold_scope": "image",
                "has_threshold_provenance": True,
                "has_split_fingerprint": True,
                "has_prediction_policy": False,
                "has_deploy_bundle": True,
                "has_bundle_manifest": True,
                "required_bundle_artifacts_present": False,
                "bundle_artifact_roles": {},
                "audit_refs": {"calibration_card": "calibration_card.json"},
                "deploy_refs": {"bundle_manifest": "bundle_manifest.json"},
            },
        },
    )

    rc = doctor_main(["--json", "--deploy-bundle", str(bundle_dir)])
    assert rc == 1

    payload = json.loads(capsys.readouterr().out)
    readiness = payload.get("readiness")
    assert isinstance(readiness, dict)
    assert readiness.get("target_kind") == "deploy_bundle"
    assert readiness.get("status") == "error"
    issues = readiness.get("issues", [])
    assert isinstance(issues, list)
    assert issues
