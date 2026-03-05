from __future__ import annotations

import json


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


def test_doctor_cli_outputs_text(capsys) -> None:
    from pyimgano.doctor_cli import main as doctor_main

    rc = doctor_main([])
    assert rc == 0

    out = capsys.readouterr().out
    assert "pyimgano-doctor" in out.lower()
    assert "pyimgano" in out.lower()
