from __future__ import annotations

import json
from pathlib import Path


def test_evaluate_cli_delegates_to_harness_service_and_emits_json(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import pyimgano.evaluate_cli as evaluate_cli
    import pyimgano.services.evaluation_harness_service as harness_service

    cfg = tmp_path / "eval.json"
    cfg.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        harness_service,
        "load_evaluation_harness_request",
        lambda path: {"loaded_from": str(path)},
    )
    monkeypatch.setattr(
        harness_service,
        "run_evaluation_harness",
        lambda request: {"ok": True, "request": request},
    )

    rc = evaluate_cli.main(["--config", str(cfg), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["request"]["loaded_from"] == str(cfg)
