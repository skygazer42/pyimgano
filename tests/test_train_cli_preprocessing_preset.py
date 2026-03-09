from __future__ import annotations

import json


def test_train_cli_dry_run_applies_preprocessing_preset(tmp_path, capsys) -> None:
    from pyimgano.train_cli import main

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "--config",
            str(cfg_path),
            "--dry-run",
            "--preprocessing-preset",
            "illumination-contrast-balanced",
        ]
    )
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    pre = payload["config"]["preprocessing"]["illumination_contrast"]
    assert pre["white_balance"] == "gray_world"
    assert pre["clahe"] is True
