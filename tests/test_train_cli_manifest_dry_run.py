from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"image_path":"a.png","category":"bottle"}',
                '{"image_path":"b.png","category":"bottle"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_train_cli_dry_run_manifest_validates_manifest_path_exists(tmp_path: Path, capsys) -> None:
    from pyimgano.train_cli import main

    missing = tmp_path / "missing.jsonl"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {
                    "name": "manifest",
                    "root": str(tmp_path),
                    "manifest_path": str(missing),
                },
                "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(cfg_path), "--dry-run"])
    assert code == 2
    err = capsys.readouterr().err.lower()
    assert "manifest" in err
    assert "not found" in err


def test_train_cli_dry_run_manifest_succeeds_with_valid_manifest(tmp_path: Path, capsys) -> None:
    from pyimgano.train_cli import main

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest)

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {
                    "name": "manifest",
                    "root": str(tmp_path),
                    "manifest_path": str(manifest),
                },
                "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(cfg_path), "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["dataset"]["name"] == "manifest"
