from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_train_cli_preflight_manifest_ok(tmp_path: Path, capsys) -> None:
    from pyimgano.train_cli import main

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "train.png").touch()
    (mdir / "good.png").touch()
    (mdir / "bad.png").touch()
    (mdir / "bad_mask.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
            {
                "image_path": "bad.png",
                "category": "bottle",
                "split": "test",
                "label": 1,
                "mask_path": "bad_mask.png",
            },
        ],
    )

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "seed": 123,
                "dataset": {
                    "name": "manifest",
                    "root": str(tmp_path),
                    "manifest_path": str(manifest),
                    "category": "bottle",
                    "resize": [16, 16],
                    "input_mode": "paths",
                },
                "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(cfg_path), "--preflight"])
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["preflight"]["dataset"] == "manifest"
    assert payload["preflight"]["category"] == "bottle"
    assert payload["preflight"]["issues"] == []


def test_train_cli_preflight_manifest_errors_exit_2(tmp_path: Path, capsys) -> None:
    from pyimgano.train_cli import main

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "train.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            {"image_path": "missing.png", "category": "bottle", "split": "test", "label": 0},
        ],
    )

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "seed": 123,
                "dataset": {
                    "name": "manifest",
                    "root": str(tmp_path),
                    "manifest_path": str(manifest),
                    "category": "bottle",
                    "resize": [16, 16],
                    "input_mode": "paths",
                },
                "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(cfg_path), "--preflight"])
    assert code == 2

    payload = json.loads(capsys.readouterr().out)
    issues = payload["preflight"]["issues"]
    assert any(i["code"] == "MANIFEST_MISSING_IMAGE" and i["severity"] == "error" for i in issues)

