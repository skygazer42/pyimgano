from __future__ import annotations

import json
import re
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

    out = capsys.readouterr().out
    assert "Preflight Summary" in out
    assert "[PREFLIGHT]" in out
    assert "[DATA bottle]" in out
    assert "[CHECK]" in out
    assert "dataset=manifest" in out
    assert "category=bottle" in out
    assert "manifest=" in out
    assert "Scope" in out
    assert "Total" in out
    assert re.search(r"Explicit\s+3\s+1\s+0\s+2\s+-\s+1\s+1", out)
    assert re.search(r"Assigned\s+3\s+1\s+0\s+2\s+1", out)
    assert "Pixel" in out
    assert "masks=1/1" in out
    assert "errors=0" in out
    assert "warnings=0" in out
    assert "infos=0" in out


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

    out = capsys.readouterr().out
    assert "Preflight Summary" in out
    assert "[PREFLIGHT]" in out
    assert "[DATA bottle]" in out
    assert "[CHECK]" in out
    assert "dataset=manifest" in out
    assert "Scope" in out
    assert "Severity" in out
    assert "Code" in out
    assert re.search(r"error\s+MANIFEST_MISSING_IMAGE", out)
    assert "errors=1" in out
    assert "MANIFEST_MISSING_IMAGE" in out


def test_train_cli_preflight_json_preserves_machine_payload(tmp_path: Path, capsys) -> None:
    from pyimgano.train_cli import main

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "train.png").touch()
    (mdir / "good.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
        ],
    )

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {
                    "name": "manifest",
                    "root": str(tmp_path),
                    "manifest_path": str(manifest),
                    "category": "bottle",
                },
                "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(cfg_path), "--preflight", "--json"])
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    issues = payload["preflight"]["issues"]
    assert payload["preflight"]["dataset"] == "manifest"
    assert payload["preflight"]["category"] == "bottle"
    assert issues == []
    assert payload["preflight"]["dataset_readiness"]["status"] == "error"
    assert payload["preflight"]["dataset_readiness"]["issue_codes"] == [
        "MISSING_TEST_ANOMALY",
        "PIXEL_METRICS_UNAVAILABLE",
        "FEWSHOT_TRAIN_SET",
    ]


def test_train_cli_preflight_text_surfaces_dataset_readiness_issue_codes(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.train_cli import main

    root = tmp_path / "custom"
    (root / "train" / "normal").mkdir(parents=True, exist_ok=True)
    (root / "test" / "normal").mkdir(parents=True, exist_ok=True)
    (root / "train" / "normal" / "train_0.png").touch()
    (root / "test" / "normal" / "good_0.png").touch()

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {
                    "name": "custom",
                    "root": str(root),
                    "category": "custom",
                },
                "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(cfg_path), "--preflight"])
    assert code == 2

    out = capsys.readouterr().out
    assert "dataset_readiness=error" in out
    assert (
        "dataset_issue_codes=MISSING_TEST_ANOMALY,PIXEL_METRICS_UNAVAILABLE,FEWSHOT_TRAIN_SET"
        in out
    )
