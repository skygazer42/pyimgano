from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import pyimgano.infer_cli as infer_cli


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_from_run_missing_config_reports_context(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "report.json").write_text("{}", encoding="utf-8")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    rc = infer_cli.main(["--from-run", str(run_dir), "--input", str(input_dir)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Missing config.json" in err
    assert "context: from_run=" in err


def test_infer_cli_from_run_invalid_report_json(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "dataset": {"name": "custom", "root": "/tmp/data"},
                    "model": {"name": "vision_ecod"},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text("{", encoding="utf-8")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    rc = infer_cli.main(["--from-run", str(run_dir), "--input", str(input_dir)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Invalid JSON in report.json" in err


def test_infer_cli_from_run_requires_category_when_multiple(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "dataset": {"name": "custom", "root": "/tmp/data"},
                    "model": {"name": "vision_ecod"},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "per_category": {
                    "a": {"threshold": 0.5},
                    "b": {"threshold": 0.6},
                }
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    rc = infer_cli.main(["--from-run", str(run_dir), "--input", str(input_dir)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "multiple categories" in err.lower()
    assert "--from-run-category" in err

