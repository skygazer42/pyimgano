from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, *, color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_cli_manifest_pixel_mode_is_ok_without_masks(tmp_path: Path, capsys) -> None:
    """Regression: --dataset manifest --pixel should not crash on mask-less records."""

    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "root"
    _write_rgb(root / "train" / "normal" / "n0.png", color=(10, 20, 30))
    _write_rgb(root / "test" / "normal" / "x0.png", color=(11, 21, 31))
    _write_rgb(root / "test" / "anomaly" / "a0.png", color=(200, 10, 10))

    manifest = root / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "train/normal/n0.png", "category": "demo", "split": "train"},
            {"image_path": "test/normal/x0.png", "category": "demo", "split": "test", "label": 0},
            {
                "image_path": "test/anomaly/a0.png",
                "category": "demo",
                "split": "test",
                "label": 1,
                # no mask_path on purpose
            },
        ],
    )

    rc = benchmark_main(
        [
            "--dataset",
            "manifest",
            "--root",
            str(root),
            "--manifest-path",
            str(manifest),
            "--category",
            "demo",
            "--model",
            "vision_iforest",
            "--pixel",
            "--resize",
            "32",
            "32",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)

    status = payload.get("pixel_metrics_status")
    assert isinstance(status, dict)
    assert status.get("enabled") is False
    assert status.get("reason"), "expected a reason explaining why pixel metrics were disabled"
