from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_synthesize_cli_from_manifest_appends_anomalies(tmp_path: Path) -> None:
    from pyimgano.synthesize_cli import main

    # Build a tiny source dataset and a manifest pointing at it (absolute paths).
    src = tmp_path / "src"
    _write_rgb(src / "a.png", color=(10, 10, 10))
    _write_rgb(src / "b.png", color=(20, 20, 20))

    in_manifest = tmp_path / "in_manifest.jsonl"
    lines = [
        {
            "image_path": str((src / "a.png").resolve()),
            "category": "c1",
            "split": "train",
            "label": 0,
        },
        {
            "image_path": str((src / "b.png").resolve()),
            "category": "c1",
            "split": "train",
            "label": 0,
        },
    ]
    in_manifest.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    out_root = tmp_path / "out"
    out_manifest = out_root / "manifest.jsonl"

    code = main(
        [
            "--from-manifest",
            str(in_manifest),
            "--from-manifest-category",
            "c1",
            "--from-manifest-split",
            "train",
            "--from-manifest-n",
            "2",
            "--out-root",
            str(out_root),
            "--manifest",
            str(out_manifest),
            "--preset",
            "scratch",
            "--seed",
            "0",
        ]
    )
    assert code == 0
    assert out_manifest.exists()

    rows = [
        json.loads(line)
        for line in out_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows

    synth = [r for r in rows if r.get("label") == 1]
    assert len(synth) >= 1
    assert all("mask_path" in r for r in synth)
    for r in synth:
        assert Path(r["image_path"]).exists()
        assert Path(r["mask_path"]).exists()
