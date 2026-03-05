from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, arr_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8, mode="RGB").save(path)


def _make_highres_pattern(h: int, w: int) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, num=h, dtype=np.float32),
        np.linspace(0.0, 1.0, num=w, dtype=np.float32),
        indexing="ij",
    )
    base = (0.45 + 0.25 * np.sin(2.0 * np.pi * (2.0 * xx + 1.3 * yy))).astype(np.float32)
    img = np.clip(base, 0.0, 1.0)
    # Keep channels identical so RGB/BGR conversions don't matter for grayscale baselines.
    rgb = np.stack([img, img, img], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


def test_e2e_tiling_defects_smoke(tmp_path: Path) -> None:
    """Smoke: high-res tiling + pixel-map detector + defects extraction."""

    from pyimgano.infer_cli import main as infer_main

    h, w = 512, 512
    base = _make_highres_pattern(h, w)

    anomaly = np.array(base, copy=True)
    anomaly[200:320, 220:340] = 255 - anomaly[200:320, 220:340]

    train_dir = tmp_path / "train"
    input_dir = tmp_path / "input"
    _write_rgb(train_dir / "n0.png", base)
    _write_rgb(input_dir / "a0.png", anomaly)

    out_jsonl = tmp_path / "out.jsonl"

    rc = infer_main(
        [
            "--model",
            "vision_pixel_mean_absdiff_map",
            "--train-dir",
            str(train_dir),
            "--input",
            str(input_dir),
            "--tile-size",
            "128",
            "--tile-stride",
            "128",
            "--tile-map-reduce",
            "hann",
            "--defects",
            "--pixel-threshold",
            "0.2",
            "--pixel-threshold-strategy",
            "fixed",
            "--defect-min-area",
            "10",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    records = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    assert any(r.get("defects", {}).get("regions") for r in records)
