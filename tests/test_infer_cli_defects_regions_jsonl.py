from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_pattern_png(path: Path, *, seed: int) -> None:
    h, w = 48, 48
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, num=h, dtype=np.float32),
        np.linspace(0.0, 1.0, num=w, dtype=np.float32),
        indexing="ij",
    )
    base = (0.55 + 0.2 * np.sin(2.0 * np.pi * (2.0 * xx + yy))).astype(np.float32)
    shift = (float(int(seed) % 5) - 2.0) * 0.01
    img = np.clip(base + shift, 0.0, 1.0)
    rgb = np.stack([img, img, img], axis=-1)
    arr_u8 = (rgb * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8, mode="RGB").save(path)


def _write_anomaly_png(path: Path, *, seed: int) -> None:
    _write_pattern_png(path, seed=seed)
    arr = np.asarray(Image.open(path).convert("RGB"))
    arr = np.array(arr, copy=True)
    arr[16:32, 16:32] = 255 - arr[16:32, 16:32]
    Image.fromarray(arr, mode="RGB").save(path)


def test_infer_cli_can_write_defects_regions_jsonl(tmp_path: Path) -> None:
    from pyimgano.infer_cli import main as infer_main

    train_dir = tmp_path / "train"
    input_dir = tmp_path / "input"
    _write_pattern_png(train_dir / "n0.png", seed=0)
    _write_pattern_png(train_dir / "n1.png", seed=1)
    _write_pattern_png(train_dir / "n2.png", seed=2)
    _write_anomaly_png(input_dir / "a0.png", seed=3)

    out_jsonl = tmp_path / "out.jsonl"
    regions_jsonl = tmp_path / "regions.jsonl"

    rc = infer_main(
        [
            "--model",
            "ssim_template_map",
            "--train-dir",
            str(train_dir),
            "--input",
            str(input_dir),
            "--defects",
            "--defects-regions-jsonl",
            str(regions_jsonl),
            "--save-jsonl",
            str(out_jsonl),
            "--pixel-threshold-strategy",
            "normal_pixel_quantile",
            "--pixel-normal-quantile",
            "0.95",
            "--defect-min-area",
            "1",
        ]
    )
    assert rc == 0
    assert regions_jsonl.exists()

    lines = regions_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload.get("input")
    defects = payload.get("defects")
    assert isinstance(defects, dict)
    assert "regions" in defects

