from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_pattern_png(path: Path, *, seed: int) -> None:
    """Write a small deterministic RGB image with some structure (SSIM-friendly)."""

    h, w = 64, 64
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, num=h, dtype=np.float32),
        np.linspace(0.0, 1.0, num=w, dtype=np.float32),
        indexing="ij",
    )
    base = (
        0.55
        + 0.18 * np.sin(2.0 * np.pi * (3.0 * xx + 1.5 * yy))
        + 0.12 * np.cos(2.0 * np.pi * (2.0 * yy))
    ).astype(np.float32)
    shift = (float(int(seed) % 7) - 3.0) * 0.01
    img = np.clip(base + shift, 0.0, 1.0)

    rgb = np.stack(
        [
            img,
            np.clip(img * 0.9 + 0.05, 0.0, 1.0),
            np.clip(np.roll(img, shift=2, axis=1) * 0.95 + 0.02, 0.0, 1.0),
        ],
        axis=-1,
    )
    arr_u8 = (rgb * 255.0).round().astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8, mode="RGB").save(path)


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_include_maps_does_not_enable_defects(tmp_path: Path) -> None:
    from pyimgano.infer_cli import main as infer_main

    train_dir = tmp_path / "train"
    in_dir = tmp_path / "in"
    for i in range(6):
        _write_pattern_png(train_dir / f"t{i}.png", seed=1000 + i)
    for i in range(3):
        _write_pattern_png(in_dir / f"x{i}.png", seed=2000 + i)

    out_jsonl = tmp_path / "out.jsonl"
    rc = infer_main(
        [
            "--model",
            "vision_pixel_mean_absdiff_map",
            "--train-dir",
            str(train_dir),
            "--input",
            str(in_dir),
            "--include-maps",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    records = _read_jsonl(out_jsonl)
    assert records, "expected JSONL records"
    assert all(
        "anomaly_map" in r for r in records
    ), "expected anomaly_map metadata when --include-maps"
    assert all("defects" not in r for r in records), "--include-maps must not imply defects export"


def test_defects_enables_maps_implicitly(tmp_path: Path) -> None:
    from pyimgano.infer_cli import main as infer_main

    train_dir = tmp_path / "train"
    in_dir = tmp_path / "in"
    for i in range(6):
        _write_pattern_png(train_dir / f"t{i}.png", seed=3000 + i)
    for i in range(3):
        _write_pattern_png(in_dir / f"x{i}.png", seed=4000 + i)

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"
    rc = infer_main(
        [
            "--model",
            "vision_pixel_mean_absdiff_map",
            "--train-dir",
            str(train_dir),
            "--input",
            str(in_dir),
            "--defects",
            "--save-masks",
            str(masks_dir),
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

    records = _read_jsonl(out_jsonl)
    assert records, "expected JSONL records"
    assert all("anomaly_map" in r for r in records), "--defects must imply --include-maps"
    assert all(
        "defects" in r for r in records
    ), "expected defects payload when --defects is enabled"
    assert (
        sorted(masks_dir.glob("*.png"))
        or sorted(masks_dir.glob("*.npy"))
        or sorted(masks_dir.glob("*.npz"))
    ), "expected at least one saved defect mask artifact"
