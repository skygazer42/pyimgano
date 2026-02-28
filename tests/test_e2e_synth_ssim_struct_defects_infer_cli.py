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


def test_e2e_synthesize_then_ssim_struct_map_then_infer_defects(tmp_path: Path) -> None:
    """End-to-end smoke for a second pixel-baseline variant (edges/structural)."""

    in_dir = tmp_path / "in"
    for i in range(8):
        _write_pattern_png(in_dir / f"{i}.png", seed=2025 + i)

    out_root = tmp_path / "out"
    manifest = out_root / "manifest.jsonl"

    from pyimgano.synthesize_cli import synthesize_dataset

    synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="synthetic",
        preset="tape",
        blend="alpha",
        alpha=0.9,
        seed=1,
        n_train=5,
        n_test_normal=1,
        n_test_anomaly=2,
        manifest_path=manifest,
        absolute_paths=True,
    )
    assert manifest.exists()

    train_dir = out_root / "train" / "normal"
    test_dir = out_root / "test"

    from pyimgano.infer_cli import main as infer_main

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"

    rc = infer_main(
        [
            "--model",
            "ssim_struct_map",
            "--train-dir",
            str(train_dir),
            "--input",
            str(test_dir),
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

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert any(r.get("defects", {}).get("regions") for r in records)
    assert sorted(masks_dir.glob("*.png")), "expected at least one exported defect mask"

