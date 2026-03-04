from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_pattern_png(path: Path, *, seed: int) -> None:
    """Write a deterministic RGB image with structure (template-inspection friendly)."""

    h, w = 64, 64
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, num=h, dtype=np.float32),
        np.linspace(0.0, 1.0, num=w, dtype=np.float32),
        indexing="ij",
    )
    base = (
        0.55
        + 0.16 * np.sin(2.0 * np.pi * (3.0 * xx + 1.25 * yy))
        + 0.11 * np.cos(2.0 * np.pi * (2.2 * yy))
    ).astype(np.float32)
    shift = (float(int(seed) % 11) - 5.0) * 0.006
    img = np.clip(base + shift, 0.0, 1.0)

    rgb = np.stack(
        [
            img,
            np.clip(img * 0.92 + 0.04, 0.0, 1.0),
            np.clip(np.roll(img, shift=1, axis=1) * 0.96 + 0.02, 0.0, 1.0),
        ],
        axis=-1,
    )
    arr_u8 = (rgb * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8, mode="RGB").save(path)


def test_e2e_synthesize_shift_warp_then_pixel_first_defects(tmp_path: Path) -> None:
    """End-to-end smoke:

    synthesize (illumination+warp) -> fit a pixel-map template baseline -> infer CLI -> defects export.
    """

    # 1) Create a tiny normal folder.
    in_dir = tmp_path / "in"
    for i in range(10):
        _write_pattern_png(in_dir / f"{i}.png", seed=2026 + i)

    # 2) Synthesize a dataset using shift+warp presets.
    out_root = tmp_path / "out"
    manifest = out_root / "manifest.jsonl"

    from pyimgano.synthesize_cli import synthesize_dataset

    synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="synthetic",
        presets=["illumination", "warp"],
        blend="alpha",
        alpha=1.0,
        seed=0,
        n_train=6,
        n_test_normal=2,
        n_test_anomaly=2,
        manifest_path=manifest,
        absolute_paths=True,
    )
    assert manifest.exists()

    train_dir = out_root / "train" / "normal"
    test_dir = out_root / "test"
    assert sorted(train_dir.glob("*.png"))
    assert sorted(test_dir.rglob("*.png"))

    # 3) Run a real pixel-map template baseline and export defects.
    from pyimgano.infer_cli import main as infer_main

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"

    rc = infer_main(
        [
            "--model",
            "vision_pixel_mean_absdiff_map",
            "--model-kwargs",
            json.dumps({"resize_hw": [64, 64], "color": "gray", "reduction": "topk_mean", "topk": 0.02}),
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

    # 4) Validate at least one defects mask exists and JSONL has defects fields.
    assert sorted(masks_dir.glob("*.png")), "expected exported defect masks"

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    records = [json.loads(line) for line in lines]
    assert all("defects" in r for r in records)
    assert any((r.get("defects", {}).get("regions") or []) for r in records)
