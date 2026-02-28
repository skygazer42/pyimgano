from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, arr_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8, mode="RGB").save(path)


def _write_mask(path: Path, mask_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8, mode="L").save(path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_cli_pixel_threshold_strategy_supervised_segf1(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "root"
    h, w = 32, 32
    base = np.tile(np.linspace(0, 255, w, dtype=np.uint8).reshape(1, -1), (h, 1))
    base_rgb = np.stack([base, base, base], axis=-1)

    anomaly_rgb = np.array(base_rgb, copy=True)
    anomaly_rgb[10:22, 12:24] = 255 - anomaly_rgb[10:22, 12:24]

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:22, 12:24] = 255

    _write_rgb(root / "train" / "normal" / "n0.png", base_rgb)
    _write_rgb(root / "test" / "normal" / "x0.png", base_rgb)
    _write_rgb(root / "test" / "anomaly" / "a0.png", anomaly_rgb)
    _write_mask(root / "masks" / "a0_mask.png", mask)

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
                "mask_path": "masks/a0_mask.png",
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
            "ssim_template_map",
            "--model-kwargs",
            json.dumps({"resize_hw": [h, w], "reduction": "mean"}, ensure_ascii=False),
            "--pixel",
            "--pixel-segf1",
            "--pixel-threshold-strategy",
            "supervised_segf1",
            "--resize",
            str(h),
            str(w),
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    results = payload.get("results")
    assert isinstance(results, dict)
    pixel_metrics = results.get("pixel_metrics")
    assert isinstance(pixel_metrics, dict)
    assert "pixel_threshold" in pixel_metrics
    assert "pixel_segf1" in pixel_metrics

