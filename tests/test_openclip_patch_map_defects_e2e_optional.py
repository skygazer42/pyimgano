from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_png(path: Path, *, value: int, size: int = 64) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((size, size, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_openclip_patch_map_defects_export_optional(tmp_path: Path) -> None:
    # Optional dependency gate: this test is only meaningful when open_clip exists.
    import pytest

    pytest.importorskip("open_clip")
    pytest.importorskip("cv2")

    from pyimgano.infer_cli import main as infer_main

    train_dir = tmp_path / "train"
    inp_dir = tmp_path / "inp"
    _write_png(train_dir / "a.png", value=120, size=64)
    _write_png(inp_dir / "a.png", value=120, size=64)

    out_masks = tmp_path / "masks"
    out_jsonl = tmp_path / "out.jsonl"
    out_regions = tmp_path / "regions.jsonl"

    rc = infer_main(
        [
            "--model",
            "vision_openclip_patch_map",
            "--model-kwargs",
            json.dumps(
                {
                    "openclip_model_name": "ViT-B-32",
                    "openclip_pretrained": None,  # critical: no implicit downloads
                    "device": "cpu",
                    "force_image_size": 32,
                }
            ),
            "--train-dir",
            str(train_dir),
            "--input",
            str(inp_dir),
            "--include-maps",
            "--defects",
            "--pixel-threshold",
            "0.5",
            "--save-masks",
            str(out_masks),
            "--save-jsonl",
            str(out_jsonl),
            "--defects-regions-jsonl",
            str(out_regions),
        ]
    )
    assert rc == 0
    assert out_jsonl.exists()
    assert out_regions.exists()
    assert out_masks.exists()

    rows = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
