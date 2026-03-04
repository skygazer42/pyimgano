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
    # Template-inspection friendly: normals should be highly similar so SSIM maps
    # are low on normal data and spike on injected anomalies.
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


def test_e2e_synthesize_then_ssim_template_map_then_infer_defects(tmp_path: Path) -> None:
    """End-to-end smoke:

    synthesize dataset -> fit SSIM template-map detector -> infer CLI -> defects export (mask + regions).
    """

    # 1) Create a tiny normal image folder.
    in_dir = tmp_path / "in"
    for i in range(10):
        _write_pattern_png(in_dir / f"{i}.png", seed=1337 + i)

    # 2) Synthesize a dataset (custom layout) + manifest.
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
        seed=0,
        n_train=6,
        n_test_normal=2,
        n_test_anomaly=2,
        manifest_path=manifest,
        absolute_paths=True,
    )
    assert manifest.exists()

    train_dir = out_root / "train" / "normal"
    assert sorted(train_dir.glob("*.png")), "expected training normals in out_root/train/normal"

    test_dir = out_root / "test"
    assert sorted(test_dir.rglob("*.png")), "expected test images in out_root/test/**/*"

    # 3) Run inference with a real pixel-map capable detector and enable defects export.
    from pyimgano.infer_cli import main as infer_main

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"
    overlays_dir = tmp_path / "overlays"

    rc = infer_main(
        [
            "--model",
            "vision_pixel_mean_absdiff_map",
            "--train-dir",
            str(train_dir),
            "--input",
            str(test_dir),
            "--defects",
            "--defects-image-space",
            "--save-masks",
            str(masks_dir),
            "--save-overlays",
            str(overlays_dir),
            "--save-jsonl",
            str(out_jsonl),
            # Make pixel-threshold calibration more forgiving for the smoke test.
            "--pixel-threshold-strategy",
            "normal_pixel_quantile",
            "--pixel-normal-quantile",
            "0.95",
            # Basic postprocess knobs (exercise plumbing, keep stable).
            "--defect-min-area",
            "1",
            "--defect-map-smoothing",
            "median",
            "--defect-map-smoothing-ksize",
            "3",
            "--roi-xyxy-norm",
            "0.02",
            "0.02",
            "0.98",
            "0.98",
        ]
    )
    assert rc == 0

    # 4) Validate JSONL output and artifact paths.
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "expected JSONL records"

    records = [json.loads(line) for line in lines]
    assert all("defects" in r for r in records)

    any_nonempty_regions = False
    for rec in records:
        defects = rec["defects"]
        assert defects["pixel_threshold_provenance"]["source"] == "train_dir"
        assert defects["pixel_threshold_provenance"]["method"] == "normal_pixel_quantile"
        assert defects["mask"]["path"]

        mask_path = Path(defects["mask"]["path"])
        assert mask_path.exists()

        regions = list(defects.get("regions") or [])
        if regions:
            any_nonempty_regions = True
            for region in regions:
                assert "bbox_xyxy_image" in region

    assert any_nonempty_regions, "expected at least one input to produce a defect region"

    # 5) Artifact directories should have produced at least one file.
    assert sorted(masks_dir.glob("*.png")), "expected exported defect masks"
    assert sorted(overlays_dir.glob("*.png")), "expected exported overlays"
