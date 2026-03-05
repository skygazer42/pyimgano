from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def test_synthesize_dataset_severity_range_and_num_defects_are_reflected_in_meta(
    tmp_path: Path,
) -> None:
    in_dir = tmp_path / "in"
    for i in range(8):
        _write_rgb(in_dir / f"{i}.png", seed=200 + i)

    from pyimgano.synthesize_cli import synthesize_dataset

    out_root = tmp_path / "out"
    records = synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="synthetic",
        preset="scratch",
        blend="alpha",
        alpha=0.9,
        seed=0,
        n_train=6,
        n_test_normal=1,
        n_test_anomaly=2,
        num_defects=3,
        severity_range=(0.25, 0.35),
        # Use a dense-mask preset to avoid flakiness in "accepted" logic.
        presets=["illumination"],
    )

    anomalies = [r for r in records if int(r.get("label", 0)) == 1]
    assert anomalies, "expected at least one anomaly record"

    for rec in anomalies:
        meta = rec.get("meta") or {}
        sev = float(meta.get("severity", -1.0))
        assert 0.25 <= sev <= 0.35
        assert int(meta.get("num_defects", 0)) == 3
        assert 1 <= int(meta.get("defects_applied", 0)) <= 3
