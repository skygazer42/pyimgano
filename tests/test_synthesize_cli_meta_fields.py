from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def test_synthesize_dataset_attaches_severity_and_preset_id_meta(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    for i in range(6):
        _write_rgb(in_dir / f"{i}.png", seed=100 + i)

    out_root = tmp_path / "out"
    manifest = out_root / "manifest.jsonl"

    from pyimgano.synthesize_cli import synthesize_dataset

    records = synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="synthetic",
        presets=["illumination", "warp"],
        blend="alpha",
        alpha=0.9,
        seed=0,
        n_train=4,
        n_test_normal=1,
        n_test_anomaly=2,
        manifest_path=manifest,
        absolute_paths=True,
    )

    anomalies = [r for r in records if int(r.get("label", 0)) == 1]
    assert anomalies, "expected at least one anomaly record"

    for rec in anomalies:
        meta = rec.get("meta") or {}
        assert "severity" in meta
        assert "preset_id" in meta
        sev = float(meta["severity"])
        assert 0.0 <= sev <= 1.0
        assert str(meta["preset_id"]) in {"illumination", "warp"}
