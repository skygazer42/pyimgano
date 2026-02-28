from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _write_mask(path: Path, *, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16), int(value), dtype=np.uint8), mode="L").save(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_rad_converter_smoke_emits_meta(tmp_path: Path) -> None:
    from pyimgano.datasets.rad import convert_rad_to_manifest

    root = tmp_path / "rad"
    _write_rgb(root / "train" / "normal" / "view_0" / "n0.png", color=(10, 20, 30))
    _write_rgb(root / "test" / "normal" / "view_0" / "x0.png", color=(20, 30, 40))
    _write_rgb(root / "test" / "anomaly" / "view_1" / "a0.png", color=(200, 10, 10))
    _write_mask(root / "ground_truth" / "anomaly" / "a0_mask.png", value=255)

    out = tmp_path / "out" / "manifest.jsonl"
    records = convert_rad_to_manifest(
        root=root,
        out_path=out,
        category="demo",
        absolute_paths=False,
        include_masks=True,
    )

    assert out.exists()
    parsed = _read_jsonl(out)
    assert len(parsed) == len(records)

    metas = [r.get("meta") for r in parsed if isinstance(r.get("meta"), dict)]
    assert metas, "expected at least one meta object for a multi-view layout"
    assert any("view_id" in m for m in metas)

