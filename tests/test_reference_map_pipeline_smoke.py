from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_png(path: Path, *, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_reference_map_pipeline_smoke_basename_matching(tmp_path: Path) -> None:
    from pyimgano.pipelines.reference_map_pipeline import ReferenceMapPipeline

    ref_dir = tmp_path / "ref"
    query_dir = tmp_path / "query"

    _write_png(ref_dir / "a.png", value=120)
    _write_png(ref_dir / "b.png", value=120)
    _write_png(query_dir / "a.png", value=120)
    _write_png(query_dir / "b.png", value=220)

    class _AbsDiffMap(ReferenceMapPipeline):
        def _compute_anomaly_map(self, *, query_path: str, reference_path: str) -> np.ndarray:
            from PIL import Image

            q = np.asarray(Image.open(query_path).convert("L"), dtype=np.float32) / 255.0
            r = np.asarray(Image.open(reference_path).convert("L"), dtype=np.float32) / 255.0
            return np.abs(q - r).astype(np.float32)

    det = _AbsDiffMap(contamination=0.5, reference_dir=ref_dir, reduction="max")
    det.fit([str(query_dir / "a.png")])

    scores = np.asarray(det.decision_function([str(query_dir / "a.png"), str(query_dir / "b.png")]))
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])

    amap = det.get_anomaly_map(str(query_dir / "b.png"))
    assert amap.shape == (16, 16)
    assert float(amap.max()) > 0.0


def test_reference_map_pipeline_rejects_duplicate_reference_basenames(tmp_path: Path) -> None:
    from pyimgano.pipelines.reference_map_pipeline import build_reference_index

    ref_dir = tmp_path / "ref"
    _write_png(ref_dir / "a.png", value=120)
    _write_png(ref_dir / "sub" / "a.png", value=121)

    try:
        build_reference_index(ref_dir)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected duplicate basename error")
