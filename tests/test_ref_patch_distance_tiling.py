from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _write_png(path: Path, *, value: int, size: int = 64) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((size, size, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_ref_patch_distance_map_tiling_runs_and_preserves_shape(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    import torch.hub

    def _blocked(*_args, **_kwargs):  # noqa: ANN001, ANN201
        raise AssertionError("Torchvision weight download is forbidden in unit tests.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _blocked, raising=True)

    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import create_model

    ref_dir = tmp_path / "ref"
    query_dir = tmp_path / "query"

    _write_png(ref_dir / "x.png", value=120, size=64)
    _write_png(query_dir / "x.png", value=120, size=64)

    det = create_model(
        "vision_ref_patch_distance_map",
        contamination=0.5,
        reference_dir=str(ref_dir),
        backbone="resnet18",
        pretrained=False,
        node="layer4",
        image_size=16,
        device="cpu",
        metric="l2",
        tile_size=32,
        tile_stride=32,
        tile_map_reduce="max",
    )

    amap = np.asarray(det.get_anomaly_map(str(query_dir / "x.png")), dtype=np.float32)
    assert amap.shape == (64, 64)
    assert np.isfinite(amap).all()
    assert float(np.max(amap)) == pytest.approx(0.0)
