from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_png(path: Path, *, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_ref_patch_distance_map_detector_smoke(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # Hard block any torchvision weight downloads in unit tests.
    import torch.hub

    def _blocked(*_args, **_kwargs):  # noqa: ANN001, ANN201
        raise AssertionError("Torchvision weight download is forbidden in unit tests.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _blocked, raising=True)

    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import create_model

    ref_dir = tmp_path / "ref"
    query_dir = tmp_path / "query"

    _write_png(ref_dir / "a.png", value=120)
    _write_png(ref_dir / "b.png", value=120)
    _write_png(query_dir / "a.png", value=120)
    _write_png(query_dir / "b.png", value=220)

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
        reduction="max",
    )
    det.fit([str(query_dir / "a.png")])

    scores = np.asarray(det.decision_function([str(query_dir / "a.png"), str(query_dir / "b.png")]), dtype=np.float64)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert float(scores[1]) > float(scores[0])

    amap = np.asarray(det.get_anomaly_map(str(query_dir / "b.png")), dtype=np.float32)
    assert amap.shape == (16, 16)
    assert np.isfinite(amap).all()


def test_ref_patch_distance_map_detector_missing_reference(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import torch.hub

    def _blocked(*_args, **_kwargs):  # noqa: ANN001, ANN201
        raise AssertionError("Torchvision weight download is forbidden in unit tests.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _blocked, raising=True)

    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model

    ref_dir = tmp_path / "ref"
    query_dir = tmp_path / "query"

    _write_png(ref_dir / "a.png", value=120)
    _write_png(query_dir / "b.png", value=120)

    det = create_model(
        "vision_ref_patch_distance_map",
        contamination=0.5,
        reference_dir=str(ref_dir),
        backbone="resnet18",
        pretrained=False,
        image_size=16,
        device="cpu",
        metric="l2",
    )

    try:
        det.get_anomaly_map(str(query_dir / "b.png"))
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected missing reference error")

