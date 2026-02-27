from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, value: int = 128, anomaly: bool = False) -> None:
    img = np.ones((64, 64, 3), dtype=np.uint8) * int(value)
    if anomaly:
        img[20:44, 20:44] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


def test_ae_resnet_unet_contract_fit_decision_predict(tmp_path: Path) -> None:
    from pyimgano.models import create_model

    normals = []
    for i in range(4):
        p = tmp_path / f"n_{i}.png"
        _write_rgb(p, value=90 + i * 5, anomaly=False)
        normals.append(str(p))

    anomaly = tmp_path / "a_0.png"
    _write_rgb(anomaly, value=100, anomaly=True)
    test_paths = normals + [str(anomaly)]

    det = create_model(
        "ae_resnet_unet",
        contamination=0.2,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        image_size=64,
        tiny=True,
        verbose=0,
    )

    det.fit(normals)
    scores = np.asarray(det.decision_function(test_paths), dtype=np.float64).reshape(-1)
    assert scores.shape == (len(test_paths),)
    assert np.all(np.isfinite(scores))

    labels = np.asarray(det.predict(test_paths), dtype=np.int64).reshape(-1)
    assert labels.shape == (len(test_paths),)
    assert set(labels.tolist()).issubset({0, 1})

