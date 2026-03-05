from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_png(path: Path, *, value: int, size: int = 32) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((size, size, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_vqvae_conv_tiny_smoke(tmp_path: Path) -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import create_model

    root = tmp_path / "ds"
    train_dir = root / "train"
    test_dir = root / "test"
    for i in range(4):
        _write_png(train_dir / f"train_{i}.png", value=110 + i, size=32)
    _write_png(test_dir / "good.png", value=120, size=32)
    _write_png(test_dir / "bad.png", value=240, size=32)

    train_paths = [str(p) for p in sorted(train_dir.iterdir())]
    test_paths = [str(p) for p in sorted(test_dir.iterdir())]

    det = create_model(
        "vqvae_conv",
        contamination=0.5,
        tiny=True,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        image_size=32,
        device="cpu",
        verbose=0,
    )

    fitted = det.fit(train_paths)
    assert fitted is det

    scores = np.asarray(det.decision_function(test_paths), dtype=np.float64)
    assert scores.shape == (len(test_paths),)
    assert np.isfinite(scores).all()

    amap = np.asarray(det.get_anomaly_map(str(test_paths[0])), dtype=np.float32)
    assert amap.shape == (32, 32)
    assert np.isfinite(amap).all()
