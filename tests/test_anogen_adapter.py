from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((32, 32, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_vision_anogen_adapter_accepts_image_paths(tmp_path: Path) -> None:
    from pyimgano.models import create_model

    train_paths: list[str] = []
    for idx, value in enumerate((70, 80, 90, 100), start=1):
        path = tmp_path / f"train_{idx}.png"
        _write_rgb(path, value=value)
        train_paths.append(str(path))

    test_paths: list[str] = []
    for idx, value in enumerate((75, 120), start=1):
        path = tmp_path / f"test_{idx}.png"
        _write_rgb(path, value=value)
        test_paths.append(str(path))

    def _generator(image):  # noqa: ANN001
        arr = np.asarray(image, dtype=np.float32)
        mask = np.zeros(arr.shape[:2], dtype=np.float32)
        mask[8:16, 8:16] = 1.0
        anomalous = arr.copy()
        anomalous[mask > 0] = 255.0 - anomalous[mask > 0]
        return anomalous, mask

    det = create_model("vision_anogen_adapter", generator=_generator, contamination=0.25)
    det.fit(train_paths)

    scores = np.asarray(det.decision_function(test_paths), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
