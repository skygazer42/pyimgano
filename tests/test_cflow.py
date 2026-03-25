from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def test_vision_cflow_contract_fit_and_score(tmp_path: Path) -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(30)
    train_paths: list[str] = []
    test_paths: list[str] = []
    for idx in range(4):
        path = tmp_path / f"train_{idx}.png"
        _write_rgb(path, rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        train_paths.append(str(path))
    for idx in range(2):
        path = tmp_path / f"test_{idx}.png"
        _write_rgb(path, rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        test_paths.append(str(path))

    det = create_model(
        "vision_cflow",
        backbone="resnet18",
        pretrained_backbone=False,
        epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )

    det.fit(train_paths)
    scores = np.asarray(det.decision_function(test_paths), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
