from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_rgb(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((48, 48, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


@pytest.mark.integration
def test_embedding_cache_reuses_rows_on_second_pass(tmp_path: Path) -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model

    root = tmp_path / "imgs"
    train = [root / f"train_{i}.png" for i in range(4)]
    test = [root / f"test_{i}.png" for i in range(2)]
    for i, p in enumerate(train):
        _write_rgb(p, value=100 + i)
    for i, p in enumerate(test):
        _write_rgb(p, value=200 + i)

    cache_dir = tmp_path / "cache"

    det = create_model(
        "vision_embedding_core",
        contamination=0.1,
        embedding_extractor="torchvision_backbone",
        embedding_kwargs={
            "backbone": "resnet18",
            "pretrained": False,
            "pool": "avg",
            "device": "cpu",
            "batch_size": 2,
            "image_size": 64,
            "cache_dir": str(cache_dir),
        },
        core_detector="core_ecod",
    )
    det.fit([str(p) for p in train])

    scores1 = np.asarray(det.decision_function([str(p) for p in test]), dtype=np.float64).reshape(-1)
    scores2 = np.asarray(det.decision_function([str(p) for p in test]), dtype=np.float64).reshape(-1)
    assert scores1.shape == (2,)
    assert np.allclose(scores2, scores1)

    fx = getattr(det, "feature_extractor", None)
    stats = getattr(fx, "last_cache_stats_", None)
    assert isinstance(stats, dict)
    assert int(stats.get("hits", 0)) >= 1

