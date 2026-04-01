from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _make_rgb_batch(*, count: int = 4, size: int = 32) -> list[np.ndarray]:
    rng = np.random.default_rng(2)
    out: list[np.ndarray] = []
    for _ in range(count):
        out.append(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))
    return out


def test_memseg_contract_accepts_numpy_image_list() -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=4)
    test = _make_rgb_batch(count=2)

    det = create_model(
        "vision_memseg",
        pretrained=False,
        device="cpu",
        memory_size=32,
        k_neighbors=1,
        use_segmentation_head=False,
    )

    det.fit(train)
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_memseg_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=4)

    det = create_model(
        "vision_memseg",
        pretrained=False,
        device="cpu",
        memory_size=32,
        k_neighbors=1,
        use_segmentation_head=False,
    )

    det.fit(train)
    out = capsys.readouterr().out
    assert out == ""
