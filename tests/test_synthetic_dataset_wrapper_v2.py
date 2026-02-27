from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


def test_synthetic_anomaly_dataset_wrapper_smoke(tmp_path: Path) -> None:
    from pyimgano.datasets.synthetic import SyntheticAnomalyDataset
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    root = tmp_path / "imgs"
    a = root / "a.png"
    b = root / "b.png"
    _write_rgb(a, color=(10, 20, 30))
    _write_rgb(b, color=(40, 50, 60))

    syn = AnomalySynthesizer(SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=0.9))
    ds = SyntheticAnomalyDataset([a, b], synthesizer=syn, p_anomaly=1.0, seed=0)

    item0a = ds[0]
    item0b = ds[0]
    assert item0a.image_u8.shape == (64, 64, 3)
    assert item0a.mask_u8.shape == (64, 64)
    assert item0a.image_u8.dtype == np.uint8
    assert item0a.mask_u8.dtype == np.uint8
    assert item0a.label in (0, 1)
    assert np.array_equal(item0a.image_u8, item0b.image_u8)
    assert np.array_equal(item0a.mask_u8, item0b.mask_u8)


def test_synthetic_anomaly_dataset_wrapper_can_disable_anomalies(tmp_path: Path) -> None:
    from pyimgano.datasets.synthetic import SyntheticAnomalyDataset
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    root = tmp_path / "imgs"
    a = root / "a.png"
    _write_rgb(a, color=(120, 120, 120))

    syn = AnomalySynthesizer(SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=0.9))
    ds = SyntheticAnomalyDataset([a], synthesizer=syn, p_anomaly=0.0, seed=0)
    item = ds[0]
    assert item.label == 0
    assert int(item.mask_u8.max()) == 0

