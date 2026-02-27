from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


def test_synthetic_anomaly_dataset_smoke(tmp_path: Path) -> None:
    from pyimgano.datasets.synthetic import SyntheticAnomalyDataset
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    p0 = tmp_path / "a.png"
    p1 = tmp_path / "b.png"
    _write_rgb(p0, color=(10, 20, 30))
    _write_rgb(p1, color=(40, 50, 60))

    syn = AnomalySynthesizer(SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=1.0))
    ds = SyntheticAnomalyDataset([p0, p1], synthesizer=syn, p_anomaly=1.0, seed=0)

    item = ds[0]
    assert item.image_u8.shape == (64, 64, 3)
    assert item.image_u8.dtype == np.uint8
    assert item.mask_u8.shape == (64, 64)
    assert item.mask_u8.dtype == np.uint8
    assert item.label in (0, 1)
    assert isinstance(item.meta, dict)

    # With p_anomaly=1.0, we expect synthetic anomalies in most cases.
    assert item.label == 1
    assert int(np.sum(item.mask_u8 > 0)) > 0


def test_synthetic_anomaly_dataset_can_disable_synthesis(tmp_path: Path) -> None:
    from pyimgano.datasets.synthetic import SyntheticAnomalyDataset
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    p0 = tmp_path / "c.png"
    _write_rgb(p0, color=(0, 0, 0))

    syn = AnomalySynthesizer(SynthSpec(preset="pit", probability=1.0))
    ds = SyntheticAnomalyDataset([p0], synthesizer=syn, p_anomaly=0.0, seed=0)

    item = ds[0]
    assert item.label == 0
    assert int(np.sum(item.mask_u8 > 0)) == 0

