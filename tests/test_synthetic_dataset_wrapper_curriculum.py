from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def test_synthetic_dataset_wrapper_severity_curriculum_progress(tmp_path: Path) -> None:
    p0 = tmp_path / "0.png"
    p1 = tmp_path / "1.png"
    _write_rgb(p0, seed=0)
    _write_rgb(p1, seed=1)

    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    syn = AnomalySynthesizer(SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=1.0))

    from pyimgano.datasets.synthetic import SyntheticAnomalyDataset

    ds = SyntheticAnomalyDataset(
        [p0, p1],
        synthesizer=syn,
        p_anomaly=1.0,
        seed=123,
        severity_range=(0.2, 1.0),
        curriculum_progress=0.0,
    )

    item_low = ds[0]
    sev_low = float(item_low.meta.get("severity", -1.0))
    assert abs(sev_low - 0.2) < 1e-8

    ds.set_curriculum_progress(1.0)
    item_high = ds[0]
    sev_high = float(item_high.meta.get("severity", -1.0))
    assert 0.2 <= sev_high <= 1.0
    assert sev_high >= sev_low
