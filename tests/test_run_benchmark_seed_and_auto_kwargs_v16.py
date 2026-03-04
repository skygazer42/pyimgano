from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def _make_custom_dataset(root: Path, *, h: int = 64, w: int = 64) -> None:
    rng = np.random.default_rng(0)

    # Train normals (a bit of variation to avoid degenerate covariance edge cases).
    for i in range(4):
        noise = rng.normal(0.0, 3.0, size=(h, w, 3))
        img = np.clip(128.0 + noise, 0, 255)
        _write_png(root / "train" / "normal" / f"train_{i}.png", img)

    # Test normals.
    for i in range(2):
        noise = rng.normal(0.0, 3.0, size=(h, w, 3))
        img = np.clip(128.0 + noise, 0, 255)
        _write_png(root / "test" / "normal" / f"normal_{i}.png", img)

    # Test anomalies: inject a bright square patch.
    for i in range(2):
        noise = rng.normal(0.0, 3.0, size=(h, w, 3))
        img = 128.0 + noise
        img[16:32, 16:32] = 240.0
        img = np.clip(img, 0, 255)
        _write_png(root / "test" / "anomaly" / f"anomaly_{i}.png", img)


def test_run_benchmark_seed_does_not_break_strict_models(tmp_path: Path, monkeypatch) -> None:
    """Regression: `seed` should not inject random_seed/random_state into strict constructors.

    This used to fail because the benchmark pipeline inspected lazy constructors
    (`*args, **kwargs`) and incorrectly assumed it was safe to pass `random_seed`
    via **kwargs.
    """

    import importlib

    pipeline = importlib.import_module("pyimgano.pipelines.run_benchmark")

    # This regression test does not require torch seeding, and importing torch
    # inside `_seed_everything()` adds significant overhead in CI.
    monkeypatch.setattr(pipeline, "_seed_everything", lambda *_a, **_k: None)

    root = tmp_path / "custom"
    _make_custom_dataset(root)

    payload = pipeline.run_benchmark(
        dataset="custom",
        root=str(root),
        category="custom",
        model="vision_structural_ecod",
        seed=123,
        device="cpu",
        pretrained=False,
        contamination=0.1,
        resize=(64, 64),
        save_run=False,
        per_image_jsonl=False,
    )

    auroc = float(payload["results"]["auroc"])
    ap = float(payload["results"]["average_precision"])
    assert auroc >= 0.99
    assert ap >= 0.99
