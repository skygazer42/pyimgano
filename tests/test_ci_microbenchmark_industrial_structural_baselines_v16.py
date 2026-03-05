from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def _make_custom_dataset(root: Path, *, h: int = 64, w: int = 64) -> None:
    """Deterministic synthetic 'industrial' dataset for CI regression tests.

    - Normals: gray + low noise.
    - Anomalies: same, plus a bright square patch.
    """

    rng = np.random.default_rng(0)

    for i in range(12):
        noise = rng.normal(0.0, 3.0, size=(h, w, 3))
        img = np.clip(128.0 + noise, 0, 255)
        _write_png(root / "train" / "normal" / f"train_{i}.png", img)

    for i in range(6):
        noise = rng.normal(0.0, 3.0, size=(h, w, 3))
        img = np.clip(128.0 + noise, 0, 255)
        _write_png(root / "test" / "normal" / f"normal_{i}.png", img)

    for i in range(6):
        noise = rng.normal(0.0, 3.0, size=(h, w, 3))
        img = 128.0 + noise
        x1 = 8 + (i % 3) * 8
        y1 = 8 + (i // 3) * 8
        img[y1 : y1 + 16, x1 : x1 + 16] = 240.0
        img = np.clip(img, 0, 255)
        _write_png(root / "test" / "anomaly" / f"anomaly_{i}.png", img)


def test_ci_microbenchmark_structural_baselines_are_stable(tmp_path: Path) -> None:
    """CI regression thresholds: keep core industrial baselines from silently degrading."""

    from pyimgano.pipelines.run_benchmark import run_benchmark

    root = tmp_path / "custom"
    _make_custom_dataset(root)

    models = [
        "vision_structural_ecod",
        "vision_structural_knn",
        "vision_structural_pca_md",
    ]
    for model in models:
        payload = run_benchmark(
            dataset="custom",
            root=str(root),
            category="custom",
            model=str(model),
            seed=None,  # avoid pulling torch into CI for classical baselines
            device="cpu",
            pretrained=False,
            contamination=0.1,
            resize=(64, 64),
            save_run=False,
            per_image_jsonl=False,
        )

        auroc = float(payload["results"]["auroc"])
        ap = float(payload["results"]["average_precision"])

        assert auroc >= 0.99, f"{model}: auroc={auroc}"
        assert ap >= 0.99, f"{model}: ap={ap}"
