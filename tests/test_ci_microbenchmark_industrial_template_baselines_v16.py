from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def _make_pattern_image(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[8:20, 10:22, :] = 200
    img[40:52, 34:46, :] = 120
    img[25:28, :, :] = 80

    cy, cx = 32, 16
    r2 = 6 * 6
    for y in range(size):
        for x in range(size):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r2:
                img[y, x, :] = 160
    return img


def _make_custom_template_dataset(root: Path, *, h: int = 64, w: int = 64) -> None:
    """Deterministic synthetic dataset for template-style pixel-first baselines.

    Unlike the structural CI dataset (gray+noise), this one has a stable pattern
    so SSIM/NCC/template methods have meaningful correspondence.
    """

    rng = np.random.default_rng(0)
    base = _make_pattern_image(size=h).astype(np.float32)

    def _noisy() -> np.ndarray:
        noise = rng.normal(0.0, 2.0, size=(h, w, 3))
        return np.clip(base + noise, 0, 255)

    for i in range(12):
        _write_png(root / "train" / "normal" / f"train_{i}.png", _noisy())

    for i in range(6):
        _write_png(root / "test" / "normal" / f"normal_{i}.png", _noisy())

    for i in range(6):
        img = _noisy()
        y0 = 20 + (i % 3) * 6
        x0 = 30 + (i // 3) * 6
        img[y0 : y0 + 12, x0 : x0 + 12, :] = 255.0 - img[y0 : y0 + 12, x0 : x0 + 12, :]
        _write_png(root / "test" / "anomaly" / f"anomaly_{i}.png", img)


def test_ci_microbenchmark_template_baselines_are_stable(tmp_path: Path) -> None:
    """CI regression thresholds: keep template-style pixel-first baselines stable."""

    from pyimgano.pipelines.run_benchmark import run_benchmark

    root = tmp_path / "custom"
    _make_custom_template_dataset(root)

    models = [
        ("vision_pixel_mean_absdiff_map", {"resize_hw": (64, 64), "color": "gray", "topk": 0.02}),
        ("vision_template_ncc_map", {"resize_hw": (64, 64), "n_templates": 1, "window_hw": (9, 9)}),
        ("vision_pixel_gaussian_map", {"resize_hw": (64, 64), "color": "gray", "topk": 0.02}),
    ]
    for model, model_kwargs in models:
        payload = run_benchmark(
            dataset="custom",
            root=str(root),
            category="custom",
            model=str(model),
            seed=None,  # keep CI microbench torch-free for these baselines
            device="cpu",
            pretrained=False,
            contamination=0.1,
            resize=(64, 64),
            model_kwargs=dict(model_kwargs),
            save_run=False,
            per_image_jsonl=False,
        )

        auroc = float(payload["results"]["auroc"])
        ap = float(payload["results"]["average_precision"])

        assert auroc >= 0.99, f"{model}: auroc={auroc}"
        assert ap >= 0.99, f"{model}: ap={ap}"
