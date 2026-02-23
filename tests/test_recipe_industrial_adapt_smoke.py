from pathlib import Path

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.recipes.registry import RECIPE_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig


def test_recipe_industrial_adapt_smoke(tmp_path):
    import cv2

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            self.fit_inputs = list(X)
            return self

        def decision_function(self, X):  # noqa: ANN001
            n = len(list(X))
            # Deterministic non-constant scores to avoid edge cases in metrics.
            if n == 0:
                return np.asarray([], dtype=np.float32)
            return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_recipe_dummy_detector",
        _DummyDetector,
        tags=("classical",),
        overwrite=True,
    )

    # Minimal custom dataset layout (same conventions used by other CLI tests).
    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    out_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
                "limit_test": 2,
            },
            "model": {
                "name": "test_recipe_dummy_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
                "per_image_jsonl": True,
            },
        }
    )

    # Ensure builtin recipes are registered.
    import pyimgano.recipes  # noqa: F401

    recipe = RECIPE_REGISTRY.get("industrial-adapt")
    report = recipe(cfg)

    assert Path(report["run_dir"]) == out_dir
    assert (out_dir / "report.json").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "environment.json").exists()
    assert (out_dir / "categories" / "custom" / "per_image.jsonl").exists()

