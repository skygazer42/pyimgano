from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.recipes.registry import RECIPE_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig


def test_recipe_micro_finetune_autoencoder_writes_checkpoint(tmp_path):
    import cv2

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            self.fit_inputs = list(X)
            return self

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ok", encoding="utf-8")

    MODEL_REGISTRY.register(
        "test_recipe_micro_finetune_autoencoder_dummy_detector",
        _DummyDetector,
        tags=("vision",),
        overwrite=True,
    )

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
            "recipe": "micro-finetune-autoencoder",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
            },
            "model": {
                "name": "test_recipe_micro_finetune_autoencoder_dummy_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
            },
        }
    )

    # Ensure builtin recipes are registered.
    import pyimgano.recipes  # noqa: F401

    recipe = RECIPE_REGISTRY.get("micro-finetune-autoencoder")
    report = recipe(cfg)

    assert Path(report["run_dir"]) == out_dir
    ckpt = out_dir / "checkpoints" / "model.pt"
    assert ckpt.exists()
    assert ckpt.read_text(encoding="utf-8") == "ok"

    report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report_json["checkpoint"]["path"].endswith("checkpoints/model.pt")

