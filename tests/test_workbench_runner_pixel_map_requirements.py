from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


def _write_rgb(path, *, value: int) -> None:  # noqa: ANN001 - test helper
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(value, value, value)).save(path)


def _write_mask(path) -> None:  # noqa: ANN001 - test helper
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=255).save(path)


def test_run_workbench_rejects_pixel_map_only_features_on_non_pixel_map_model(tmp_path) -> None:
    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, x):  # noqa: ANN001 - test stub
            self.fit_inputs = list(x)
            return self

        def decision_function(self, x):  # noqa: ANN001 - test stub
            items = list(x)
            return np.zeros(len(items), dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_runner_non_pixel_map_detector",
        _DummyDetector,
        tags=("vision", "classical"),
        overwrite=True,
    )

    root = tmp_path / "custom"
    _write_rgb(root / "train" / "normal" / "train_0.png", value=120)
    _write_rgb(root / "train" / "normal" / "train_1.png", value=121)
    _write_rgb(root / "test" / "normal" / "good_0.png", value=120)
    _write_rgb(root / "test" / "anomaly" / "bad_0.png", value=240)
    _write_mask(root / "ground_truth" / "anomaly" / "bad_0.png")

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "custom",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
                "limit_test": 2,
            },
            "model": {
                "name": "test_runner_non_pixel_map_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "adaptation": {
                "save_maps": True,
                "postprocess": {
                    "normalize": True,
                    "normalize_method": "minmax",
                },
            },
            "defects": {
                "enabled": True,
                "pixel_threshold": 0.5,
                "pixel_threshold_strategy": "fixed",
            },
            "output": {
                "output_dir": str(tmp_path / "run_out"),
                "save_run": True,
                "per_image_jsonl": True,
            },
        }
    )

    with pytest.raises(ValueError, match="pixel maps"):
        run_workbench(config=cfg, recipe_name="industrial-adapt")
