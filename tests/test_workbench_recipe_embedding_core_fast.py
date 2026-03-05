from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyimgano.recipes.registry import RECIPE_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig


def _write_png(path: Path, *, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_recipe_industrial_embedding_core_fast_smoke(tmp_path: Path) -> None:
    # Minimal custom dataset layout (same as other workbench tests).
    root = tmp_path / "custom"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "train" / "normal" / "train_1.png", value=121)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    _write_png(root / "test" / "anomaly" / "bad_0.png", value=240)

    out_dir = tmp_path / "run_out"
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-embedding-core-fast",
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
                # Intentionally set a placeholder name; the recipe should override.
                "name": "vision_padim",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
                # Keep test fast/offline: use torchvision backbone without pretrained weights,
                # and keep a tiny image_size to reduce compute.
                "model_kwargs": {
                    "embedding_extractor": "torchvision_backbone",
                    "embedding_kwargs": {
                        "backbone": "resnet18",
                        "pretrained": False,
                        "pool": "avg",
                        "image_size": 16,
                        "batch_size": 2,
                        "device": "cpu",
                    },
                    "core_detector": "core_ecod",
                },
            },
            "output": {
                "output_dir": str(out_dir),
                "save_run": True,
                "per_image_jsonl": True,
            },
        }
    )

    import pyimgano.models  # noqa: F401 - ensure models registered
    import pyimgano.recipes  # noqa: F401 - ensure builtin recipes registered

    recipe = RECIPE_REGISTRY.get("industrial-embedding-core-fast")
    report = recipe(cfg)

    assert Path(report["run_dir"]) == out_dir
    assert (out_dir / "report.json").exists()
    cfg_path = out_dir / "config.json"
    assert cfg_path.exists()

    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["config"]["recipe"] == "industrial-embedding-core-fast"
    assert saved["config"]["model"]["name"] == "vision_embedding_core"
