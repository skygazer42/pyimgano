from __future__ import annotations

import json
from pathlib import Path

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.run_context import initialize_workbench_run_context


def test_workbench_run_context_initializes_run_dir_and_metadata(tmp_path: Path) -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "custom",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {
                "output_dir": str(tmp_path / "run_out"),
                "save_run": True,
                "per_image_jsonl": False,
            },
        }
    )

    context = initialize_workbench_run_context(config=cfg, recipe_name="industrial-adapt")

    assert context is not None
    assert context.run_dir == Path(str(cfg.output.output_dir))
    assert context.paths.categories_dir.exists()
    assert context.paths.checkpoints_dir.exists()
    assert context.paths.artifacts_dir.exists()

    config_json = json.loads(context.paths.config_json.read_text(encoding="utf-8"))
    env_json = json.loads(context.paths.environment_json.read_text(encoding="utf-8"))
    assert config_json["config"]["seed"] == 123
    assert isinstance(env_json, dict)


def test_workbench_run_context_returns_none_when_save_run_disabled(tmp_path: Path) -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "custom",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    context = initialize_workbench_run_context(config=cfg, recipe_name="industrial-adapt")

    assert context is None
