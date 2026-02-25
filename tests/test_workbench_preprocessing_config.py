from __future__ import annotations

import pytest

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import build_infer_config_payload


def test_workbench_config_parses_preprocessing_illumination_contrast() -> None:
    raw = {
        "recipe": "industrial-adapt",
        "dataset": {
            "name": "custom",
            "root": "/tmp/custom",
            "category": "custom",
            "resize": [16, 16],
            "input_mode": "paths",
        },
        "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False, "contamination": 0.1},
        "preprocessing": {
            "illumination_contrast": {
                "white_balance": "gray_world",
                "clahe": True,
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": [4, 4],
                "gamma": 0.9,
            }
        },
    }

    cfg = WorkbenchConfig.from_dict(raw)
    assert cfg.preprocessing is not None
    assert cfg.preprocessing.illumination_contrast is not None

    knobs = cfg.preprocessing.illumination_contrast
    assert knobs.white_balance == "gray_world"
    assert knobs.clahe is True
    assert knobs.clahe_clip_limit == pytest.approx(2.0)
    assert knobs.clahe_tile_grid_size == (4, 4)
    assert knobs.gamma == pytest.approx(0.9)


def test_build_infer_config_payload_includes_preprocessing() -> None:
    raw = {
        "recipe": "industrial-adapt",
        "dataset": {
            "name": "custom",
            "root": "/tmp/custom",
            "category": "custom",
            "resize": [16, 16],
            "input_mode": "paths",
        },
        "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False, "contamination": 0.1},
        "preprocessing": {
            "illumination_contrast": {
                "white_balance": "gray_world",
                "clahe": True,
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": [4, 4],
                "gamma": 0.9,
            }
        },
    }

    cfg = WorkbenchConfig.from_dict(raw)
    payload = build_infer_config_payload(config=cfg, report={"threshold": 0.5})

    pre = payload["preprocessing"]["illumination_contrast"]
    assert pre["white_balance"] == "gray_world"
    assert pre["clahe"] is True
    assert pre["clahe_clip_limit"] == pytest.approx(2.0)
    assert pre["clahe_tile_grid_size"] == [4, 4]
    assert pre["gamma"] == pytest.approx(0.9)
