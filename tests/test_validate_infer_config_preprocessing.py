from __future__ import annotations

import pytest

from pyimgano.inference.validate_infer_config import validate_infer_config_payload


def test_validate_infer_config_coerces_preprocessing_knobs() -> None:
    payload = {
        "model": {
            "name": "vision_ecod",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
            "model_kwargs": {},
            "checkpoint_path": None,
        },
        "preprocessing": {
            "illumination_contrast": {
                "white_balance": "gray_world",
                "homomorphic": 0,
                "clahe": 1,
                "clahe_clip_limit": "2.0",
                "clahe_tile_grid_size": ["8", "8"],
                "gamma": "0.9",
                "contrast_stretch": False,
            }
        },
    }

    validated = validate_infer_config_payload(payload, check_files=False)
    pre = validated.payload["preprocessing"]["illumination_contrast"]

    assert pre["white_balance"] == "gray_world"
    assert pre["homomorphic"] is False
    assert pre["clahe"] is True
    assert pre["clahe_clip_limit"] == pytest.approx(2.0)
    assert pre["clahe_tile_grid_size"] == [8, 8]
    assert pre["gamma"] == pytest.approx(0.9)
