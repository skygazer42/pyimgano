from __future__ import annotations

import json

from pyimgano.services.infer_context_service import (
    InferConfigContextRequest,
    prepare_infer_config_context,
)


def test_prepare_infer_config_context_returns_threshold_and_checkpoint(tmp_path) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "model.pt").write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            }
        ),
        encoding="utf-8",
    )

    context = prepare_infer_config_context(
        InferConfigContextRequest(config_path=str(infer_cfg_path))
    )

    assert context.model_name == "vision_ecod"
    assert context.threshold == 0.7
    assert context.trained_checkpoint_path is not None
    assert context.trained_checkpoint_path.endswith("model.pt")
