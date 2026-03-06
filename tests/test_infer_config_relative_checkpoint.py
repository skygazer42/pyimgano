from __future__ import annotations

import json
from pathlib import Path


def test_resolve_infer_checkpoint_path_from_config_sibling_dir(tmp_path: Path) -> None:
    from pyimgano.inference.config import resolve_infer_checkpoint_path

    deploy_dir = tmp_path / "deploy"
    weights_dir = tmp_path / "weights"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    ckpt = weights_dir / "model.pt"
    ckpt.write_text("checkpoint", encoding="utf-8")

    cfg_path = deploy_dir / "infer_config.json"
    payload = {
        "checkpoint": {"path": "../weights/model.pt"},
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved = resolve_infer_checkpoint_path(payload, config_path=cfg_path)
    assert resolved == ckpt


def test_resolve_infer_model_checkpoint_path_from_config_sibling_dir(tmp_path: Path) -> None:
    from pyimgano.inference.config import resolve_infer_model_checkpoint_path

    deploy_dir = tmp_path / "deploy"
    model_dir = tmp_path / "artifacts"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = model_dir / "embedder.ts"
    checkpoint.write_text("torchscript", encoding="utf-8")

    cfg_path = deploy_dir / "infer_config.json"
    payload = {
        "model": {
            "name": "vision_embedding_core",
            "checkpoint_path": "../artifacts/embedder.ts",
        }
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved = resolve_infer_model_checkpoint_path(payload, config_path=cfg_path)
    assert resolved == checkpoint
