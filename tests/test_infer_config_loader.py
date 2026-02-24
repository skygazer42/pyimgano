from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_infer_config_resolves_checkpoint_relative_to_artifacts_dir(tmp_path: Path) -> None:
    from pyimgano.inference.config import resolve_infer_checkpoint_path

    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "custom").mkdir(parents=True, exist_ok=True)

    ckpt = run_dir / "checkpoints" / "custom" / "model.pt"
    ckpt.write_text("ckpt", encoding="utf-8")

    cfg_path = run_dir / "artifacts" / "infer_config.json"
    payload = {
        "from_run": str(run_dir),
        "checkpoint": {"path": "checkpoints/custom/model.pt"},
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved = resolve_infer_checkpoint_path(payload, config_path=cfg_path)
    assert resolved == ckpt


def test_infer_config_category_selection_requires_flag_when_ambiguous(tmp_path: Path) -> None:
    from pyimgano.inference.config import select_infer_category

    payload = {
        "category": "all",
        "per_category": {
            "a": {"threshold": 0.1},
            "b": {"threshold": 0.2},
        },
    }
    with pytest.raises(ValueError) as exc:
        select_infer_category(payload, category=None)
    msg = str(exc.value).lower()
    assert "infer-config" in msg
    assert "infer-category" in msg


def test_infer_config_category_selection_propagates_threshold_and_checkpoint(tmp_path: Path) -> None:
    from pyimgano.inference.config import select_infer_category

    payload = {
        "category": "all",
        "threshold": 0.5,
        "threshold_provenance": {"method": "quantile", "quantile": 0.5},
        "checkpoint": {"path": "ignored.pt"},
        "per_category": {
            "cat": {
                "threshold": 0.7,
                "threshold_provenance": {"method": "quantile", "quantile": 0.7},
                "checkpoint": {"path": "checkpoints/cat/model.pt"},
            },
        },
    }
    out = select_infer_category(payload, category="cat")
    assert out["category"] == "cat"
    assert out["threshold"] == 0.7
    assert out["threshold_provenance"]["quantile"] == 0.7
    assert out["checkpoint"]["path"] == "checkpoints/cat/model.pt"
    assert "per_category" not in out
