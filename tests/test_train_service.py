from __future__ import annotations

import json

from pyimgano.services.train_service import TrainRunRequest, build_train_dry_run_payload


def test_build_train_dry_run_payload_returns_config_payload(tmp_path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    payload = build_train_dry_run_payload(TrainRunRequest(config_path=str(cfg_path)))

    assert payload["config"]["recipe"] == "industrial-adapt"
