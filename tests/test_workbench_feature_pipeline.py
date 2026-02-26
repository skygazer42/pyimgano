from __future__ import annotations

import numpy as np


def test_vision_feature_pipeline_model_smoke() -> None:
    import pyimgano.models  # noqa: F401 - registry population

    from pyimgano.models import create_model

    det = create_model(
        "vision_feature_pipeline",
        contamination=0.1,
        core_detector="core_loop",
        core_kwargs={"n_neighbors": 1},
        feature_extractor={"name": "hog", "kwargs": {"resize_hw": [32, 32]}},
    )

    imgs = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.ones((32, 32, 3), dtype=np.uint8) * 255,
        np.pad(np.ones((16, 16, 3), dtype=np.uint8) * 120, ((8, 8), (8, 8), (0, 0))),
    ]
    det.fit(imgs)
    scores = det.decision_function(imgs)
    labels = det.predict(imgs)

    assert scores.shape == (len(imgs),)
    assert labels.shape == (len(imgs),)
    assert set(labels).issubset({0, 1})
    assert np.all(np.isfinite(scores))


def test_export_infer_config_payload_keeps_feature_pipeline_kwargs() -> None:
    from pyimgano.workbench.config import WorkbenchConfig
    from pyimgano.workbench.runner import build_infer_config_payload

    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": ".", "category": "all", "resize": [32, 32]},
            "model": {
                "name": "vision_feature_pipeline",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
                "model_kwargs": {
                    "core_detector": "core_loop",
                    "core_kwargs": {"n_neighbors": 1},
                    "feature_extractor": {"name": "hog", "kwargs": {"resize_hw": [32, 32]}},
                },
            },
            "output": {"save_run": False, "per_image_jsonl": False},
        }
    )

    payload = build_infer_config_payload(config=cfg, report={})
    mk = payload["model"]["model_kwargs"]
    assert mk["core_detector"] == "core_loop"
    assert mk["core_kwargs"]["n_neighbors"] == 1
    assert mk["feature_extractor"]["name"] == "hog"
