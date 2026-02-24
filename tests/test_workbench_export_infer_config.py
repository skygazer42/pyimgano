import json

import numpy as np
import pytest

from pyimgano.models.registry import MODEL_REGISTRY


def test_train_cli_export_infer_config_writes_artifact(tmp_path):
    import cv2

    from pyimgano.train_cli import main

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            self.fit_inputs = list(X)
            return self

        def decision_function(self, X):  # noqa: ANN001
            n = len(list(X))
            if n == 0:
                return np.asarray([], dtype=np.float32)
            return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_export_infer_config_dummy_detector",
        _DummyDetector,
        tags=("classical",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    out_dir = tmp_path / "run_out"
    cfg = {
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
            "name": "test_export_infer_config_dummy_detector",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
        },
        "output": {
            "output_dir": str(out_dir),
            "save_run": True,
            "per_image_jsonl": False,
        },
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path), "--export-infer-config"])
    assert code == 0

    infer_cfg_path = out_dir / "artifacts" / "infer_config.json"
    assert infer_cfg_path.exists()
    payload = json.loads(infer_cfg_path.read_text(encoding="utf-8"))

    assert payload["model"]["name"] == "test_export_infer_config_dummy_detector"
    assert "threshold" in payload
    prov = payload["threshold_provenance"]
    assert prov["method"] == "quantile"
    assert prov["quantile"] == pytest.approx(0.9)
    assert prov["source"] == "contamination"
    assert prov["contamination"] == pytest.approx(0.1)

    defects = payload["defects"]
    assert defects["enabled"] is False
    assert defects["pixel_threshold"] is None
    assert defects["pixel_threshold_strategy"] == "normal_pixel_quantile"
    assert defects["pixel_normal_quantile"] == pytest.approx(0.999)
    assert defects["mask_format"] == "png"
    assert defects["roi_xyxy_norm"] is None
    assert defects["min_area"] == 0
    assert defects["open_ksize"] == 0
    assert defects["close_ksize"] == 0
    assert defects["fill_holes"] is False
    assert defects["max_regions"] is None
