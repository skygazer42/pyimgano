from __future__ import annotations

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.preflight import run_preflight


def test_preflight_errors_when_preprocessing_enabled_on_non_numpy_model(tmp_path) -> None:
    class _DummyPathOnlyDetector:
        def __init__(self, **_kwargs):  # noqa: ANN003 - test stub
            pass

    MODEL_REGISTRY.register(
        "test_preflight_path_only_detector",
        _DummyPathOnlyDetector,
        tags=("vision", "classical"),
        overwrite=True,
    )

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "does_not_exist"),
                "category": "custom",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {
                "name": "test_preflight_path_only_detector",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "preprocessing": {
                "illumination_contrast": {
                    "white_balance": "gray_world",
                    "clahe": True,
                }
            },
        }
    )

    report = run_preflight(config=cfg)
    codes = {issue.code for issue in report.issues}

    assert "PREPROCESSING_REQUIRES_NUMPY_MODEL" in codes

