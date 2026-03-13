from __future__ import annotations

import numpy as np

from pyimgano.reporting.report import REPORT_SCHEMA_VERSION
from pyimgano.services.workbench_service import WorkbenchThresholdCalibration
from pyimgano.workbench.category_report import (
    WorkbenchCategoryReportInputs,
    build_workbench_category_report,
)
from pyimgano.workbench.config import WorkbenchConfig


def test_workbench_category_report_builds_stamped_payload_with_dataset_summary() -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": "/tmp/data",
                "category": "custom",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {
                "name": "vision_ecod",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "output": {"save_run": False},
        }
    )

    payload = build_workbench_category_report(
        inputs=WorkbenchCategoryReportInputs(
            config=cfg,
            recipe_name="industrial-adapt",
            category="custom",
            train_count=3,
            calibration_count=2,
            test_labels=np.asarray([0, 1, 1, 0], dtype=np.int64),
            test_masks=None,
            pixel_skip_reason=None,
            threshold_used=0.42,
            threshold_calibration=WorkbenchThresholdCalibration(
                threshold=0.42,
                quantile=0.9,
                quantile_source="default",
            ),
            eval_results={"auroc": 0.8, "threshold": 0.42},
            training_report={"fit_kwargs_used": {"epochs": 2}},
            checkpoint_meta={"path": "checkpoints/custom/model.pt"},
        )
    )

    assert payload["dataset"] == "custom"
    assert payload["category"] == "custom"
    assert payload["recipe"] == "industrial-adapt"
    assert payload["threshold"] == 0.42
    assert payload["threshold_provenance"] == {
        "method": "quantile",
        "quantile": 0.9,
        "source": "default",
        "contamination": 0.1,
        "calibration_count": 2,
    }
    assert payload["dataset_summary"] == {
        "train_count": 3,
        "calibration_count": 2,
        "test_count": 4,
        "test_anomaly_count": 2,
        "test_anomaly_ratio": 0.5,
        "pixel_metrics": {
            "enabled": False,
            "reason": "No ground-truth test masks available.",
        },
    }
    assert payload["training"] == {"fit_kwargs_used": {"epochs": 2}}
    assert payload["checkpoint"] == {"path": "checkpoints/custom/model.pt"}
    assert payload["schema_version"] == int(REPORT_SCHEMA_VERSION)
    assert isinstance(payload["timestamp_utc"], str)
    assert "pixel_metrics_status" not in payload


def test_workbench_category_report_surfaces_pixel_skip_reason() -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "manifest",
                "root": "/tmp/data",
                "manifest_path": "/tmp/data/manifest.jsonl",
                "category": "bottle",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {
                "name": "vision_ecod",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.2,
            },
            "output": {"save_run": False},
        }
    )

    payload = build_workbench_category_report(
        inputs=WorkbenchCategoryReportInputs(
            config=cfg,
            recipe_name="industrial-adapt",
            category="bottle",
            train_count=1,
            calibration_count=1,
            test_labels=np.asarray([0, 1], dtype=np.int64),
            test_masks=None,
            pixel_skip_reason="Missing mask_path for 1 anomaly test sample(s).",
            threshold_used=0.3,
            threshold_calibration=WorkbenchThresholdCalibration(
                threshold=0.3,
                quantile=0.8,
                quantile_source="config",
            ),
            eval_results={"auroc": 1.0, "threshold": 0.3},
        )
    )

    assert payload["dataset_summary"]["pixel_metrics"] == {
        "enabled": False,
        "reason": "Missing mask_path for 1 anomaly test sample(s).",
    }
    assert payload["pixel_metrics_status"] == {
        "enabled": False,
        "reason": "Missing mask_path for 1 anomaly test sample(s).",
    }
