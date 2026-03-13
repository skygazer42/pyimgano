from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pyimgano.reporting.report import stamp_report_payload
from pyimgano.services.workbench_service import WorkbenchThresholdCalibration
from pyimgano.workbench.config import WorkbenchConfig


@dataclass(frozen=True)
class WorkbenchCategoryReportInputs:
    config: WorkbenchConfig
    recipe_name: str
    category: str
    train_count: int
    calibration_count: int
    test_labels: np.ndarray
    test_masks: np.ndarray | None
    pixel_skip_reason: str | None
    threshold_used: float
    threshold_calibration: WorkbenchThresholdCalibration
    eval_results: Mapping[str, Any]
    training_report: dict[str, Any] | None = None
    checkpoint_meta: dict[str, Any] | None = None


def _build_dataset_summary(inputs: WorkbenchCategoryReportInputs) -> dict[str, Any]:
    test_labels = np.asarray(inputs.test_labels)
    dataset_summary: dict[str, Any] = {
        "train_count": int(inputs.train_count),
        "calibration_count": int(inputs.calibration_count),
        "test_count": int(len(test_labels)),
        "test_anomaly_count": int(np.sum(test_labels == 1)),
        "test_anomaly_ratio": (float(np.mean(test_labels == 1)) if len(test_labels) > 0 else None),
    }
    if inputs.pixel_skip_reason is not None:
        dataset_summary["pixel_metrics"] = {
            "enabled": False,
            "reason": str(inputs.pixel_skip_reason),
        }
    elif inputs.test_masks is None:
        dataset_summary["pixel_metrics"] = {
            "enabled": False,
            "reason": "No ground-truth test masks available.",
        }
    else:
        dataset_summary["pixel_metrics"] = {"enabled": True, "reason": None}
    return dataset_summary


def build_workbench_category_report(
    *,
    inputs: WorkbenchCategoryReportInputs,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": str(inputs.config.dataset.name),
        "category": str(inputs.category),
        "model": str(inputs.config.model.name),
        "recipe": str(inputs.recipe_name),
        "seed": (int(inputs.config.seed) if inputs.config.seed is not None else None),
        "input_mode": str(inputs.config.dataset.input_mode),
        "device": str(inputs.config.model.device),
        "preset": inputs.config.model.preset,
        "resize": [
            int(inputs.config.dataset.resize[0]),
            int(inputs.config.dataset.resize[1]),
        ],
        "threshold": float(inputs.threshold_used),
        "threshold_provenance": {
            "method": "quantile",
            "quantile": float(inputs.threshold_calibration.quantile),
            "source": str(inputs.threshold_calibration.quantile_source),
            "contamination": float(inputs.config.model.contamination),
            "calibration_count": int(inputs.calibration_count),
        },
        "dataset_summary": _build_dataset_summary(inputs),
        "results": dict(inputs.eval_results),
    }
    if inputs.pixel_skip_reason is not None:
        payload["pixel_metrics_status"] = {
            "enabled": False,
            "reason": str(inputs.pixel_skip_reason),
        }
    if inputs.training_report is not None:
        payload["training"] = dict(inputs.training_report)
    if inputs.checkpoint_meta is not None:
        payload["checkpoint"] = dict(inputs.checkpoint_meta)
    return stamp_report_payload(payload)


__all__ = ["WorkbenchCategoryReportInputs", "build_workbench_category_report"]
