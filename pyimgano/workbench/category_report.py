from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pyimgano.reporting.evaluation_contract import build_evaluation_contract
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
    split_fingerprint: dict[str, Any] | None = None


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


def _fallback_dataset_readiness(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    issue_codes: list[str] = []
    issue_details: list[dict[str, str]] = []

    train_count = int(dataset_summary.get("train_count", 0) or 0)
    test_count = int(dataset_summary.get("test_count", 0) or 0)
    test_anomaly_count = int(dataset_summary.get("test_anomaly_count", 0) or 0)
    test_normal_count = max(0, test_count - test_anomaly_count)
    pixel_metrics = dict(dataset_summary.get("pixel_metrics", {}))

    def _append(code: str, message: str) -> None:
        if code in issue_codes:
            return
        issue_codes.append(code)
        issue_details.append({"code": code, "message": message})

    if train_count <= 0:
        _append(
            "MISSING_TRAIN_SPLIT", "Train split is missing; image-level benchmarking cannot run."
        )
    if test_normal_count <= 0:
        _append(
            "MISSING_TEST_NORMAL",
            "Test normal split is missing; false-positive estimates and image metrics are incomplete.",
        )
    if test_anomaly_count <= 0:
        _append(
            "MISSING_TEST_ANOMALY",
            "Test anomaly split is missing; anomaly recall and AUROC cannot be computed.",
        )
    if not bool(pixel_metrics.get("enabled")):
        _append(
            "PIXEL_METRICS_UNAVAILABLE",
            "Pixel metrics are unavailable because anomaly masks are missing or no anomalous test samples carry masks.",
        )
    if train_count < 16:
        _append(
            "FEWSHOT_TRAIN_SET",
            "Train split has fewer than 16 normal samples; results may be unstable.",
        )

    if "MISSING_TRAIN_SPLIT" in issue_codes or "MISSING_TEST_ANOMALY" in issue_codes:
        status = "error"
    elif issue_codes:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": str(status),
        "issue_codes": issue_codes,
        "issue_details": issue_details,
    }


def _build_dataset_readiness(
    *,
    inputs: WorkbenchCategoryReportInputs,
    dataset_summary: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from pyimgano.datasets.inspection import profile_dataset_target

        dataset_name = str(inputs.config.dataset.name)
        category = str(inputs.category)
        if dataset_name.lower() == "manifest":
            manifest_path = inputs.config.dataset.manifest_path
            if manifest_path is None:
                raise ValueError("manifest_path missing from config")
            if not Path(str(manifest_path)).exists():
                raise FileNotFoundError(str(manifest_path))
            payload = profile_dataset_target(
                target=str(manifest_path),
                dataset="manifest",
                category=category,
                root_fallback=str(inputs.config.dataset.root),
            )
        else:
            if not Path(str(inputs.config.dataset.root)).exists():
                raise FileNotFoundError(str(inputs.config.dataset.root))
            payload = profile_dataset_target(
                target=str(inputs.config.dataset.root),
                dataset=dataset_name,
                category=category,
            )
        readiness = payload.get("readiness", None)
        if isinstance(readiness, dict):
            return dict(readiness)
    except Exception:
        pass
    return _fallback_dataset_readiness(dataset_summary)


def _evaluation_metric_names(eval_results: Mapping[str, Any]) -> list[str]:
    names = [
        "auroc",
        "average_precision",
        "pixel_auroc",
        "pixel_average_precision",
        "aupro",
        "pixel_segf1",
    ]
    for key in ("auroc", "average_precision"):
        if isinstance(eval_results.get(key), (int, float)):
            names.append(str(key))

    pixel_metrics = eval_results.get("pixel_metrics", None)
    if isinstance(pixel_metrics, Mapping):
        for key in ("pixel_auroc", "pixel_average_precision", "aupro", "pixel_segf1"):
            if isinstance(pixel_metrics.get(key), (int, float)):
                names.append(str(key))

    return sorted({str(name) for name in names})


def build_workbench_category_report(
    *,
    inputs: WorkbenchCategoryReportInputs,
) -> dict[str, Any]:
    dataset_summary = _build_dataset_summary(inputs)
    dataset_readiness = _build_dataset_readiness(
        inputs=inputs,
        dataset_summary=dataset_summary,
    )
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
        "dataset_summary": dataset_summary,
        "dataset_readiness": dataset_readiness,
        "evaluation_contract": build_evaluation_contract(
            metric_names=_evaluation_metric_names(inputs.eval_results),
            primary_metric="auroc",
            ranking_metric="auroc",
            pixel_metrics_enabled=bool(dataset_summary["pixel_metrics"]["enabled"]),
        ),
        "results": dict(inputs.eval_results),
    }
    if inputs.threshold_calibration.score_summary is not None:
        payload["threshold_provenance"]["score_summary"] = dict(
            inputs.threshold_calibration.score_summary
        )
    if inputs.split_fingerprint is not None:
        payload["split_fingerprint"] = dict(inputs.split_fingerprint)
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
