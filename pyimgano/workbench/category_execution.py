from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import pyimgano.services.workbench_service as workbench_service
from pyimgano.reporting.split_fingerprint import build_split_fingerprint
from pyimgano.train_progress import get_active_train_progress_reporter
from pyimgano.workbench.adaptation_runtime import build_postprocess
from pyimgano.workbench.category_outputs import (
    WorkbenchCategoryOutputs,
    save_workbench_category_outputs,
)
from pyimgano.workbench.category_report import (
    WorkbenchCategoryReportInputs,
    build_workbench_category_report,
)
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.dataset_loader import load_workbench_split
from pyimgano.workbench.detector_setup import build_workbench_runtime_detector
from pyimgano.workbench.inference_runtime import run_workbench_inference
from pyimgano.workbench.runtime_split import prepare_workbench_runtime_split
from pyimgano.workbench.training_runtime import run_workbench_training


def run_workbench_category(
    *,
    config: WorkbenchConfig,
    recipe_name: str,
    category: str,
    run_dir: Path | None,
) -> dict[str, Any]:
    reporter = get_active_train_progress_reporter()
    split = prepare_workbench_runtime_split(
        config=config,
        split=load_workbench_split(
            config=config,
            category=str(category),
            load_masks=True,
        ),
    )
    train_inputs = list(split.train_inputs)
    calibration_inputs = list(split.calibration_inputs)
    test_inputs = list(split.test_inputs)
    test_labels = np.asarray(split.test_labels)
    test_masks = split.test_masks
    pixel_skip_reason = split.pixel_skip_reason
    test_meta = split.test_meta
    input_format = split.input_format
    if pixel_skip_reason is not None:
        pixel_metrics_reason = str(pixel_skip_reason)
    elif test_masks is None:
        pixel_metrics_reason = "No ground-truth test masks available."
    else:
        pixel_metrics_reason = None

    reporter.on_dataset_loaded(
        category=str(category),
        train_count=int(len(train_inputs)),
        calibration_count=int(len(calibration_inputs)),
        test_count=int(len(test_inputs)),
        anomaly_count=int(np.sum(test_labels == 1)),
        pixel_metrics_enabled=bool(test_masks is not None and pixel_skip_reason is None),
        pixel_metrics_reason=pixel_metrics_reason,
    )

    detector = build_workbench_runtime_detector(config=config)

    training_result = run_workbench_training(
        detector=detector,
        train_inputs=train_inputs,
        config=config,
        category=str(category),
        run_dir=run_dir,
    )
    detector = training_result.detector
    training_report = training_result.training_report
    checkpoint_meta = training_result.checkpoint_meta
    if training_report is not None:
        reporter.on_training_end(
            category=str(category),
            report=training_report,
            checkpoint_meta=checkpoint_meta,
        )
    threshold_calibration = workbench_service.calibrate_workbench_threshold(
        detector=detector,
        calibration_inputs=list(calibration_inputs),
        input_format=input_format,
    )
    reporter.on_calibration_end(
        category=str(category),
        threshold=float(threshold_calibration.threshold),
        quantile=float(threshold_calibration.quantile),
        source=str(threshold_calibration.quantile_source),
        score_summary=threshold_calibration.score_summary,
    )
    threshold = float(threshold_calibration.threshold)

    postprocess = build_postprocess(config.adaptation.postprocess)
    inference_result = run_workbench_inference(
        detector=detector,
        test_inputs=test_inputs,
        input_format=input_format,
        postprocess=postprocess,
        save_maps=bool(config.adaptation.save_maps),
        test_labels=np.asarray(test_labels),
        test_masks=(np.asarray(test_masks) if test_masks is not None else None),
        threshold=float(threshold),
    )
    scores = np.asarray(inference_result.scores)
    maps_list = inference_result.maps
    eval_results = inference_result.eval_results

    threshold_used = float(eval_results["threshold"])
    split_fingerprint = build_split_fingerprint(
        train_inputs=train_inputs,
        calibration_inputs=calibration_inputs,
        test_inputs=test_inputs,
        test_labels=np.asarray(test_labels),
        input_format=input_format,
        test_meta=test_meta,
    )

    payload = build_workbench_category_report(
        inputs=WorkbenchCategoryReportInputs(
            config=config,
            recipe_name=recipe_name,
            category=str(category),
            train_count=int(len(train_inputs)),
            calibration_count=int(len(calibration_inputs)),
            split_fingerprint=split_fingerprint,
            test_labels=np.asarray(test_labels),
            test_masks=(np.asarray(test_masks) if test_masks is not None else None),
            pixel_skip_reason=pixel_skip_reason,
            threshold_used=float(threshold_used),
            threshold_calibration=threshold_calibration,
            eval_results=eval_results,
            training_report=training_report,
            checkpoint_meta=checkpoint_meta,
        )
    )
    reporter.on_evaluation_end(
        category=str(category),
        results=dict(payload.get("results", {})),
        dataset_summary=dict(payload.get("dataset_summary", {})),
    )

    if run_dir is not None and bool(config.output.save_run):
        save_workbench_category_outputs(
            run_dir=run_dir,
            outputs=WorkbenchCategoryOutputs(
                payload=payload,
                test_inputs=list(test_inputs),
                test_labels=np.asarray(test_labels),
                scores=np.asarray(scores),
                threshold=float(threshold_used),
                maps=list(maps_list) if maps_list is not None else None,
                test_meta=list(test_meta) if test_meta is not None else None,
            ),
            save_maps=bool(config.adaptation.save_maps),
            per_image_jsonl=bool(config.output.per_image_jsonl),
        )

    return payload


__all__ = ["run_workbench_category"]
