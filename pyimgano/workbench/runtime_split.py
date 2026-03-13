from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.dataset_loader import WorkbenchSplit


@dataclass(frozen=True)
class PreparedWorkbenchSplit:
    train_inputs: list[Any]
    calibration_inputs: list[Any]
    test_inputs: list[Any]
    test_labels: np.ndarray
    test_masks: np.ndarray | None
    input_format: str | None
    pixel_skip_reason: str | None = None
    test_meta: list[Mapping[str, Any] | None] | None = None


def prepare_workbench_runtime_split(
    *,
    config: WorkbenchConfig,
    split: WorkbenchSplit,
) -> PreparedWorkbenchSplit:
    train_inputs = list(split.train_inputs)
    calibration_inputs = list(split.calibration_inputs)
    test_inputs = list(split.test_inputs)
    test_labels = np.asarray(split.test_labels)
    test_masks = np.asarray(split.test_masks) if split.test_masks is not None else None
    test_meta = list(split.test_meta) if split.test_meta is not None else None

    if config.dataset.limit_train is not None:
        limit_train = int(config.dataset.limit_train)
        train_inputs = list(train_inputs)[:limit_train]
        calibration_inputs = list(calibration_inputs)[:limit_train]

    if config.dataset.limit_test is not None:
        limit_test = int(config.dataset.limit_test)
        test_inputs = list(test_inputs)[:limit_test]
        test_labels = np.asarray(test_labels)[:limit_test]
        if test_masks is not None:
            test_masks = np.asarray(test_masks)[:limit_test]
        if test_meta is not None:
            test_meta = list(test_meta)[:limit_test]

    return PreparedWorkbenchSplit(
        train_inputs=train_inputs,
        calibration_inputs=calibration_inputs,
        test_inputs=test_inputs,
        test_labels=np.asarray(test_labels),
        test_masks=test_masks,
        input_format=split.input_format,
        pixel_skip_reason=split.pixel_skip_reason,
        test_meta=test_meta,
    )


__all__ = ["PreparedWorkbenchSplit", "prepare_workbench_runtime_split"]
