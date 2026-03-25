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


def _calibration_matches_train(train_inputs: list[Any], calibration_inputs: list[Any]) -> bool:
    if len(train_inputs) != len(calibration_inputs):
        return False
    return all(train_item is calibration_item for train_item, calibration_item in zip(train_inputs, calibration_inputs))


def _split_calibration_holdout(
    train_inputs: list[Any],
    *,
    seed: int | None,
    fraction: float = 0.2,
) -> tuple[list[Any], list[Any]]:
    if len(train_inputs) <= 1:
        return list(train_inputs), list(train_inputs)

    calibration_count = int(np.ceil(float(len(train_inputs)) * float(fraction)))
    calibration_count = max(1, min(len(train_inputs) - 1, calibration_count))

    indices = np.arange(len(train_inputs), dtype=int)
    rng = np.random.default_rng(0 if seed is None else int(seed))
    rng.shuffle(indices)

    calibration_idx = set(int(i) for i in sorted(indices[:calibration_count]))
    calibration_inputs = [train_inputs[i] for i in range(len(train_inputs)) if i in calibration_idx]
    fit_inputs = [train_inputs[i] for i in range(len(train_inputs)) if i not in calibration_idx]
    return fit_inputs, calibration_inputs


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

    if _calibration_matches_train(train_inputs, calibration_inputs):
        train_inputs, calibration_inputs = _split_calibration_holdout(
            list(train_inputs),
            seed=config.seed,
        )

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
