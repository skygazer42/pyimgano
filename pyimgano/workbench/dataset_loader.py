from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.manifest_split_policy import build_manifest_split_policy


@dataclass(frozen=True)
class WorkbenchSplit:
    train_inputs: list[Any]
    calibration_inputs: list[Any]
    test_inputs: list[Any]
    test_labels: np.ndarray
    test_masks: np.ndarray | None
    input_format: str | None
    pixel_skip_reason: str | None = None
    test_meta: list[Mapping[str, Any] | None] | None = None


def _load_paths_split(
    *,
    config: WorkbenchConfig,
    category: str,
    load_masks: bool,
) -> WorkbenchSplit:
    dataset = str(config.dataset.name)
    if dataset.lower() == "manifest":
        if config.dataset.input_mode != "paths":
            raise ValueError(
                "dataset.name='manifest' currently supports only dataset.input_mode='paths'."
            )
        if config.dataset.manifest_path is None:
            raise ValueError("dataset.manifest_path is required when dataset.name='manifest'.")

        from pyimgano.datasets.manifest import load_manifest_benchmark_split

        policy = build_manifest_split_policy(config=config)
        split = load_manifest_benchmark_split(
            manifest_path=str(config.dataset.manifest_path),
            root_fallback=str(config.dataset.root),
            category=str(category),
            resize=tuple(config.dataset.resize),
            load_masks=bool(load_masks),
            split_policy=policy,
        )
        calibration = (
            list(split.calibration_paths) if split.calibration_paths else list(split.train_paths)
        )
        return WorkbenchSplit(
            train_inputs=list(split.train_paths),
            calibration_inputs=calibration,
            test_inputs=list(split.test_paths),
            test_labels=np.asarray(split.test_labels),
            test_masks=split.test_masks,
            input_format=None,
            pixel_skip_reason=split.pixel_skip_reason,
            test_meta=split.test_meta,
        )

    from pyimgano.pipelines.mvtec_visa import load_benchmark_split

    split = load_benchmark_split(
        dataset=dataset,  # type: ignore[arg-type]
        root=str(config.dataset.root),
        category=str(category),
        resize=tuple(config.dataset.resize),
        load_masks=bool(load_masks),
    )
    train_inputs = list(split.train_paths)
    return WorkbenchSplit(
        train_inputs=train_inputs,
        calibration_inputs=list(train_inputs),
        test_inputs=list(split.test_paths),
        test_labels=np.asarray(split.test_labels),
        test_masks=split.test_masks,
        input_format=None,
        pixel_skip_reason=None,
        test_meta=None,
    )


def _load_numpy_split(
    *,
    dataset: str,
    root: str,
    category: str,
    resize: tuple[int, int],
    load_masks: bool,
) -> WorkbenchSplit:
    from pyimgano.datasets import load_dataset

    ds = load_dataset(  # nosec B615 - pyimgano.datasets.load_dataset, not Hugging Face Hub
        dataset,
        root,
        category=category,
        resize=tuple(resize),
        load_masks=bool(load_masks),
    )
    train_data = np.asarray(ds.get_train_data())
    test_data, test_labels, test_masks = ds.get_test_data()
    test_arr = np.asarray(test_data)

    train_inputs = [np.asarray(train_data[i]) for i in range(int(train_data.shape[0]))]
    test_inputs = [np.asarray(test_arr[i]) for i in range(int(test_arr.shape[0]))]
    return WorkbenchSplit(
        train_inputs=train_inputs,
        calibration_inputs=list(train_inputs),
        test_inputs=test_inputs,
        test_labels=np.asarray(test_labels),
        test_masks=test_masks,
        input_format="rgb_u8_hwc",
        pixel_skip_reason=None,
        test_meta=None,
    )


def load_workbench_split(
    *,
    config: WorkbenchConfig,
    category: str,
    load_masks: bool,
) -> WorkbenchSplit:
    input_mode = str(config.dataset.input_mode)
    dataset = str(config.dataset.name)

    if input_mode == "paths":
        return _load_paths_split(
            config=config,
            category=str(category),
            load_masks=bool(load_masks),
        )

    if input_mode == "numpy":
        if dataset.lower() == "manifest":
            raise ValueError(
                "dataset.name='manifest' currently supports only dataset.input_mode='paths'."
            )
        return _load_numpy_split(
            dataset=dataset,
            root=str(config.dataset.root),
            category=str(category),
            resize=tuple(config.dataset.resize),
            load_masks=bool(load_masks),
        )

    raise ValueError(f"Unknown input_mode: {config.dataset.input_mode!r}. Choose from: paths, numpy.")


def list_workbench_categories(*, config: WorkbenchConfig) -> list[str]:
    from pyimgano.datasets.catalog import list_dataset_categories

    return list(
        list_dataset_categories(
            dataset=str(config.dataset.name),
            root=str(config.dataset.root),
            manifest_path=(
                str(config.dataset.manifest_path) if config.dataset.manifest_path is not None else None
            ),
        )
    )


__all__ = ["WorkbenchSplit", "load_workbench_split", "list_workbench_categories"]
