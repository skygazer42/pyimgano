from __future__ import annotations

from pathlib import Path

import numpy as np

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.dataset_loader import WorkbenchSplit
from pyimgano.workbench.runtime_split import prepare_workbench_runtime_split


def test_workbench_runtime_split_applies_train_and_test_limits() -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": "/tmp/data",
                "category": "custom",
                "limit_train": 2,
                "limit_test": 1,
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    prepared = prepare_workbench_runtime_split(
        config=cfg,
        split=WorkbenchSplit(
            train_inputs=["train_a.png", "train_b.png", "train_c.png"],
            calibration_inputs=["cal_a.png", "cal_b.png", "cal_c.png"],
            test_inputs=["test_a.png", "test_b.png"],
            test_labels=np.asarray([0, 1], dtype=np.int64),
            test_masks=np.ones((2, 4, 4), dtype=np.uint8),
            input_format=None,
            pixel_skip_reason="skip-reason",
            test_meta=[{"frame": 0}, {"frame": 1}],
        ),
    )

    assert prepared.train_inputs == ["train_a.png", "train_b.png"]
    assert prepared.calibration_inputs == ["cal_a.png", "cal_b.png"]
    assert prepared.test_inputs == ["test_a.png"]
    assert np.asarray(prepared.test_labels).tolist() == [0]
    assert prepared.test_masks is not None
    assert np.asarray(prepared.test_masks).shape == (1, 4, 4)
    assert prepared.test_meta == [{"frame": 0}]
    assert prepared.pixel_skip_reason == "skip-reason"


def test_workbench_runtime_split_preserves_full_split_without_limits(tmp_path: Path) -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path),
                "category": "custom",
                "input_mode": "numpy",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    prepared = prepare_workbench_runtime_split(
        config=cfg,
        split=WorkbenchSplit(
            train_inputs=[np.zeros((2, 2, 3), dtype=np.uint8)],
            calibration_inputs=[np.ones((2, 2, 3), dtype=np.uint8)],
            test_inputs=[np.full((2, 2, 3), fill_value=2, dtype=np.uint8)],
            test_labels=np.asarray([1], dtype=np.int64),
            test_masks=None,
            input_format="rgb_u8_hwc",
            pixel_skip_reason=None,
            test_meta=None,
        ),
    )

    assert len(prepared.train_inputs) == 1
    assert len(prepared.calibration_inputs) == 1
    assert len(prepared.test_inputs) == 1
    assert prepared.input_format == "rgb_u8_hwc"
    assert prepared.test_masks is None
    assert prepared.test_meta is None


def test_workbench_runtime_split_creates_disjoint_calibration_holdout_when_reused() -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": "/tmp/data",
                "category": "custom",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    shared = ["train_a.png", "train_b.png", "train_c.png", "train_d.png", "train_e.png"]
    split = WorkbenchSplit(
        train_inputs=list(shared),
        calibration_inputs=list(shared),
        test_inputs=["test.png"],
        test_labels=np.asarray([0], dtype=np.int64),
        test_masks=None,
        input_format=None,
        pixel_skip_reason=None,
        test_meta=None,
    )

    prepared_a = prepare_workbench_runtime_split(config=cfg, split=split)
    prepared_b = prepare_workbench_runtime_split(config=cfg, split=split)

    assert prepared_a.train_inputs == prepared_b.train_inputs
    assert prepared_a.calibration_inputs == prepared_b.calibration_inputs
    assert set(prepared_a.train_inputs).isdisjoint(set(prepared_a.calibration_inputs))
    assert sorted(prepared_a.train_inputs + prepared_a.calibration_inputs) == sorted(shared)
    assert len(prepared_a.calibration_inputs) == 1
    assert len(prepared_a.train_inputs) == 4
