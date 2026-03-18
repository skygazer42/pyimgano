from __future__ import annotations

import numpy as np


def test_build_split_fingerprint_is_stable_for_identical_path_splits() -> None:
    from pyimgano.reporting.split_fingerprint import build_split_fingerprint

    fp_a = build_split_fingerprint(
        train_inputs=["/tmp/a.png", "/tmp/b.png"],
        calibration_inputs=["/tmp/a.png"],
        test_inputs=["/tmp/c.png"],
        test_labels=np.asarray([1], dtype=np.int64),
        input_format=None,
        test_meta=[{"station": "s1"}],
    )
    fp_b = build_split_fingerprint(
        train_inputs=["/tmp/a.png", "/tmp/b.png"],
        calibration_inputs=["/tmp/a.png"],
        test_inputs=["/tmp/c.png"],
        test_labels=np.asarray([1], dtype=np.int64),
        input_format=None,
        test_meta=[{"station": "s1"}],
    )

    assert fp_a["sha256"] == fp_b["sha256"]
    assert fp_a["train_count"] == 2
    assert fp_a["calibration_count"] == 1
    assert fp_a["test_count"] == 1


def test_build_split_fingerprint_changes_when_test_labels_change() -> None:
    from pyimgano.reporting.split_fingerprint import build_split_fingerprint

    fp_a = build_split_fingerprint(
        train_inputs=["/tmp/a.png"],
        calibration_inputs=[],
        test_inputs=["/tmp/c.png"],
        test_labels=np.asarray([0], dtype=np.int64),
    )
    fp_b = build_split_fingerprint(
        train_inputs=["/tmp/a.png"],
        calibration_inputs=[],
        test_inputs=["/tmp/c.png"],
        test_labels=np.asarray([1], dtype=np.int64),
    )

    assert fp_a["sha256"] != fp_b["sha256"]
