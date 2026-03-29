from __future__ import annotations

import numpy as np
import pytest


def test_resolve_rejection_threshold_validates_range() -> None:
    from pyimgano.inference.runtime_support import resolve_rejection_threshold

    assert resolve_rejection_threshold(None) is None
    assert resolve_rejection_threshold(0.5) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="reject_confidence_below must be in"):
        resolve_rejection_threshold(0.0)


def test_apply_rejection_policy_marks_low_confidence_labels() -> None:
    from pyimgano.inference.runtime_support import apply_rejection_policy

    labels, rejected = apply_rejection_policy(
        detector=type("_Det", (), {"reject_label": -7})(),
        labels=np.asarray([0, 1], dtype=np.int64),
        confidences=np.asarray([0.9, 0.4], dtype=np.float64),
        reject_confidence_below=0.5,
        reject_label=None,
    )

    assert labels.tolist() == [0, -7]
    assert rejected.tolist() == [False, True]


def test_normalize_inputs_rejects_mixed_input_types() -> None:
    from pyimgano.inference.runtime_support import normalize_inputs

    with pytest.raises(TypeError, match="Mixed input types are not supported"):
        normalize_inputs(
            ["a.png", np.zeros((2, 2, 3), dtype=np.uint8)],
            input_format=None,
        )
