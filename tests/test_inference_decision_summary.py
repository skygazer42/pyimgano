from __future__ import annotations


def test_maybe_build_decision_summary_returns_none_for_empty_inputs() -> None:
    from pyimgano.inference.decision_summary import maybe_build_decision_summary

    assert (
        maybe_build_decision_summary(
            label=None,
            label_confidence=None,
            rejected=None,
        )
        is None
    )


def test_build_decision_summary_marks_rejected_samples_for_review() -> None:
    from pyimgano.inference.decision_summary import build_decision_summary

    summary = build_decision_summary(
        label=-2,
        label_confidence=0.42,
        rejected=True,
    )

    assert summary == {
        "decision": "rejected_low_confidence",
        "threshold_applied": True,
        "has_confidence": True,
        "rejected": True,
        "requires_review": True,
        "review_reason": "low_confidence",
    }


def test_build_decision_summary_marks_unthresholded_scores_without_review() -> None:
    from pyimgano.inference.decision_summary import build_decision_summary

    summary = build_decision_summary(
        label=None,
        label_confidence=None,
        rejected=False,
    )

    assert summary == {
        "decision": "score_only",
        "threshold_applied": False,
        "has_confidence": False,
        "rejected": False,
        "requires_review": False,
        "review_reason": "unthresholded_score",
    }
