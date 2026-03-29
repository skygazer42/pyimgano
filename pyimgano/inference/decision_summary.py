from __future__ import annotations

from typing import Any


def build_decision_summary(
    *,
    label: int | None,
    label_confidence: float | None,
    rejected: bool | None,
) -> dict[str, Any]:
    rejected_flag = bool(rejected)
    threshold_applied = label is not None
    has_confidence = label_confidence is not None

    if rejected_flag:
        decision = "rejected_low_confidence"
        requires_review = True
        review_reason = "low_confidence"
    elif label is None:
        decision = "score_only"
        requires_review = False
        review_reason = "unthresholded_score"
    elif int(label) == 0:
        decision = "normal"
        requires_review = False
        review_reason = "none"
    else:
        decision = "anomalous"
        requires_review = True
        review_reason = "anomaly_label"

    return {
        "decision": str(decision),
        "threshold_applied": bool(threshold_applied),
        "has_confidence": bool(has_confidence),
        "rejected": bool(rejected_flag),
        "requires_review": bool(requires_review),
        "review_reason": str(review_reason),
    }


def maybe_build_decision_summary(
    *,
    label: int | None,
    label_confidence: float | None,
    rejected: bool | None,
) -> dict[str, Any] | None:
    if label is None and label_confidence is None and rejected is None:
        return None
    return build_decision_summary(
        label=label,
        label_confidence=label_confidence,
        rejected=rejected,
    )


__all__ = [
    "build_decision_summary",
    "maybe_build_decision_summary",
]
