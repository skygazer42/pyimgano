from __future__ import annotations

from typing import Any, Mapping


CALIBRATION_CARD_SCHEMA_VERSION = 1


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_score_distribution(provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    score_summary = provenance.get("score_summary", None)
    if not isinstance(score_summary, Mapping):
        return None
    return dict(score_summary)


def _validate_numeric_score_summary_field(
    payload: Mapping[str, Any],
    *,
    name: str,
    key: str,
) -> str | None:
    value = payload.get(key, None)
    if value is not None and not _is_numeric_value(value):
        return f"{name}.{key} must be numeric."
    return None


def _validate_score_summary_quantiles(payload: Mapping[str, Any], *, name: str) -> list[str]:
    quantiles = payload.get("quantiles", None)
    if quantiles is None:
        return []
    if not isinstance(quantiles, Mapping):
        return [f"{name}.quantiles must be a JSON object/dict."]
    return [
        f"{name}.quantiles[{quantile_name!r}] must be numeric."
        for quantile_name, quantile_value in quantiles.items()
        if not _is_numeric_value(quantile_value)
    ]


def _validate_score_summary_payload(payload: Any, *, name: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return [f"{name} must be a JSON object/dict."]

    for key in ("min", "max", "mean", "std"):
        error = _validate_numeric_score_summary_field(payload, name=name, key=key)
        if error is not None:
            errors.append(error)
    count_error = _validate_numeric_score_summary_field(payload, name=name, key="count")
    if count_error is not None:
        errors.append(count_error)
    errors.extend(_validate_score_summary_quantiles(payload, name=name))

    return errors


def _coerce_threshold_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    threshold = payload.get("threshold", None)
    provenance = payload.get("threshold_provenance", None)
    if not _is_numeric_value(threshold) or not isinstance(provenance, Mapping):
        raise ValueError("threshold and threshold_provenance are required to build calibration cards.")
    out = {
        "threshold": float(threshold),
        "provenance": dict(provenance),
    }
    score_distribution = _coerce_score_distribution(provenance)
    if score_distribution is not None:
        out["score_distribution"] = score_distribution
    return out


def _coerce_prediction_policy(report: Mapping[str, Any]) -> dict[str, Any] | None:
    prediction = report.get("prediction", None)
    if not isinstance(prediction, Mapping):
        prediction = {
            "reject_confidence_below": report.get("reject_confidence_below", None),
            "reject_label": report.get("reject_label", None),
        }

    reject_confidence_below = prediction.get("reject_confidence_below", None)
    reject_label = prediction.get("reject_label", None)
    if reject_confidence_below is None and reject_label is None:
        return None

    out: dict[str, Any] = {}
    if reject_confidence_below is not None:
        out["reject_confidence_below"] = float(reject_confidence_below)
    if reject_label is not None:
        out["reject_label"] = int(reject_label)
    return out


def _validate_threshold_payload(payload: Any, *, name: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return [f"{name} must be a JSON object/dict."]

    threshold = payload.get("threshold", None)
    if not _is_numeric_value(threshold):
        errors.append(f"{name}.threshold must be a number.")

    provenance = payload.get("provenance", None)
    if not isinstance(provenance, Mapping):
        errors.append(f"{name}.provenance must be a JSON object/dict.")
    else:
        score_summary = provenance.get("score_summary", None)
        if score_summary is not None:
            errors.extend(
                _validate_score_summary_payload(
                    score_summary,
                    name=f"{name}.provenance.score_summary",
                )
            )

    score_distribution = payload.get("score_distribution", None)
    if score_distribution is not None:
        errors.extend(
            _validate_score_summary_payload(
                score_distribution,
                name=f"{name}.score_distribution",
            )
        )

    return errors


def build_calibration_card_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": int(CALIBRATION_CARD_SCHEMA_VERSION),
        "run_dir": report.get("run_dir", None),
        "dataset": report.get("dataset", None),
        "category": report.get("category", None),
        "model": report.get("model", None),
        "recipe": report.get("recipe", None),
        "pyimgano_version": report.get("pyimgano_version", None),
    }
    split_fingerprint = report.get("split_fingerprint", None)
    if isinstance(split_fingerprint, Mapping):
        payload["split_fingerprint"] = dict(split_fingerprint)

    prediction_policy = _coerce_prediction_policy(report)
    if prediction_policy is not None:
        payload["prediction_policy"] = prediction_policy

    per_category = report.get("per_category", None)
    if isinstance(per_category, Mapping):
        items: dict[str, Any] = {}
        for name, cat_payload in per_category.items():
            if not isinstance(cat_payload, Mapping):
                continue
            try:
                items[str(name)] = _coerce_threshold_payload(cat_payload)
            except ValueError:
                continue
        if not items:
            raise ValueError("No per-category threshold payloads found for calibration card.")
        payload["per_category"] = items
        payload["threshold_context"] = {
            "scope": "per_category",
            "category_count": int(len(items)),
        }
        return payload

    payload["image_threshold"] = _coerce_threshold_payload(report)
    payload["threshold_context"] = {"scope": "image", "category_count": 1}
    return payload


def validate_calibration_card_payload(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if int(payload.get("schema_version", 0) or 0) != int(CALIBRATION_CARD_SCHEMA_VERSION):
        errors.append("Unsupported calibration card schema_version.")

    image_threshold = payload.get("image_threshold", None)
    per_category = payload.get("per_category", None)
    _validate_threshold_sections(errors, image_threshold=image_threshold, per_category=per_category)
    _validate_split_fingerprint(errors, payload.get("split_fingerprint", None))
    _validate_threshold_context(errors, payload.get("threshold_context", None))
    _validate_prediction_policy(errors, payload.get("prediction_policy", None))
    return errors


def _validate_threshold_sections(
    errors: list[str],
    *,
    image_threshold: Any,
    per_category: Any,
) -> None:
    if not isinstance(image_threshold, Mapping) and not isinstance(per_category, Mapping):
        errors.append("Calibration card must provide image_threshold or per_category.")
    elif isinstance(image_threshold, Mapping):
        errors.extend(_validate_threshold_payload(image_threshold, name="image_threshold"))

    if per_category is None:
        return
    if not isinstance(per_category, Mapping):
        errors.append("per_category must be a JSON object/dict.")
        return
    if len(per_category) == 0:
        errors.append("per_category must not be empty.")
        return
    for name, item in per_category.items():
        errors.extend(_validate_threshold_payload(item, name=f"per_category[{name!r}]"))


def _validate_split_fingerprint(errors: list[str], split_fingerprint: Any) -> None:
    if split_fingerprint is None:
        return
    if not isinstance(split_fingerprint, Mapping):
        errors.append("split_fingerprint must be a JSON object/dict.")
        return
    sha256 = split_fingerprint.get("sha256", None)
    if not isinstance(sha256, str) or not sha256.strip():
        errors.append("split_fingerprint.sha256 must be a non-empty string.")


def _validate_threshold_context(errors: list[str], threshold_context: Any) -> None:
    if threshold_context is None:
        return
    if not isinstance(threshold_context, Mapping):
        errors.append("threshold_context must be a JSON object/dict.")
        return
    scope = threshold_context.get("scope", None)
    if not isinstance(scope, str) or scope not in {"image", "per_category"}:
        errors.append("threshold_context.scope must be 'image' or 'per_category'.")
    category_count = threshold_context.get("category_count", None)
    if category_count is not None and not isinstance(category_count, (int, float)):
        errors.append("threshold_context.category_count must be numeric.")


def _validate_prediction_policy(errors: list[str], prediction_policy: Any) -> None:
    if prediction_policy is None:
        return
    if not isinstance(prediction_policy, Mapping):
        errors.append("prediction_policy must be a JSON object/dict.")
        return
    reject_confidence_below = prediction_policy.get("reject_confidence_below", None)
    if reject_confidence_below is not None:
        if not isinstance(reject_confidence_below, (int, float)):
            errors.append("prediction_policy.reject_confidence_below must be numeric.")
        elif not (0.0 < float(reject_confidence_below) <= 1.0):
            errors.append("prediction_policy.reject_confidence_below must be in (0,1].")
    reject_label = prediction_policy.get("reject_label", None)
    if reject_label is not None and not isinstance(reject_label, int):
        errors.append("prediction_policy.reject_label must be an integer.")


__all__ = [
    "CALIBRATION_CARD_SCHEMA_VERSION",
    "build_calibration_card_payload",
    "validate_calibration_card_payload",
]
