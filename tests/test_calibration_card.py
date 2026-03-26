import pytest


def test_build_calibration_card_payload_from_single_category_report() -> None:
    from pyimgano.reporting.calibration_card import build_calibration_card_payload

    payload = build_calibration_card_payload(
        {
            "run_dir": "/tmp/run_a",
            "dataset": "mvtec",
            "category": "bottle",
            "model": "vision_patchcore",
            "threshold": 0.73,
            "threshold_provenance": {
                "method": "quantile",
                "quantile": 0.95,
                "source": "contamination",
                "contamination": 0.05,
                "calibration_count": 20,
                "score_summary": {"count": 20, "min": 0.1, "max": 0.9},
            },
            "prediction": {
                "reject_confidence_below": 0.75,
                "reject_label": -9,
            },
            "split_fingerprint": {"sha256": "f" * 64},
        }
    )

    assert payload["schema_version"] == 1
    assert payload["run_dir"] == "/tmp/run_a"
    assert payload["split_fingerprint"]["sha256"] == "f" * 64
    assert payload["threshold_context"]["scope"] == "image"
    assert payload["threshold_context"]["category_count"] == 1
    assert payload["prediction_policy"]["reject_confidence_below"] == pytest.approx(0.75)
    assert payload["prediction_policy"]["reject_label"] == -9
    assert payload["image_threshold"]["threshold"] == pytest.approx(0.73)
    assert payload["image_threshold"]["provenance"]["quantile"] == pytest.approx(0.95)
    assert payload["image_threshold"]["provenance"]["score_summary"]["count"] == 20
    assert payload["image_threshold"]["score_distribution"]["count"] == 20
    assert payload["image_threshold"]["score_distribution"]["min"] == pytest.approx(0.1)
    assert payload["image_threshold"]["score_distribution"]["max"] == pytest.approx(0.9)


def test_validate_calibration_card_payload_accepts_per_category_payload() -> None:
    from pyimgano.reporting.calibration_card import validate_calibration_card_payload

    errors = validate_calibration_card_payload(
        {
            "schema_version": 1,
            "dataset": "mvtec",
            "threshold_context": {"scope": "per_category", "category_count": 1},
            "prediction_policy": {
                "reject_confidence_below": 0.75,
                "reject_label": -9,
            },
            "split_fingerprint": {"sha256": "a" * 64},
            "per_category": {
                "bottle": {
                    "threshold": 0.71,
                    "provenance": {
                        "method": "quantile",
                        "source": "contamination",
                        "score_summary": {"count": 12, "min": 0.1, "max": 0.8},
                    },
                    "score_distribution": {"count": 12, "min": 0.1, "max": 0.8},
                }
            },
        }
    )

    assert errors == []


def test_validate_calibration_card_payload_rejects_missing_threshold() -> None:
    from pyimgano.reporting.calibration_card import validate_calibration_card_payload

    errors = validate_calibration_card_payload({"schema_version": 1, "dataset": "mvtec"})

    assert any("image_threshold" in item for item in errors)


def test_validate_calibration_card_payload_rejects_bad_threshold_payload_shape() -> None:
    from pyimgano.reporting.calibration_card import validate_calibration_card_payload

    errors = validate_calibration_card_payload(
        {
            "schema_version": 1,
            "threshold_context": "image",
            "prediction_policy": {
                "reject_confidence_below": 1.5,
                "reject_label": "bad",
            },
            "image_threshold": {
                "threshold": "bad",
                "provenance": [],
                "score_distribution": {"count": "oops"},
            },
            "split_fingerprint": {"sha256": ""},
        }
    )

    assert any("threshold_context" in item for item in errors)
    assert any("prediction_policy.reject_confidence_below" in item for item in errors)
    assert any("prediction_policy.reject_label" in item for item in errors)
    assert any("image_threshold.threshold" in item for item in errors)
    assert any("image_threshold.provenance" in item for item in errors)
    assert any("image_threshold.score_distribution.count" in item for item in errors)
    assert any("split_fingerprint.sha256" in item for item in errors)


def test_validate_calibration_card_payload_rejects_boolean_score_summary_values() -> None:
    from pyimgano.reporting.calibration_card import validate_calibration_card_payload

    errors = validate_calibration_card_payload(
        {
            "schema_version": 1,
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.42,
                "provenance": {
                    "method": "quantile",
                    "score_summary": {
                        "count": True,
                        "quantiles": {"p95": False},
                    },
                },
            },
            "split_fingerprint": {"sha256": "f" * 64},
        }
    )

    assert any("image_threshold.provenance.score_summary.count" in item for item in errors)
    assert any("image_threshold.provenance.score_summary.quantiles['p95']" in item for item in errors)
