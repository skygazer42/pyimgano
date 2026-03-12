from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyimgano.services.infer_options_service import (
    apply_defects_defaults,
    resolve_defects_preset_payload,
    resolve_preprocessing_preset_knobs,
)


def test_resolve_defects_preset_payload_returns_json_friendly_payload() -> None:
    payload = resolve_defects_preset_payload("industrial-defects-fp40")

    assert payload["roi_xyxy_norm"] == [0.1, 0.1, 0.9, 0.9]
    assert payload["min_area"] == 16
    assert payload["max_regions"] == 20


def test_resolve_defects_preset_payload_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown defects preset"):
        resolve_defects_preset_payload("missing-preset")


def test_resolve_preprocessing_preset_knobs_parses_supported_payload() -> None:
    knobs = resolve_preprocessing_preset_knobs("illumination-contrast-balanced")

    assert knobs.white_balance == "gray_world"
    assert knobs.homomorphic is True
    assert knobs.clahe is True
    assert knobs.clahe_tile_grid_size == (8, 8)


def test_resolve_preprocessing_preset_knobs_rejects_unsupported_config_key(monkeypatch) -> None:
    import pyimgano.services.infer_options_service as infer_options_service

    monkeypatch.setattr(
        infer_options_service,
        "resolve_preprocessing_preset",
        lambda _name: SimpleNamespace(
            name="unsupported",
            config_key="preprocessing.not_supported",
            payload={"enabled": True},
        ),
    )

    with pytest.raises(ValueError, match="Only preprocessing.illumination_contrast"):
        resolve_preprocessing_preset_knobs("unsupported")


def test_apply_defects_defaults_populates_unset_cli_knobs() -> None:
    import pyimgano.infer_cli as infer_cli

    parser = infer_cli._build_parser()
    args = parser.parse_args(
        [
            "--model",
            "ssim_template_map",
            "--input",
            "dummy.png",
        ]
    )

    apply_defects_defaults(
        args,
        {
            "roi_xyxy_norm": [0.2, 0.2, 0.8, 0.8],
            "min_area": 7,
            "map_smoothing": {"method": "median", "ksize": 3, "sigma": 0.0},
            "hysteresis": {"enabled": True, "low": 0.1, "high": 0.2},
        },
    )

    assert args.roi_xyxy_norm == [0.2, 0.2, 0.8, 0.8]
    assert int(args.defect_min_area) == 7
    assert str(args.defect_map_smoothing) == "median"
    assert int(args.defect_map_smoothing_ksize) == 3
    assert bool(args.defect_hysteresis) is True
    assert float(args.defect_hysteresis_low) == 0.1
    assert float(args.defect_hysteresis_high) == 0.2


def test_apply_defects_defaults_preserves_explicit_cli_values() -> None:
    import pyimgano.infer_cli as infer_cli

    parser = infer_cli._build_parser()
    args = parser.parse_args(
        [
            "--model",
            "ssim_template_map",
            "--input",
            "dummy.png",
            "--defect-min-area",
            "1",
            "--defect-map-smoothing",
            "gaussian",
        ]
    )

    apply_defects_defaults(
        args,
        {
            "min_area": 7,
            "map_smoothing": {"method": "median", "ksize": 3, "sigma": 0.0},
        },
    )

    assert int(args.defect_min_area) == 1
    assert str(args.defect_map_smoothing) == "gaussian"
