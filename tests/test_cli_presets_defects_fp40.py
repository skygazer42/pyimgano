from __future__ import annotations

import json


def test_defects_fp40_preset_is_json_friendly() -> None:
    from pyimgano.cli_presets import resolve_defects_preset

    preset = resolve_defects_preset("industrial-defects-fp40")
    assert preset is not None

    payload = {"name": preset.name, "payload": dict(preset.payload)}
    text = json.dumps(payload, sort_keys=True)
    assert "industrial-defects-fp40" in text
    assert "roi_xyxy_norm" in text


def test_infer_cli_applies_defects_fp40_preset_defaults() -> None:
    from pyimgano.infer_cli import _apply_defects_preset_if_requested, _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--model",
            "ssim_template_map",
            "--input",
            "dummy.png",
            "--defects-preset",
            "industrial-defects-fp40",
        ]
    )
    _apply_defects_preset_if_requested(args)

    assert bool(args.defects) is True
    assert args.roi_xyxy_norm == [0.1, 0.1, 0.9, 0.9]
    assert int(args.defect_border_ignore_px) == 2
    assert str(args.defect_map_smoothing) == "median"
    assert int(args.defect_map_smoothing_ksize) == 3
    assert bool(args.defect_hysteresis) is True
    assert bool(args.defect_merge_nearby) is True
    assert float(args.defect_min_fill_ratio) == 0.15
    assert float(args.defect_max_aspect_ratio) == 6.0
    assert float(args.defect_min_solidity) == 0.8
    assert int(args.defect_min_area) == 16
    assert float(args.defect_min_score_max) == 0.6
    assert int(args.defect_max_regions) == 20


def test_defects_preset_does_not_override_explicit_cli_knobs() -> None:
    from pyimgano.infer_cli import _apply_defects_preset_if_requested, _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--model",
            "ssim_template_map",
            "--input",
            "dummy.png",
            "--defects-preset",
            "industrial-defects-fp40",
            "--defect-min-area",
            "1",
        ]
    )
    _apply_defects_preset_if_requested(args)
    assert int(args.defect_min_area) == 1


def test_infer_cli_defects_preset_wrapper_delegates_to_infer_options_service(monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.infer_options_service as infer_options_service

    parser = infer_cli._build_parser()
    args = parser.parse_args(
        [
            "--model",
            "ssim_template_map",
            "--input",
            "dummy.png",
            "--defects-preset",
            "industrial-defects-fp40",
        ]
    )

    calls: list[str] = []
    monkeypatch.setattr(
        infer_options_service,
        "resolve_defects_preset_payload",
        lambda name: calls.append(str(name))
        or {
            "min_area": 7,
            "roi_xyxy_norm": [0.2, 0.2, 0.8, 0.8],
        },
    )

    infer_cli._apply_defects_preset_if_requested(args)

    assert calls == ["industrial-defects-fp40"]
    assert bool(args.defects) is True
    assert int(args.defect_min_area) == 7
    assert args.roi_xyxy_norm == [0.2, 0.2, 0.8, 0.8]


def test_infer_cli_defects_payload_wrapper_delegates_to_infer_options_service(monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.infer_options_service as infer_options_service

    parser = infer_cli._build_parser()
    args = parser.parse_args(
        [
            "--model",
            "ssim_template_map",
            "--input",
            "dummy.png",
        ]
    )

    calls: list[dict[str, object]] = []

    def _apply(ns, payload):  # noqa: ANN001
        calls.append(dict(payload))
        ns.defect_min_area = 9

    monkeypatch.setattr(infer_options_service, "apply_defects_defaults", _apply)

    infer_cli._apply_defects_defaults_from_payload(args, {"min_area": 9})

    assert calls == [{"min_area": 9}]
    assert int(args.defect_min_area) == 9
