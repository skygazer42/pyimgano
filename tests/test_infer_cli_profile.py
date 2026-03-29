from __future__ import annotations

import json
from pathlib import Path


def test_format_infer_profile_summary_renders_expected_order() -> None:
    from pyimgano.infer_cli_profile import format_infer_profile_summary

    line = format_infer_profile_summary(
        load_model=1.25,
        fit_calibrate=2.5,
        infer=3.75,
        artifacts=0.5,
        total=8.0,
    )

    assert (
        line
        == "profile: load_model=1.250s fit_calibrate=2.500s infer=3.750s artifacts=0.500s total=8.000s"
    )


def test_build_infer_profile_payload_includes_counts_and_timings() -> None:
    from pyimgano.infer_cli_profile import build_infer_profile_payload

    payload = build_infer_profile_payload(
        inputs=5,
        processed=4,
        errors=1,
        load_model=1.0,
        fit_calibrate=2.0,
        infer=3.0,
        artifacts=4.0,
        total=10.0,
    )

    assert payload == {
        "tool": "pyimgano-infer",
        "counts": {"inputs": 5, "processed": 4, "errors": 1},
        "timing_seconds": {
            "load_model": 1.0,
            "fit_calibrate": 2.0,
            "infer": 3.0,
            "artifacts": 4.0,
            "total": 10.0,
        },
    }


def test_write_infer_profile_payload_creates_parent_dir_and_json(tmp_path: Path) -> None:
    from pyimgano.infer_cli_profile import write_infer_profile_payload

    target = tmp_path / "nested" / "profile.json"
    payload = {"tool": "pyimgano-infer", "counts": {"inputs": 1}, "timing_seconds": {"total": 1.0}}

    write_infer_profile_payload(target, payload)

    assert json.loads(target.read_text(encoding="utf-8")) == payload
