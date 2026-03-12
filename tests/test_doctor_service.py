from __future__ import annotations

from pyimgano.services.doctor_service import collect_doctor_payload


def test_collect_doctor_payload_returns_json_ready_shape() -> None:
    payload = collect_doctor_payload()

    assert payload["tool"] == "pyimgano-doctor"
    assert "optional_modules" in payload
    assert "baselines" in payload
