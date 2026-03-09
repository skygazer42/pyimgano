from __future__ import annotations

import json


def test_preprocessing_preset_is_json_friendly() -> None:
    from pyimgano.cli_presets import resolve_preprocessing_preset

    preset = resolve_preprocessing_preset("illumination-contrast-balanced")
    assert preset is not None

    payload = {
        "name": preset.name,
        "deployable": preset.deployable,
        "config_key": preset.config_key,
        "payload": dict(preset.payload or {}),
    }
    text = json.dumps(payload, sort_keys=True)
    assert "illumination-contrast-balanced" in text
    assert "white_balance" in text
