from __future__ import annotations

import json


def test_pyim_list_preprocessing_deployable_only_includes_new_schemes(capsys) -> None:
    from pyimgano.pyim_cli import main

    rc = main(["--list", "preprocessing", "--json", "--deployable-only"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    names = {item["name"] for item in payload}

    assert "illumination-contrast-aggressive" in names
    assert "illumination-contrast-no-homomorphic" in names
    assert "illumination-contrast-color-stable" in names


def test_cli_presets_resolve_preprocessing_preset_supports_new_schemes() -> None:
    from pyimgano.cli_presets import resolve_preprocessing_preset

    for name in [
        "illumination-contrast-aggressive",
        "illumination-contrast-no-homomorphic",
        "illumination-contrast-color-stable",
    ]:
        preset = resolve_preprocessing_preset(name)
        assert preset is not None
        assert preset.deployable is True
        assert preset.config_key == "preprocessing.illumination_contrast"
        assert isinstance(preset.payload, dict)
        assert "white_balance" in preset.payload
        assert "clahe" in preset.payload
