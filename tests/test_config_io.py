import json

import pytest

from pyimgano.config.io import load_config


def test_load_config_json(tmp_path):
    config_path = tmp_path / "cfg.json"
    payload = {"a": 1, "b": {"c": [1, 2, 3]}, "d": True, "e": None}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_config(config_path) == payload


def test_load_config_unknown_extension_raises(tmp_path):
    config_path = tmp_path / "cfg.txt"
    config_path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_config(config_path)

    assert ".txt" in str(exc.value)


def test_load_config_yaml_optional(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("a: 1\nb: test\n", encoding="utf-8")

    try:
        import yaml  # noqa: F401
    except Exception:
        with pytest.raises(ImportError) as exc:
            load_config(config_path)
        msg = str(exc.value)
        assert "PyYAML" in msg
        assert "pip install" in msg
    else:
        assert load_config(config_path) == {"a": 1, "b": "test"}

