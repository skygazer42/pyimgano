import json


def test_pyim_list_model_presets_supports_family_filter(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "model-presets", "--family", "graph"])
    assert code == 0
    out = capsys.readouterr().out
    assert "industrial-structural-rgraph" in out
    assert "industrial-structural-lof" not in out


def test_pyim_list_model_presets_outputs_json_metadata(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "model-presets", "--family", "neighbors", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(
        item["name"] == "industrial-structural-lof" and "neighbors" in item["tags"]
        for item in payload
    )
