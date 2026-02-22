import json


def test_cli_list_models_outputs_text(capsys):
    from pyimgano.cli import main

    code = main(["--list-models"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_patchcore" in out


def test_cli_list_models_outputs_json(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert "vision_patchcore" in parsed


def test_cli_model_info_outputs_text(capsys):
    from pyimgano.cli import main

    code = main(["--model-info", "vision_patchcore"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_patchcore" in out
    assert "Signature" in out


def test_cli_model_info_outputs_json(capsys):
    from pyimgano.cli import main

    code = main(["--model-info", "vision_patchcore", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["name"] == "vision_patchcore"
    assert "signature" in parsed


def test_cli_discovery_flags_are_mutually_exclusive(capsys):
    from pyimgano.cli import main

    code = main(["--list-models", "--model-info", "vision_patchcore"])
    assert code != 0
    err = capsys.readouterr().err
    assert "mutually" in err.lower() or "exclusive" in err.lower()
