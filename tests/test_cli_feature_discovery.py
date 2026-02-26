import json


def test_cli_list_feature_extractors_outputs_text(capsys) -> None:
    from pyimgano.cli import main

    code = main(["--list-feature-extractors"])
    assert code == 0
    out = capsys.readouterr().out
    assert "identity" in out


def test_cli_list_feature_extractors_outputs_json(capsys) -> None:
    from pyimgano.cli import main

    code = main(["--list-feature-extractors", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert "identity" in parsed


def test_cli_feature_info_outputs_text(capsys) -> None:
    from pyimgano.cli import main

    code = main(["--feature-info", "identity"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Name:" in out
    assert "identity" in out


def test_cli_feature_discovery_flags_are_mutually_exclusive(capsys) -> None:
    from pyimgano.cli import main

    code = main(["--list-models", "--list-feature-extractors"])
    assert code != 0
    err = capsys.readouterr().err
    assert "mutually" in err.lower() or "exclusive" in err.lower()

