from __future__ import annotations

import json


def test_feature_cli_list_extractors_includes_hog_and_lbp(capsys) -> None:
    from pyimgano.feature_cli import main as feature_main

    code = feature_main(["--list-extractors"])
    assert code == 0
    out = capsys.readouterr().out
    assert "hog" in out
    assert "lbp" in out


def test_feature_cli_extractor_info_emits_json(capsys) -> None:
    from pyimgano.feature_cli import main as feature_main

    code = feature_main(["--extractor-info", "hog", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "hog"
    assert "signature" in payload
    assert "accepted_kwargs" in payload
    assert isinstance(payload["accepted_kwargs"], list)
    assert "resize_hw" in payload["accepted_kwargs"]
