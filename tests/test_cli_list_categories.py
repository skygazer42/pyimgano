from __future__ import annotations

import json


def test_cli_list_categories_outputs_text(capsys, tmp_path) -> None:
    from pyimgano.cli import main

    (tmp_path / "bottle").mkdir()
    (tmp_path / "not_a_category").mkdir()

    code = main(
        [
            "--dataset",
            "mvtec",
            "--root",
            str(tmp_path),
            "--list-categories",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["bottle"]


def test_cli_list_categories_outputs_json(capsys, tmp_path) -> None:
    from pyimgano.cli import main

    (tmp_path / "bottle").mkdir()

    code = main(
        [
            "--dataset",
            "mvtec",
            "--root",
            str(tmp_path),
            "--list-categories",
            "--json",
        ]
    )
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == ["bottle"]

