from __future__ import annotations

import json


def test_cli_list_categories_manifest_outputs_text(capsys, tmp_path) -> None:
    from pyimgano.cli import main

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"image_path":"a.png","category":"bottle"}',
                '{"image_path":"b.png","category":"cable"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "--dataset",
            "manifest",
            "--manifest-path",
            str(manifest),
            "--list-categories",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["bottle", "cable"]


def test_cli_list_categories_manifest_outputs_json(capsys, tmp_path) -> None:
    from pyimgano.cli import main

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"image_path":"a.png","category":"bottle"}',
                '{"image_path":"b.png","category":"cable"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "--dataset",
            "manifest",
            "--manifest-path",
            str(manifest),
            "--list-categories",
            "--json",
        ]
    )
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == ["bottle", "cable"]
