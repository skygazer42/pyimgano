from __future__ import annotations

from pathlib import Path

import pytest


def test_parse_json_mapping_arg_returns_dict() -> None:
    from pyimgano.infer_cli_inputs import parse_json_mapping_arg

    payload = parse_json_mapping_arg('{"alpha": 1, "beta": true}', arg_name="--options")

    assert payload == {"alpha": 1, "beta": True}


def test_parse_json_mapping_arg_rejects_non_object_json() -> None:
    from pyimgano.infer_cli_inputs import parse_json_mapping_arg

    with pytest.raises(ValueError, match="--options must be a JSON object"):
        parse_json_mapping_arg("[1, 2, 3]", arg_name="--options")


def test_parse_csv_ints_arg_parses_and_skips_empty_tokens() -> None:
    from pyimgano.infer_cli_inputs import parse_csv_ints_arg

    values = parse_csv_ints_arg("1, 2, , 4", arg_name="--threads")

    assert values == [1, 2, 4]


def test_parse_csv_ints_arg_raises_with_argument_name() -> None:
    from pyimgano.infer_cli_inputs import parse_csv_ints_arg

    with pytest.raises(ValueError, match="--threads must be a comma-separated list of ints"):
        parse_csv_ints_arg("1,nope,3", arg_name="--threads")


def test_parse_csv_strs_arg_returns_non_empty_values() -> None:
    from pyimgano.infer_cli_inputs import parse_csv_strs_arg

    values = parse_csv_strs_arg(" basic, ,fast ,accurate ", arg_name="--presets")

    assert values == ["basic", "fast", "accurate"]


def test_collect_image_paths_collects_supported_images_from_directory(tmp_path: Path) -> None:
    from pyimgano.infer_cli_inputs import collect_image_paths

    (tmp_path / "root.png").write_bytes(b"png")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.jpg").write_bytes(b"jpg")
    (nested / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = collect_image_paths(tmp_path)

    assert paths == [
        str(nested / "a.jpg"),
        str(tmp_path / "root.png"),
    ]


def test_collect_image_paths_rejects_missing_path(tmp_path: Path) -> None:
    from pyimgano.infer_cli_inputs import collect_image_paths

    with pytest.raises(FileNotFoundError, match="Input not found"):
        collect_image_paths(tmp_path / "missing")


def test_collect_image_paths_rejects_unsupported_file_suffix(tmp_path: Path) -> None:
    from pyimgano.infer_cli_inputs import collect_image_paths

    bad = tmp_path / "bad.txt"
    bad.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported image type"):
        collect_image_paths(bad)
