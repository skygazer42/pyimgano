from __future__ import annotations

from pathlib import Path

from pyimgano.utils.path_normalize import normalize_path


def test_normalize_path_converts_backslashes() -> None:
    assert normalize_path(r"a\\b\\c.png") == "a/b/c.png"
    assert normalize_path(r"a\b\c.png") == "a/b/c.png"


def test_normalize_path_handles_windows_drive_letters() -> None:
    assert normalize_path(r"C:\data\images\0.png") == "C:/data/images/0.png"


def test_normalize_path_preserves_url_like_strings() -> None:
    assert normalize_path("s3://bucket/key") == "s3://bucket/key"


def test_normalize_path_accepts_path_objects() -> None:
    assert normalize_path(Path("a") / "b" / "c.png") == "a/b/c.png"


def test_normalize_path_collapses_repeated_separators() -> None:
    assert normalize_path("a//b///c.png") == "a/b/c.png"
    assert normalize_path(r"a\\\b\\c.png") == "a/b/c.png"


def test_normalize_path_preserves_unc_prefix() -> None:
    assert normalize_path(r"\\server\share\dir\0.png") == "//server/share/dir/0.png"
