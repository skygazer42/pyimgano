from __future__ import annotations

from pyimgano.utils.security import ErrorCode, SecurityValidator, validate_image_file


def test_validate_path_rejects_sibling_prefix_path(tmp_path) -> None:
    base_dir = tmp_path / "base"
    sibling_dir = tmp_path / "base_evil"
    base_dir.mkdir()
    sibling_dir.mkdir()
    image_path = sibling_dir / "sample.png"
    image_path.write_bytes(b"png")

    valid, message = SecurityValidator.validate_path(str(image_path), str(base_dir))

    assert valid is False
    assert "Path traversal" in message


def test_validate_image_file_rejects_path_outside_base_dir(tmp_path) -> None:
    base_dir = tmp_path / "base"
    sibling_dir = tmp_path / "base_evil"
    base_dir.mkdir()
    sibling_dir.mkdir()
    image_path = sibling_dir / "sample.png"
    image_path.write_bytes(b"png")

    code, message = validate_image_file(str(image_path), base_dir=str(base_dir))

    assert code is ErrorCode.SECURITY_VIOLATION
    assert "Path traversal" in message
