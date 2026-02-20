from pyimgano.utils.optional_deps import optional_import, require


def test_optional_import_missing():
    module, error = optional_import("this_package_does_not_exist_123")
    assert module is None
    assert error is not None


def test_require_raises_importerror():
    try:
        require("this_package_does_not_exist_123", extra="backends", purpose="unit test")
    except ImportError as exc:
        message = str(exc)
        assert "Optional dependency" in message
        assert "pip install" in message
    else:
        raise AssertionError("Expected ImportError to be raised")

