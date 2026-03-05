from __future__ import annotations


def test_optional_import_returns_module_when_present() -> None:
    from pyimgano.utils.optional_deps import optional_import

    mod, err = optional_import("math")
    assert err is None
    assert mod is not None
    assert hasattr(mod, "sqrt")


def test_optional_import_returns_error_when_missing() -> None:
    from pyimgano.utils.optional_deps import optional_import

    mod, err = optional_import("pyimgano__definitely_missing_module__xyz")
    assert mod is None
    assert err is not None


def test_require_returns_module_when_present() -> None:
    from pyimgano.utils.optional_deps import require

    mod = require("math")
    assert hasattr(mod, "sqrt")


def test_require_raises_import_error_with_hint_for_missing_module() -> None:
    from pyimgano.utils.optional_deps import require

    try:
        require("pyimgano__definitely_missing_module__xyz")
    except ImportError as exc:
        msg = str(exc)
        assert "Optional dependency" in msg
        assert "pip install" in msg
    else:  # pragma: no cover
        raise AssertionError("Expected ImportError for missing module")


def test_require_with_extra_mentions_pyimgano_extras_hint() -> None:
    from pyimgano.utils.optional_deps import require

    try:
        require("pyimgano__definitely_missing_module__xyz", extra="clip", purpose="unit-test")
    except ImportError as exc:
        msg = str(exc)
        assert "pyimgano[clip]" in msg
        assert "unit-test" in msg
    else:  # pragma: no cover
        raise AssertionError("Expected ImportError for missing module")
