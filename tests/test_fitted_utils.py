from __future__ import annotations


def test_require_fitted_raises_for_missing_attrs() -> None:
    from pyimgano.utils.fitted import require_fitted

    class Dummy:
        pass

    try:
        require_fitted(Dummy(), ["a", "b"])
    except RuntimeError as exc:
        assert "Missing" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing attributes")


def test_require_fitted_ok_when_present() -> None:
    from pyimgano.utils.fitted import require_fitted

    class Dummy:
        def __init__(self) -> None:
            self.a = 1
            self.b = 2

    require_fitted(Dummy(), ["a", "b"])

