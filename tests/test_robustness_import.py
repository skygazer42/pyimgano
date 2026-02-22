from __future__ import annotations


def test_robustness_package_importable() -> None:
    import pyimgano.robustness as robustness

    assert hasattr(robustness, "Corruption")

