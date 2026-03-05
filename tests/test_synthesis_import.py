from __future__ import annotations


def test_synthesis_import_smoke() -> None:
    import pyimgano.synthesis as syn

    assert hasattr(syn, "AnomalySynthesizer")
    assert hasattr(syn, "perlin_noise_2d")
    assert hasattr(syn, "make_preset")
