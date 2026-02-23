from __future__ import annotations

import pytest


def test_pickle_roundtrip_classical_detector(tmp_path) -> None:
    import pyimgano.models  # noqa: F401 - registry population
    from pyimgano.models.registry import create_model
    from pyimgano.serialization.pickle import load_detector, save_detector

    detector = create_model("vision_ecod", contamination=0.1)
    path = tmp_path / "detector.pkl"

    save_detector(path, detector)
    loaded = load_detector(path)
    assert loaded.__class__ is detector.__class__
    assert hasattr(loaded, "decision_function")


def test_pickle_rejects_non_classical_detector(tmp_path) -> None:
    from pyimgano.serialization.pickle import save_detector

    with pytest.raises(TypeError):
        save_detector(tmp_path / "x.pkl", object())

