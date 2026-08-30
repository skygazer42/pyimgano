from __future__ import annotations

import pickle

import pytest

from pyimgano import __version__
from pyimgano.utils.model_utils import load_model, save_model


def test_save_model_stamps_current_package_version(tmp_path) -> None:
    path = tmp_path / "detector.pkl"

    save_model({"detector": "stub"}, str(path))
    payload = pickle.loads(path.read_bytes())

    assert payload["pyimgano_version"] == __version__


def test_load_model_requires_explicit_trust(tmp_path) -> None:
    path = tmp_path / "detector.pkl"
    save_model({"detector": "stub"}, str(path))

    with pytest.raises(ValueError, match="trusted=True"):
        load_model(str(path))

    assert load_model(str(path), trusted=True) == {"detector": "stub"}
