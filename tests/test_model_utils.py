from __future__ import annotations

import pickle

from pyimgano import __version__
from pyimgano.utils.model_utils import save_model


def test_save_model_stamps_current_package_version(tmp_path) -> None:
    path = tmp_path / "detector.pkl"

    save_model({"detector": "stub"}, str(path))
    payload = pickle.loads(path.read_bytes())

    assert payload["pyimgano_version"] == __version__
