from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_feature_info_materializes_only_requested_extractor_module() -> None:
    payload = _run_py(
        """
import json
import sys

import pyimgano.features as features

before = {
    "pca_projector": "pyimgano.features.pca_projector" in sys.modules,
    "torchvision_backbone": "pyimgano.features.torchvision_backbone" in sys.modules,
}
info = features.feature_info("pca_projector")
after = {
    "pca_projector": "pyimgano.features.pca_projector" in sys.modules,
    "torchvision_backbone": "pyimgano.features.torchvision_backbone" in sys.modules,
}
print(json.dumps({"before": before, "after": after, "accepted_kwargs": info["accepted_kwargs"]}))
"""
    )

    assert payload["before"] == {
        "pca_projector": False,
        "torchvision_backbone": False,
    }
    assert "n_components" in payload["accepted_kwargs"]
    assert payload["after"] == {
        "pca_projector": True,
        "torchvision_backbone": False,
    }


def test_models_package_materializes_base_vision_detector_on_attribute_access() -> None:
    payload = _run_py(
        """
import json
import sys

import pyimgano.models as models

before = {
    "baseml": "pyimgano.models.baseml" in sys.modules,
    "baseCv": "pyimgano.models.baseCv" in sys.modules,
}
base = models.BaseVisionDetector
after = {
    "baseml": "pyimgano.models.baseml" in sys.modules,
    "baseCv": "pyimgano.models.baseCv" in sys.modules,
}
print(json.dumps({"before": before, "after": after, "name": base.__name__}))
"""
    )

    assert payload["before"] == {"baseml": False, "baseCv": False}
    assert payload["name"] == "BaseVisionDetector"
    assert payload["after"] == {"baseml": True, "baseCv": False}


def test_models_package_materializes_base_vision_deep_detector_on_attribute_access() -> None:
    payload = _run_py(
        """
import json
import sys

import pyimgano.models as models

before = {"baseCv": "pyimgano.models.baseCv" in sys.modules}
base = models.BaseVisionDeepDetector
after = {"baseCv": "pyimgano.models.baseCv" in sys.modules}
print(json.dumps({"before": before, "after": after, "name": base.__name__}))
"""
    )

    assert payload["before"] == {"baseCv": False}
    assert payload["name"] == "BaseVisionDeepDetector"
    assert payload["after"] == {"baseCv": True}


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_py(code: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    last = (proc.stdout or "").strip().splitlines()[-1]
    return json.loads(last)
