from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    ("module_name", "heavy_roots", "must_not_load"),
    [
        (
            "pyimgano.models",
            ["torch", "torchvision", "cv2", "open_clip", "faiss", "anomalib", "sklearn"],
            ["pyimgano.models.baseml", "pyimgano.models.baseCv"],
        ),
        (
            "pyimgano.features",
            ["torch", "torchvision", "cv2", "open_clip", "anomalib", "sklearn"],
            [
                "pyimgano.features.pca_projector",
                "pyimgano.features.scaler",
                "pyimgano.features.torchvision_backbone",
            ],
        ),
    ],
)
def test_import_contracts_keep_core_packages_lazy(
    module_name: str, heavy_roots: list[str], must_not_load: list[str]
) -> None:
    payload = _run_py(
        f"""
import importlib
import json
import sys

importlib.import_module("{module_name}")

heavy_roots = {heavy_roots!r}
must_not_load = {must_not_load!r}


def _is_loaded(root: str) -> bool:
    if root in sys.modules:
        return True
    prefix = root + "."
    return any(name.startswith(prefix) for name in sys.modules)


present = [root for root in heavy_roots if _is_loaded(root)]
loaded = [name for name in must_not_load if name in sys.modules]
print(json.dumps({{"present": present, "loaded": loaded}}))
"""
    )

    assert payload["present"] == [], f"Unexpected heavy imports: {payload['present']}"
    assert payload["loaded"] == [], f"Unexpected eager module imports: {payload['loaded']}"


@pytest.mark.parametrize(
    ("module_name", "heavy_roots", "payload_checks"),
    [
        (
            "pyimgano",
            ["torch", "torchvision", "cv2", "open_clip", "onnxruntime", "openvino"],
            "payload['has_version'] is True",
        ),
        (
            "pyimgano.detectors",
            [
                "torch",
                "torchvision",
                "cv2",
                "onnxruntime",
                "openvino",
                "open_clip",
                "diffusers",
                "faiss",
                "anomalib",
                "mamba_ssm",
            ],
            "payload['has_isolation_forest'] is True and payload['has_autoencoder'] is True",
        ),
        (
            "pyimgano.preprocessing",
            ["torch", "torchvision", "onnxruntime", "openvino", "skimage"],
            "payload['n_exports'] > 0",
        ),
        (
            "pyimgano.pyim_cli",
            [
                "torch",
                "torchvision",
                "cv2",
                "open_clip",
                "onnxruntime",
                "openvino",
                "faiss",
                "anomalib",
            ],
            "payload['has_main'] is True",
        ),
    ],
)
def test_other_package_surfaces_stay_lightweight(
    module_name: str, heavy_roots: list[str], payload_checks: str
) -> None:
    payload = _run_py(
        f"""
import importlib
import json
import sys

module = importlib.import_module("{module_name}")

heavy_roots = {heavy_roots!r}


def _is_loaded(root: str) -> bool:
    if root in sys.modules:
        return True
    prefix = root + "."
    return any(name.startswith(prefix) for name in sys.modules)


payload = {{
    "present": [root for root in heavy_roots if _is_loaded(root)],
    "has_version": hasattr(module, "__version__"),
    "has_isolation_forest": hasattr(module, "IsolationForestDetector"),
    "has_autoencoder": hasattr(module, "AutoencoderDetector"),
    "n_exports": len(getattr(module, "__all__", [])),
    "has_main": hasattr(module, "main"),
}}
print(json.dumps(payload))
"""
    )

    assert payload["present"] == [], f"Unexpected heavy imports: {payload['present']}"
    assert eval(payload_checks, {"__builtins__": {}}, {"payload": payload})


def test_models_discovery_keeps_optional_openclip_variants_registered() -> None:
    payload = _run_py(
        """
import json

import pyimgano.models as models

names = set(models.list_models())
print(
    json.dumps(
        {
            "patchknn": "vision_openclip_patchknn" in names,
            "promptscore": "vision_openclip_promptscore" in names,
            "winclip": "winclip" in names,
            "vision_winclip": "vision_winclip" in names,
        }
    )
)
"""
    )

    assert payload == {
        "patchknn": True,
        "promptscore": True,
        "winclip": True,
        "vision_winclip": True,
    }
