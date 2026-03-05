from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_py(code: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=True,
    )

    # Be robust to accidental prints at import time: treat the last stdout line as JSON.
    last = (proc.stdout or "").strip().splitlines()[-1]
    return json.loads(last)


def test_import_pyimgano_detectors_does_not_import_heavy_roots_by_default() -> None:
    payload = _run_py(
        r"""
import json
import sys

import pyimgano.detectors as detectors

HEAVY_ROOTS = [
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
]


def _is_loaded(root: str) -> bool:
    if root in sys.modules:
        return True
    prefix = root + "."
    return any(name.startswith(prefix) for name in sys.modules)


present = [root for root in HEAVY_ROOTS if _is_loaded(root)]
print(
    json.dumps(
        {
            "present": present,
            "has_isolation_forest": bool(hasattr(detectors, "IsolationForestDetector")),
            "has_autoencoder": bool(hasattr(detectors, "AutoencoderDetector")),
        }
    )
)
""",
    )

    assert payload["has_isolation_forest"] is True
    # Accessing the attribute should not import heavy deps unless instantiated.
    assert payload["has_autoencoder"] is True
    assert payload["present"] == [], f"Unexpected heavy imports: {payload['present']}"
