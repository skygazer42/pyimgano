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

    last = (proc.stdout or "").strip().splitlines()[-1]
    return json.loads(last)


def test_import_pyimgano_pyim_cli_does_not_import_heavy_roots_by_default() -> None:
    payload = _run_py(
        r"""
import json
import sys

import pyimgano.pyim_cli as pyim_cli

HEAVY_ROOTS = [
    "torch",
    "torchvision",
    "cv2",
    "open_clip",
    "onnxruntime",
    "openvino",
    "faiss",
    "anomalib",
]


def _is_loaded(root: str) -> bool:
    if root in sys.modules:
        return True
    prefix = root + "."
    return any(name.startswith(prefix) for name in sys.modules)


present = [root for root in HEAVY_ROOTS if _is_loaded(root)]
print(json.dumps({"present": present, "has_main": hasattr(pyim_cli, "main")}))
""",
    )

    assert payload["has_main"] is True
    assert payload["present"] == [], f"Unexpected heavy imports: {payload['present']}"
