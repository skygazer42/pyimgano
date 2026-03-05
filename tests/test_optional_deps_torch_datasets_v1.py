from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


@pytest.mark.parametrize(
    "module",
    [
        "pyimgano.datasets.image",
        "pyimgano.datasets.array",
        "pyimgano.datasets.transforms",
        "pyimgano.datasets.corruptions",
        "pyimgano.datasets.datamodule",
    ],
)
def test_torch_backed_dataset_modules_raise_actionable_import_error(module: str) -> None:
    """Torch-backed dataset modules should fail with an extras install hint.

    We simulate a missing torch/torchvision environment by patching the optional
    deps import path. If a module uses direct `import torch`, this test will NOT
    fail (torch may be installed in the test env), revealing an import-boundary
    regression.
    """

    payload = _run_py(
        rf"""
import importlib
import json

import pyimgano.utils.optional_deps as optional_deps

orig = optional_deps.import_module


def _fake_import_module(name, package=None):
    if str(name).startswith(("torch", "torchvision")):
        raise ModuleNotFoundError(f"No module named {{name!r}}")
    return orig(name, package=package)


optional_deps.import_module = _fake_import_module

module = {module!r}
ok = False
err = None
try:
    importlib.import_module(module)
except ImportError as exc:
    ok = "pyimgano[torch]" in str(exc)
    err = str(exc)
except Exception as exc:
    ok = False
    err = f"{{type(exc).__name__}}: {{exc}}"

print(json.dumps({{"module": module, "ok": ok, "error": err}}))
""",
    )

    assert payload.get("module") == module
    assert payload.get("ok") is True, payload.get("error")
