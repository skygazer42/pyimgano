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


@pytest.mark.parametrize("module", ["pyimgano.models.imdd", "pyimgano.models.qmcd"])
def test_numba_backed_model_modules_raise_actionable_import_error(module: str) -> None:
    """Models that depend on numba should fail with an extras install hint."""

    payload = _run_py(
        rf"""
import builtins
import importlib
import json

orig_import = builtins.__import__


def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "numba" or str(name).startswith("numba."):
        raise ModuleNotFoundError("No module named 'numba'")
    return orig_import(name, globals, locals, fromlist, level)


builtins.__import__ = _blocked_import

module = {module!r}
ok = False
err = None
try:
    importlib.import_module(module)
except ImportError as exc:
    ok = "pyimgano[numba]" in str(exc)
    err = str(exc)
except Exception as exc:
    ok = False
    err = f"{{type(exc).__name__}}: {{exc}}"

print(json.dumps({{"module": module, "ok": ok, "error": err}}))
""",
    )

    assert payload.get("module") == module
    assert payload.get("ok") is True, payload.get("error")
