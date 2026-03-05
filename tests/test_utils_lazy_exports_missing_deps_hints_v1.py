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


def test_utils_lazy_export_missing_cv2_has_actionable_install_hint() -> None:
    payload = _run_py(
        r"""
import importlib.abc
import importlib.machinery
import json
import sys


BLOCK_ROOTS = {"cv2"}


class _BlockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise ModuleNotFoundError(f"No module named {spec.name!r}", name=spec.name)

    def exec_module(self, module):
        return None


class _BlockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = str(fullname).split(".", 1)[0]
        if root in BLOCK_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, _BlockLoader())
        return None


sys.meta_path.insert(0, _BlockFinder())

import pyimgano.utils as u

ok = False
err = None
try:
    _ = u.gaussian_blur
except ImportError as exc:
    ok = "pip install 'opencv-python'" in str(exc)
    err = str(exc)
except Exception as exc:
    ok = False
    err = f"{type(exc).__name__}: {exc}"

print(json.dumps({"ok": ok, "error": err}))
""",
    )
    assert payload.get("ok") is True, payload.get("error")


def test_utils_lazy_export_missing_matplotlib_has_actionable_viz_extras_hint() -> None:
    payload = _run_py(
        r"""
import importlib.abc
import importlib.machinery
import json
import sys


BLOCK_ROOTS = {"matplotlib", "seaborn"}


class _BlockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise ModuleNotFoundError(f"No module named {spec.name!r}", name=spec.name)

    def exec_module(self, module):
        return None


class _BlockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = str(fullname).split(".", 1)[0]
        if root in BLOCK_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, _BlockLoader())
        return None


sys.meta_path.insert(0, _BlockFinder())

import pyimgano.utils as u

ok = False
err = None
try:
    _ = u.plot_roc_curve
except ImportError as exc:
    ok = "pyimgano[viz]" in str(exc)
    err = str(exc)
except Exception as exc:
    ok = False
    err = f"{type(exc).__name__}: {exc}"

print(json.dumps({"ok": ok, "error": err}))
""",
    )
    assert payload.get("ok") is True, payload.get("error")

