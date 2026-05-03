from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pyimgano.utils.optional_deps as optional_deps

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


def test_optional_import_and_require_basic_contracts() -> None:
    mod, err = optional_deps.optional_import("math")
    assert err is None
    assert mod is not None
    assert hasattr(mod, "sqrt")

    mod, err = optional_deps.optional_import("pyimgano__definitely_missing_module__xyz")
    assert mod is None
    assert err is not None

    mod = optional_deps.require("math")
    assert hasattr(mod, "sqrt")


def test_optional_import_suppresses_failed_import_stderr(capsys) -> None:
    module_name = "pyimgano__optional_native_backend_with_noisy_import__"

    class _NoisyImportLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module) -> None:
            print("native extension traceback detail", file=sys.stderr)
            raise ImportError("native extension failed")

    class _NoisyImportFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == module_name:
                return importlib.machinery.ModuleSpec(fullname, _NoisyImportLoader())
            return None

    finder = _NoisyImportFinder()
    sys.meta_path.insert(0, finder)
    try:
        mod, err = optional_deps.optional_import(module_name)
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop(module_name, None)

    captured = capsys.readouterr()
    assert mod is None
    assert isinstance(err, ImportError)
    assert captured.err == ""


def test_require_reports_actionable_install_hints() -> None:
    with pytest.raises(ImportError) as excinfo:
        optional_deps.require("pyimgano__definitely_missing_module__xyz")
    assert "Optional dependency" in str(excinfo.value)
    assert "pip install" in str(excinfo.value)

    with pytest.raises(ImportError) as excinfo:
        optional_deps.require(
            "pyimgano__definitely_missing_module__xyz", extra="clip", purpose="unit-test"
        )
    message = str(excinfo.value)
    assert "pyimgano[clip]" in message
    assert "unit-test" in message


@pytest.mark.parametrize(
    ("blocked_roots", "attr_name", "expected_hint"),
    [
        ({"cv2"}, "gaussian_blur", "pip install 'opencv-python'"),
        ({"matplotlib", "seaborn"}, "plot_roc_curve", "pyimgano[viz]"),
    ],
)
def test_utils_lazy_exports_report_actionable_hints(
    blocked_roots: set[str], attr_name: str, expected_hint: str
) -> None:
    blocked_roots_literal = sorted(blocked_roots)
    payload = _run_py(
        f"""
import importlib.abc
import importlib.machinery
import json
import sys


BLOCK_ROOTS = set({blocked_roots_literal!r})


class _BlockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise ModuleNotFoundError(f"No module named {{spec.name!r}}", name=spec.name)

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
    _ = getattr(u, "{attr_name}")
except ImportError as exc:
    ok = "{expected_hint}" in str(exc)
    err = str(exc)
except Exception as exc:
    ok = False
    err = f"{{type(exc).__name__}}: {{exc}}"

print(json.dumps({{"ok": ok, "error": err}}))
"""
    )

    assert payload.get("ok") is True, payload.get("error")
