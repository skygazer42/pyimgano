from __future__ import annotations

import subprocess
import sys
import textwrap


def test_python_module_doctor_help_mentions_extras_recommendation_flags() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pyimgano.doctor_cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--recommend-extras" in completed.stdout
    assert "--for-command" in completed.stdout
    assert "--for-model" in completed.stdout


def test_python_module_pyim_help_mentions_selection_flags() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pyimgano.pyim_cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--objective" in completed.stdout
    assert "--selection-profile" in completed.stdout
    assert "--topk" in completed.stdout


def test_python_module_doctor_help_does_not_require_cv2() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import importlib.machinery
        import runpy
        import sys


        class _BlockLoader(importlib.abc.Loader):
            def create_module(self, spec):
                raise ModuleNotFoundError(f"No module named {spec.name!r}", name=spec.name)

            def exec_module(self, module):
                return None


        class _BlockFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if str(fullname).split(".", 1)[0] == "cv2":
                    return importlib.machinery.ModuleSpec(fullname, _BlockLoader())
                return None


        sys.meta_path.insert(0, _BlockFinder())
        sys.argv = ["pyimgano.doctor_cli", "--help"]
        runpy.run_module("pyimgano.doctor_cli", run_name="__main__")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--recommend-extras" in completed.stdout
