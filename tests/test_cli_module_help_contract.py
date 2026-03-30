from __future__ import annotations

import subprocess
import sys


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
