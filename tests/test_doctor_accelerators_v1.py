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


def test_doctor_accelerators_flag_emits_stable_json_payload() -> None:
    payload = _run_py(
        r"""
import importlib.abc
import importlib.machinery
import json
import sys


BLOCK_ROOTS = {"torch", "torchvision", "onnxruntime", "onnx", "onnxscript", "openvino"}


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

from pyimgano import doctor_cli

rc = doctor_cli.main(["--json", "--accelerators"])
assert int(rc) == 0
""",
    )

    assert payload.get("tool") == "pyimgano-doctor"
    assert "accelerators" in payload

    acc = payload.get("accelerators") or {}
    assert isinstance(acc, dict)

    torch_acc = acc.get("torch") or {}
    assert torch_acc.get("available") is False
    assert "pyimgano[torch]" in str(torch_acc.get("install_hint") or "")

    ort_acc = acc.get("onnxruntime") or {}
    assert ort_acc.get("available") is False
    assert "pyimgano[onnx]" in str(ort_acc.get("install_hint") or "")

    ov_acc = acc.get("openvino") or {}
    assert ov_acc.get("available") is False
    assert "pyimgano[openvino]" in str(ov_acc.get("install_hint") or "")


def test_doctor_without_accelerators_flag_does_not_include_accelerators() -> None:
    payload = _run_py(
        r"""
import json
from pyimgano import doctor_cli

rc = doctor_cli.main(["--json"])
assert int(rc) == 0
""",
    )
    assert payload.get("tool") == "pyimgano-doctor"
    assert "accelerators" not in payload
