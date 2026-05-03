from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(script: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_importing_workbench_package_does_not_eagerly_import_runner_or_preflight() -> None:
    payload = _run_python(
        """
import json
import sys
import pyimgano.workbench as workbench

print(json.dumps({
    "runner_loaded": "pyimgano.workbench.runner" in sys.modules,
    "preflight_loaded": "pyimgano.workbench.preflight" in sys.modules,
    "workbench_loaded": "pyimgano.workbench" in sys.modules,
}))
"""
    )

    assert payload == {
        "runner_loaded": False,
        "preflight_loaded": False,
        "workbench_loaded": True,
    }


def test_workbench_package_exposes_public_symbols() -> None:
    import pyimgano.workbench as workbench

    assert workbench.WorkbenchConfig.__name__ == "WorkbenchConfig"
    assert workbench.PreflightReport.__name__ == "PreflightReport"
    assert callable(workbench.run_workbench)
    assert callable(workbench.run_preflight)
    assert callable(workbench.build_infer_config_payload)


def test_workbench_package_declares_explicit_export_source_map() -> None:
    import importlib

    import pyimgano.workbench as workbench

    assert list(workbench._WORKBENCH_EXPORT_SOURCES) == list(workbench.__all__)

    for export_name, module_name in workbench._WORKBENCH_EXPORT_SOURCES.items():
        module = importlib.import_module(module_name)
        assert hasattr(module, export_name)


def test_workbench_package_declares_grouped_export_spec() -> None:
    import pyimgano.workbench as workbench

    flattened_exports: list[str] = []
    flattened_sources: dict[str, str] = {}

    for group_name, items in workbench._WORKBENCH_EXPORT_GROUPS:
        assert isinstance(group_name, str)
        assert group_name
        assert items

        for export_name, module_name in items:
            flattened_exports.append(export_name)
            flattened_sources[export_name] = module_name

    assert flattened_exports == list(workbench.__all__)
    assert flattened_sources == workbench._WORKBENCH_EXPORT_SOURCES


def test_workbench_package_resolves_exports_without_loading_unrelated_modules() -> None:
    payload = _run_python(
        """
import json
import sys
import pyimgano.workbench as workbench

_ = workbench.WorkbenchConfig

print(json.dumps({
    "config_loaded": "pyimgano.workbench.config" in sys.modules,
    "runner_loaded": "pyimgano.workbench.runner" in sys.modules,
    "preflight_loaded": "pyimgano.workbench.preflight" in sys.modules,
}))
"""
    )

    assert payload == {
        "config_loaded": True,
        "runner_loaded": False,
        "preflight_loaded": False,
    }


def test_workbench_package_resolves_preflight_type_exports_without_loading_preflight_runtime() -> (
    None
):
    payload = _run_python(
        """
import json
import sys
import pyimgano.workbench as workbench

_ = workbench.PreflightReport

print(json.dumps({
    "preflight_types_loaded": "pyimgano.workbench.preflight_types" in sys.modules,
    "preflight_loaded": "pyimgano.workbench.preflight" in sys.modules,
}))
"""
    )

    assert payload == {
        "preflight_types_loaded": True,
        "preflight_loaded": False,
    }
