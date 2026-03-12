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


def test_importing_services_package_does_not_eagerly_import_benchmark_service() -> None:
    payload = _run_python(
        """
import json
import sys
import pyimgano.services

print(json.dumps({
    "benchmark_loaded": "pyimgano.services.benchmark_service" in sys.modules,
    "doctor_loaded": "pyimgano.services.doctor_service" in sys.modules,
}))
"""
    )

    assert payload == {
        "benchmark_loaded": False,
        "doctor_loaded": False,
    }


def test_services_package_still_exposes_public_symbols() -> None:
    import pyimgano.services as services

    assert services.BenchmarkRunRequest.__name__ == "BenchmarkRunRequest"
    assert callable(services.collect_doctor_payload)


def test_services_package_declares_explicit_export_source_map() -> None:
    import importlib

    import pyimgano.services as services

    assert list(services._SERVICE_EXPORT_SOURCES) == list(services.__all__)

    for export_name, module_name in services._SERVICE_EXPORT_SOURCES.items():
        module = importlib.import_module(module_name)
        assert export_name in module.__all__


def test_services_package_declares_grouped_export_spec() -> None:
    import pyimgano.services as services

    flattened_exports: list[str] = []
    flattened_sources: dict[str, str] = {}

    for group_name, items in services._SERVICE_EXPORT_GROUPS:
        assert isinstance(group_name, str)
        assert group_name
        assert items

        for export_name, module_name in items:
            flattened_exports.append(export_name)
            flattened_sources[export_name] = module_name

    assert flattened_exports == list(services.__all__)
    assert flattened_sources == services._SERVICE_EXPORT_SOURCES


def test_services_package_resolves_exports_without_loading_unrelated_service_modules() -> None:
    payload = _run_python(
        """
import json
import sys
import pyimgano.services as services

_ = services.resolve_model_options

print(json.dumps({
    "benchmark_loaded": "pyimgano.services.benchmark_service" in sys.modules,
    "model_options_loaded": "pyimgano.services.model_options" in sys.modules,
}))
"""
    )

    assert payload == {
        "benchmark_loaded": False,
        "model_options_loaded": True,
    }
