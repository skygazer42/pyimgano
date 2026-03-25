import json
import subprocess
import sys
import types


def test_audit_registry_skips_optional_dependency_import_errors(monkeypatch):
    import tools.audit_registry as audit_registry

    fake_models = types.ModuleType("pyimgano.models")
    fake_models.list_models = lambda: ["vision_fastflow", "vision_ecod"]  # type: ignore[attr-defined]

    fake_registry = types.ModuleType("pyimgano.models.registry")

    def _fake_model_info(name: str):
        if name == "vision_fastflow":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return {"name": name}

    fake_registry.model_info = _fake_model_info  # type: ignore[attr-defined]

    fake_jsonable = types.ModuleType("pyimgano.utils.jsonable")
    fake_jsonable.to_jsonable = lambda payload: payload  # type: ignore[attr-defined]

    fake_extras = types.ModuleType("pyimgano.utils.extras")
    fake_extras.extra_for_root_module = lambda root: "torch" if root == "torch" else None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pyimgano.models", fake_models)
    monkeypatch.setitem(sys.modules, "pyimgano.models.registry", fake_registry)
    monkeypatch.setitem(sys.modules, "pyimgano.utils.jsonable", fake_jsonable)
    monkeypatch.setitem(sys.modules, "pyimgano.utils.extras", fake_extras)

    assert audit_registry.audit_registry() == []


def test_audit_registry_skips_optional_dependency_import_errors_wrapped_as_importerror(
    monkeypatch,
):
    import tools.audit_registry as audit_registry

    fake_models = types.ModuleType("pyimgano.models")
    fake_models.list_models = lambda: ["core_imdd"]  # type: ignore[attr-defined]

    fake_registry = types.ModuleType("pyimgano.models.registry")

    def _fake_model_info(_name: str):
        raise ImportError(
            "Optional dependency 'numba' is required for IMDD/LMDD detector acceleration."
        )

    fake_registry.model_info = _fake_model_info  # type: ignore[attr-defined]

    fake_jsonable = types.ModuleType("pyimgano.utils.jsonable")
    fake_jsonable.to_jsonable = lambda payload: payload  # type: ignore[attr-defined]

    fake_extras = types.ModuleType("pyimgano.utils.extras")
    fake_extras.extra_for_root_module = lambda root: "numba" if root == "numba" else None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pyimgano.models", fake_models)
    monkeypatch.setitem(sys.modules, "pyimgano.models.registry", fake_registry)
    monkeypatch.setitem(sys.modules, "pyimgano.utils.jsonable", fake_jsonable)
    monkeypatch.setitem(sys.modules, "pyimgano.utils.extras", fake_extras)

    assert audit_registry.audit_registry() == []


def test_audit_registry_script_runs():
    subprocess.run([sys.executable, "tools/audit_registry.py"], check=True)


def test_audit_registry_script_can_report_metadata_contract_json():
    proc = subprocess.run(
        [sys.executable, "tools/audit_registry.py", "--metadata-contract", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode in {0, 1}
    payload = json.loads(proc.stdout)
    assert payload["summary"]["total_models"] > 0
    assert "contract_fields" in payload


def test_audit_registry_metadata_summary_shows_required_issues_but_no_invalid_fields():
    proc = subprocess.run(
        [sys.executable, "tools/audit_registry.py", "--metadata-contract", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode in {0, 1}
    payload = json.loads(proc.stdout)
    assert payload["summary"]["models_with_invalid_fields"] == 0
