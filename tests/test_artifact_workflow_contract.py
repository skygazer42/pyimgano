from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "artifact-e2e.yml"


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [a-z0-9][a-z0-9-]*:\n|\Z)",
        workflow,
    )
    assert match is not None, f"missing workflow job {name!r}"
    return str(match.group("body"))


def _published_scripts() -> set[str]:
    pyproject = _read("pyproject.toml")
    match = re.search(
        r"(?ms)^\[project\.scripts\]\n(?P<body>.*?)(?=^\[|\Z)",
        pyproject,
    )
    assert match is not None
    return {
        line.split("=", 1)[0].strip()
        for line in match.group("body").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }


def test_artifact_gate_is_reusable_and_has_independent_format_jobs() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert re.search(r"(?m)^  workflow_call:\s*$", workflow)
    assert re.search(r"(?m)^  workflow_dispatch:\s*$", workflow)
    for name in ("build-wheel", "native", "onnx", "torchscript", "openvino"):
        _job(workflow, name)

    for name in ("native", "onnx", "torchscript", "openvino", "docs-security"):
        body = _job(workflow, name)
        assert "runs-on: ubuntu-latest" in body
        assert "needs: build-wheel" in body
        assert "actions/download-artifact@v8" in body
        assert "python-version: '3.10'" in body


def test_artifact_gate_builds_one_wheel_and_uses_only_wheel_installs() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    build = _job(workflow, "build-wheel")

    assert workflow.count("python -m build --wheel") == 1
    assert "assert len(wheels) == 1" in build
    assert "actions/upload-artifact@v7" in build
    assert "if-no-files-found: error" in build
    assert "pip install -e" not in workflow
    assert "pip install --editable" not in workflow

    for name in ("native", "onnx", "torchscript", "openvino", "docs-security"):
        body = _job(workflow, name)
        assert "*.whl" in body
        assert "-m pip check" in body


def test_each_declared_runtime_exports_relocates_loads_and_infers() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    expected = {
        "native": ("[torch]", "EXPECTED_BACKEND: pyimgano"),
        "onnx": ("[onnx-export]", "EXPECTED_BACKEND: onnxruntime"),
        "torchscript": ("[torch]", "EXPECTED_BACKEND: torchscript"),
        "openvino": ("[openvino-export]", "EXPECTED_BACKEND: openvino"),
    }

    for name, (creation_extra, backend) in expected.items():
        body = _job(workflow, name)
        assert creation_extra in body
        assert backend in body
        assert "test_trained_graph_export.py" in body
        assert "artifact_manifest.json" in body
        assert "-relocated" in body
        assert 'pyimgano-export" --help' in body
        assert 'pyimgano-artifact"' in body
        assert 'pyimgano-infer"' in body
        assert "results.jsonl" in body


def test_runtime_only_jobs_prove_torch_is_absent_and_skips_are_errors() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    onnx = _job(workflow, "onnx")
    openvino = _job(workflow, "openvino")

    assert "[onnx-runtime]" in onnx
    assert 'find_spec("onnxruntime") is not None' in onnx
    assert 'find_spec("torch") is None' in onnx
    assert "PYIMGANO_E2E_EXPECT_WHEEL: '1'" in onnx
    assert "tests/test_artifact_portability_e2e.py" in onnx

    assert "[openvino-runtime]" in openvino
    assert 'find_spec("openvino") is not None' in openvino
    assert 'find_spec("torch") is None' in openvino

    assert "continue-on-error" not in workflow
    assert not re.search(r"(?m)^\s+if:\s", workflow)
    assert "|| true" not in workflow
    for name in ("native", "onnx", "torchscript", "openvino", "docs-security"):
        body = _job(workflow, name)
        assert "pytest-error-for-skips" in body
        assert "--error-for-skips" in body


def test_runtime_wheel_import_checks_run_outside_the_checkout() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    for name in ("native", "onnx", "torchscript", "openvino"):
        body = _job(workflow, name)
        runtime_step = body[body.index("Relocate, load, and infer") :]
        assert runtime_step.index('cd "$RUNNER_TEMP"') < runtime_step.index("import pyimgano")


def test_onnx_and_torchscript_jobs_gate_real_ecod_composite_relocation_parity() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    onnx = _job(workflow, "onnx")
    torchscript = _job(workflow, "torchscript")

    for body in (onnx, torchscript):
        assert "cp tests/test_ecod_composite_artifact.py" in body
        assert "$RUNNER_TEMP/test_ecod_composite_artifact.py::" in body
        assert (
            "test_ecod_composite_export_relocate_delete_source_and_fresh_load_score_parity" in body
        )
        assert "--error-for-skips" in body

    assert "vision_onnx_ecod-onnx-.onnx-onnxruntime" in onnx
    assert "onnx-runtime-venv/bin/python" in onnx
    assert 'find_spec("torch") is None' in onnx
    assert "vision_torchscript_ecod-torchscript-.pt-torchscript" in torchscript
    assert "torchscript-runtime-venv/bin/python" in torchscript
    assert "--trust-checkpoint" in torchscript


def test_publish_prerequisites_cover_docs_security_version_and_every_script() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    contracts = _job(workflow, "docs-security")

    assert "tools/audit_repo_links.py" in contracts
    assert "tests/test_trained_artifact_docs_contract.py" in contracts
    assert "tests/test_artifact_security.py" in contracts
    assert "tests/test_artifact_security_e2e.py" in contracts
    assert "tests/test_package_version_sync.py" in contracts
    assert "tests/test_publishing_docs_contract.py" in contracts
    assert "tests/test_publish_workflow_contract.py" in contracts
    assert "tests/test_artifact_workflow_contract.py" in contracts
    assert "[onnx-runtime]" in contracts
    listed = set(re.findall(r"(?m)^            ([a-z][a-z0-9-]*)(?: \\)?$", contracts))
    assert _published_scripts() <= listed


def test_ci_and_publish_call_the_same_artifact_gate() -> None:
    ci = _read(".github/workflows/ci.yml")
    publish = _read(".github/workflows/publish.yml")

    for workflow in (ci, publish):
        called = _job(workflow, "artifact-e2e")
        assert "uses: ./.github/workflows/artifact-e2e.yml" in called

    assert "artifact-e2e" in re.search(
        r"(?ms)^  build:\n(?P<body>.*?)(?=^  [a-z0-9][a-z0-9-]*:\n|\Z)", ci
    ).group("body")
    publish_job = _job(publish, "build-and-publish")
    assert "needs: [artifact-e2e, release-readiness]" in publish_job
    assert "actions/download-artifact@v8" in publish_job
    assert "python -m build --sdist" in publish_job


def test_release_gate_matrix_is_explicit_and_cpu_only() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "ARTIFACT_GATE_OS: ubuntu-latest" in workflow
    assert "ARTIFACT_GATE_PYTHON: '3.10'" in workflow
    assert "ARTIFACT_GATE_PROVIDER: CPU" in workflow
    assert "CPUExecutionProvider" in _job(workflow, "onnx")
    assert "· CPU" in _job(workflow, "native")
    assert "· CPU" in _job(workflow, "torchscript")
    assert "· CPU" in _job(workflow, "openvino")
