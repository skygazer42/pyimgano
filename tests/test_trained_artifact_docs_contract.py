from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _extra_body(pyproject: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(name)}\s*=\s*\[\n(?P<body>.*?)^\]$",
        pyproject,
    )
    assert match is not None, f"missing optional extra {name!r}"
    return str(match.group("body"))


def test_runtime_and_export_extras_are_split_without_forcing_torch() -> None:
    pyproject = _read("pyproject.toml")

    onnx_runtime = _extra_body(pyproject, "onnx-runtime")
    assert '"onnx>=' in onnx_runtime
    assert '"onnxruntime>=' in onnx_runtime
    assert "torch" not in onnx_runtime

    openvino_runtime = _extra_body(pyproject, "openvino-runtime")
    assert '"openvino>=' in openvino_runtime
    assert "torch" not in openvino_runtime

    assert '"pyimgano[torch,onnx-runtime]"' in _extra_body(pyproject, "onnx-export")
    assert '"pyimgano[onnx-export,openvino-runtime]"' in _extra_body(pyproject, "openvino-export")
    assert '"pyimgano[onnx-export]"' in _extra_body(pyproject, "onnx")
    assert '"pyimgano[openvino-runtime]"' in _extra_body(pyproject, "openvino")
    assert '"pyimgano[onnx-export,openvino-export]"' in _extra_body(pyproject, "deploy")


def test_artifact_console_scripts_are_published() -> None:
    pyproject = _read("pyproject.toml")
    assert 'pyimgano-export = "pyimgano.export_cli:main"' in pyproject
    assert 'pyimgano-artifact = "pyimgano.artifact_cli:main"' in pyproject


def test_public_docs_cover_the_safe_trained_artifact_paths() -> None:
    readme = _read("README.md")
    guide = _read("docs/TRAINED_ARTIFACTS.md")
    cli = _read("docs/CLI_REFERENCE.md")

    for text in (readme, guide, cli):
        assert "pyimgano-export" in text
        assert "pyimgano-artifact" in text
        assert "--artifact" in text

    assert "from pyimgano.inference import infer, load_artifact" in guide
    assert '"schema_family": "pyimgano-onnx-import"' in guide
    assert '"score_order": "higher_is_more_anomalous"' in guide
    assert "Raw ONNX files must first be imported" in readme
    assert "artifact_refs" in guide
    assert "onnx-runtime" in guide
    assert "openvino-runtime" in guide
    assert "`ae_resnet_unet` is the only" in guide
    assert "`vision_patchcore`" in guide
    assert "`vision_onnx_ecod`" in guide
    assert "`vision_torchscript_ecod`" in guide
    assert "Ubuntu x86_64, Python 3.10" in guide
    assert "--trust-checkpoint" in guide
    assert "docs.pytorch.org/docs/stable/generated/torch.jit.load.html" in guide
    assert "`vision_onnx_ecod`" in cli
    assert "`vision_torchscript_ecod`" in cli


def test_site_export_replaces_invalid_legacy_deployment_claims() -> None:
    export_doc = _read("docs/site/deployment/export.md")

    assert "--from-run runs/<run_dir>" in export_doc
    assert "pyimgano-artifact import" in export_doc
    assert "pyimgano-onnx-import" in export_doc
    assert "pyimgano-export-onnx" in export_doc
    assert "embedding/backbone exporter" in export_doc
    assert "`vision_onnx_ecod`" in export_doc
    assert "`vision_torchscript_ecod`" in export_doc
    assert "Ubuntu x86_64" in export_doc
    assert "--trust-checkpoint" in export_doc
    assert "torch.jit.load.html" in export_doc
    assert "mo --input_model" not in export_doc
    assert "--model onnx" not in export_doc


def test_documented_reference_support_matrix_matches_registry() -> None:
    import pyimgano.models  # noqa: F401 - populate lazy model/export registries
    from pyimgano.models.registry import model_info

    reference = model_info("ae_resnet_unet")["capabilities"]["trained_export"]
    assert reference["native"]["status"] == "supported"
    assert {name: reference[name]["status"] for name in ("onnx", "torchscript", "openvino")} == {
        "onnx": "conditional",
        "torchscript": "conditional",
        "openvino": "conditional",
    }

    patchcore = model_info("vision_patchcore")["capabilities"]["trained_export"]
    assert {cell["status"] for cell in patchcore.values()} == {"unsupported"}

    onnx_ecod = model_info("vision_onnx_ecod")["capabilities"]["trained_export"]
    assert onnx_ecod["onnx"]["status"] == "conditional"
    assert onnx_ecod["onnx"]["layout"] == "composite"
    assert {name: onnx_ecod[name]["status"] for name in ("native", "torchscript", "openvino")} == {
        name: "unsupported" for name in ("native", "torchscript", "openvino")
    }

    torchscript_ecod = model_info("vision_torchscript_ecod")["capabilities"]["trained_export"]
    assert torchscript_ecod["torchscript"]["status"] == "conditional"
    assert torchscript_ecod["torchscript"]["layout"] == "composite"
    assert {name: torchscript_ecod[name]["status"] for name in ("native", "onnx", "openvino")} == {
        name: "unsupported" for name in ("native", "onnx", "openvino")
    }


def test_checked_in_quickstart_fixture_matches_current_parsers_and_contract() -> None:
    fixture = json.loads(_read("tests/fixtures/docs/artifact_quickstart.json"))
    commands = fixture["commands"]
    assert fixture["certified_model"] == "ae_resnet_unet"

    from pyimgano.artifact_cli import _build_parser as artifact_parser
    from pyimgano.artifacts.onnx_contract import normalize_onnx_import_contract
    from pyimgano.bundle_cli import _build_parser as bundle_parser
    from pyimgano.export_cli import _build_parser as export_parser
    from pyimgano.infer_cli import _build_parser as infer_parser

    assert export_parser().parse_args(commands["export_from_run"]).formats == ["native"]
    assert artifact_parser().parse_args(commands["import_onnx"]).command == "import"
    assert infer_parser().parse_args(commands["infer_artifact"]).artifact_format == "native"
    assert bundle_parser().parse_args(commands["run_bundle"]).artifact_format == "native"

    normalized = normalize_onnx_import_contract(fixture["onnx_import_contract"])
    assert normalized["input"]["dynamic_axes"] == {"batch": True, "spatial": False}
    assert normalized["outputs"]["score"]["name"] == "score"
