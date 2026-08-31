from __future__ import annotations

import copy
import hashlib

import pytest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _policy() -> dict[str, object]:
    return {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "postprocess": {"image_threshold": {"threshold": 0.5}},
    }


def _base(layout: str, backend: str) -> dict[str, object]:
    input_contract = (
        {"kind": "image_batch", "dtype": "uint8", "layout": "HWC"}
        if layout == "native_detector"
        else {
            "kind": "image_batch",
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "size": [224, 224],
        }
    )
    return {
        "schema_family": "pyimgano-artifact",
        "schema_version": 1,
        "layout": layout,
        "runtime": {
            "backend": backend,
            "allowed_providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "verified_providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "entrypoint": "state/detector.pyim",
        },
        "input_contract": input_contract,
        "output_contract": {
            "score": {
                "name": "score",
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            }
        },
        "components": [],
        "policy_ref": {"path": "infer_config.json"},
        "compatibility": {
            "pyimgano": ">=0.10,<0.11",
            "python": ">=3.9,<3.13",
            "platforms": ["linux-x86_64"],
            "runtime_versions": {},
            "adapter": {"id": "reference", "version": 1},
            "codecs": [{"id": "tensor-state", "version": 1}],
        },
        "verification": {
            "level": "reference_parity",
            "reference_backend": "pyimgano",
            "report": {
                "path": "verification/parity.json",
                "size_bytes": 2,
                "sha256": _sha(b"{}"),
            },
        },
    }


def _component(
    component_id: str,
    path: str,
    role: str,
    fmt: str,
    serialization: str,
) -> dict[str, object]:
    data = component_id.encode("utf-8")
    return {
        "id": component_id,
        "path": path,
        "role": role,
        "format": fmt,
        "serialization": serialization,
        "size_bytes": len(data),
        "sha256": _sha(data),
    }


def test_native_layout_requires_registry_codec_and_trained_state() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _base("native_detector", "pyimgano")
    payload["model"] = {
        "registry_name": "ae_resnet_unet",
        "constructor_kwargs": {"device": "cpu"},
    }
    payload["components"] = [
        _component(
            "trained-state",
            "state/detector.pyim",
            "trained_state",
            "pyimgano-state",
            "safe-data",
        )
    ]
    assert build_artifact_manifest(payload, _policy())["layout"] == "native_detector"

    bad = copy.deepcopy(payload)
    bad["runtime"]["backend"] = "onnxruntime"
    with pytest.raises(ArtifactManifestError, match="native_detector"):
        build_artifact_manifest(bad, _policy())

    bad = copy.deepcopy(payload)
    bad["components"] = []
    with pytest.raises(ArtifactManifestError, match="component|trained_state"):
        build_artifact_manifest(bad, _policy())


def test_single_graph_allows_vendor_graph_without_registry_model() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _base("single_graph", "onnxruntime")
    payload["runtime"]["entrypoint"] = "model/vendor.onnx"
    payload["components"] = [
        _component("graph", "model/vendor.onnx", "runtime_model", "onnx", "onnx")
    ]
    payload["verification"]["level"] = "runtime_smoke"
    payload["verification"].pop("reference_backend")
    assert "model" not in build_artifact_manifest(payload, _policy())

    bad = copy.deepcopy(payload)
    bad["runtime"]["entrypoint"] = "model/unlisted.onnx"
    with pytest.raises(ArtifactManifestError, match="entrypoint"):
        build_artifact_manifest(bad, _policy())

    bad = copy.deepcopy(payload)
    bad["components"].append(
        _component("graph-2", "model/second.onnx", "runtime_model", "onnx", "onnx")
    )
    with pytest.raises(ArtifactManifestError, match="exactly one"):
        build_artifact_manifest(bad, _policy())


def test_openvino_layout_requires_exact_xml_bin_pair() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _base("single_graph", "openvino")
    payload["runtime"]["entrypoint"] = "model/detector.xml"
    payload["components"] = [
        _component(
            "graph",
            "model/detector.xml",
            "runtime_model",
            "openvino-ir",
            "openvino-ir",
        ),
        _component(
            "weights",
            "model/detector.bin",
            "openvino_weights",
            "openvino-weights",
            "safe-data",
        ),
    ]
    assert build_artifact_manifest(payload, _policy())["layout"] == "single_graph"

    bad = copy.deepcopy(payload)
    bad["components"][1]["path"] = "model/unrelated.bin"
    with pytest.raises(ArtifactManifestError, match="bin sibling"):
        build_artifact_manifest(bad, _policy())

    bad = copy.deepcopy(payload)
    bad["components"].append(
        _component(
            "weights-2",
            "model/detector-2.bin",
            "openvino_weights",
            "openvino-weights",
            "safe-data",
        )
    )
    with pytest.raises(ArtifactManifestError, match="exactly one"):
        build_artifact_manifest(bad, _policy())

    bad = copy.deepcopy(payload)
    bad["components"][1]["format"] = "opaque-weights"
    with pytest.raises(ArtifactManifestError, match="openvino-weights"):
        build_artifact_manifest(bad, _policy())


def test_composite_layout_requires_named_acyclic_bound_components() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _base("composite", "pyimgano")
    payload["model"] = {"registry_name": "vision_onnx_ecod"}
    payload["runtime"].pop("entrypoint")
    payload["runtime"]["composition_adapter"] = {"id": "embedding-core", "version": 1}
    payload["compatibility"]["adapter"] = {"id": "embedding-core", "version": 1}
    payload["components"] = [
        _component("embed", "model/embed.onnx", "runtime_model", "onnx", "onnx"),
        _component("core", "state/core.pyim", "trained_state", "pyimgano-state", "safe-data"),
    ]
    payload["composition"] = {
        "nodes": [
            {"id": "features", "component": "embed", "depends_on": []},
            {"id": "score", "component": "core", "depends_on": ["features"]},
        ],
        "bindings": {"output.score": "score"},
    }
    assert build_artifact_manifest(payload, _policy())["layout"] == "composite"

    bad = copy.deepcopy(payload)
    bad["runtime"].pop("composition_adapter")
    with pytest.raises(ArtifactManifestError, match="composition_adapter"):
        build_artifact_manifest(bad, _policy())

    bad = copy.deepcopy(payload)
    bad["composition"]["nodes"][0]["depends_on"] = ["score"]
    with pytest.raises(ArtifactManifestError, match="cycle"):
        build_artifact_manifest(bad, _policy())

    bad = copy.deepcopy(payload)
    bad["composition"]["nodes"][1]["component"] = "missing"
    with pytest.raises(ArtifactManifestError, match="component"):
        build_artifact_manifest(bad, _policy())


@pytest.mark.parametrize("field", ["module", "class", "import_path"])
def test_manifest_rejects_arbitrary_python_import_paths(field: str) -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _base("native_detector", "pyimgano")
    payload["model"] = {
        "registry_name": "ae_resnet_unet",
        "constructor_kwargs": {field: "evil.payload"},
    }
    payload["components"] = [
        _component(
            "trained-state",
            "state/detector.pyim",
            "trained_state",
            "pyimgano-state",
            "safe-data",
        )
    ]
    with pytest.raises(ArtifactManifestError, match=field):
        build_artifact_manifest(payload, _policy())
