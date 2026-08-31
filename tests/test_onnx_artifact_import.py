from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest


def _write_score_model(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input"], ["score"], axes=[1, 2, 3], keepdims=0)],
        "score-model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 4, 4])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save_model(model, str(path))


def _write_external_score_model(
    path: Path,
    *,
    location: str = "weights/bias.bin",
) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    external_path = path.parent.joinpath(*location.split("/"))
    external_path.parent.mkdir(parents=True, exist_ok=True)
    bias = numpy_helper.from_array(
        np.full((1, 3, 4, 4), 2.0, dtype=np.float32),
        name="bias",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["input", "bias"], ["adjusted"]),
            helper.make_node("ReduceMean", ["adjusted"], ["score"], axes=[1, 2, 3], keepdims=0),
        ],
        "external-score-model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 4, 4])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None])],
        initializer=[bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=0,
    )


def _write_score_map_model(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input"], ["map"], axes=[1], keepdims=1),
            helper.make_node("ReduceMean", ["map"], ["score"], axes=[1, 2, 3], keepdims=0),
        ],
        "score-map-model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 4, 4])],
        [
            helper.make_tensor_value_info("score", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("map", TensorProto.FLOAT, [None, 1, 4, 4]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save_model(model, str(path))


def _write_class_score_model(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["logits"], axis=1),
        ],
        "class-score-model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 4, 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save_model(model, str(path))


def _write_score_model_for_opset(path: Path, opset: int) -> None:
    """Write a ReduceMean graph valid on both sides of the opset-18 change."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    initializers = []
    if opset >= 18:
        axes = helper.make_tensor("axes", TensorProto.INT64, [3], [1, 2, 3])
        initializers.append(axes)
        node = helper.make_node("ReduceMean", ["input", "axes"], ["score"], keepdims=0)
    else:
        node = helper.make_node("ReduceMean", ["input"], ["score"], axes=[1, 2, 3], keepdims=0)
    graph = helper.make_graph(
        [node],
        "score-model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 4, 4])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    onnx.save_model(model, str(path))


def _contract() -> dict:
    return {
        "schema_family": "pyimgano-onnx-import",
        "schema_version": 1,
        "input": {
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "size": [4, 4],
            "dynamic_batch": True,
            "resize": {"mode": "stretch", "interpolation": "bilinear"},
            "scale": {"divisor": 255.0},
            "normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        },
        "outputs": {
            "score": {
                "name": "score",
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            }
        },
    }


def _score_map_contract() -> dict:
    contract = _contract()
    contract["outputs"]["anomaly_map"] = {
        "name": "map",
        "layout": "NCHW",
        "resize_to_source": True,
    }
    return contract


def _selection_contract() -> dict:
    contract = _contract()
    contract["outputs"]["score"] = {
        "name": "logits",
        "transform": "softmax_select",
        "axis": -1,
        "index": 1,
        "score_order": "higher_is_more_anomalous",
    }
    return contract


def _rewrite_model(path: Path, mutate) -> None:
    onnx = pytest.importorskip("onnx")
    model = onnx.load_model(str(path), load_external_data=False)
    mutate(model, onnx)
    onnx.save_model(model, str(path))


def test_import_onnx_requires_explicit_contract(tmp_path):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)

    with pytest.raises(ValueError, match="contract"):
        import_onnx(model, contract=None, out=tmp_path / "artifact")


def test_import_onnx_builds_score_only_artifact(tmp_path):
    from pyimgano.artifacts import load_artifact_manifest
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")

    result = import_onnx(model, contract=contract_path, out=tmp_path / "artifact")

    root = Path(result["artifact_root"])
    assert (root / "model" / "model.onnx").is_file()
    assert (root / "infer_config.json").is_file()
    assert (root / "verification" / "runtime_smoke.json").is_file()
    manifest = load_artifact_manifest(root)
    from pyimgano.artifacts.compatibility import current_platform_tag

    assert manifest["layout"] == "single_graph"
    assert manifest["runtime"]["backend"] == "onnxruntime"
    assert manifest["verification"]["level"] == "runtime_smoke"
    assert manifest["compatibility"]["platforms"] == [current_platform_tag()]
    assert manifest["compatibility"]["onnx_opset"] == 13
    assert manifest["compatibility"]["onnx_ir"] == 10
    assert manifest["compatibility"]["runtime_versions"]["onnxruntime"] == ">=1.18,<2"
    assert "model" not in json.loads((root / "infer_config.json").read_text(encoding="utf-8"))


def test_import_onnx_rejects_wrong_output_name(tmp_path):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)
    contract = _contract()
    contract["outputs"]["score"]["name"] = "embedding"

    with pytest.raises(ValueError, match="output"):
        import_onnx(model, contract=contract, out=tmp_path / "artifact")


@pytest.mark.parametrize(
    ("ir_version", "accepted"),
    [(2, False), (3, True), (10, True), (11, False)],
)
def test_import_onnx_enforces_ir_range(tmp_path, ir_version, accepted):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)
    _rewrite_model(model, lambda graph, _onnx: setattr(graph, "ir_version", ir_version))

    if accepted:
        result = import_onnx(model, contract=_contract(), out=tmp_path / "artifact")
        assert Path(result["artifact_root"]).is_dir()
    else:
        with pytest.raises(ValueError, match="IR version"):
            import_onnx(model, contract=_contract(), out=tmp_path / "artifact")


@pytest.mark.parametrize(("opset", "accepted"), [(6, False), (7, True), (21, True), (22, False)])
def test_import_onnx_enforces_default_opset_range(tmp_path, opset, accepted):
    from pyimgano.artifacts import load_artifact_manifest
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model_for_opset(model, opset)

    if accepted:
        result = import_onnx(model, contract=_contract(), out=tmp_path / "artifact")
        assert Path(result["artifact_root"]).is_dir()
        manifest = load_artifact_manifest(result["artifact_root"])
        expected_runtime = ">=1.18,<2" if opset == 21 else ">=1.17,<2"
        assert manifest["compatibility"]["runtime_versions"]["onnxruntime"] == expected_runtime
    else:
        with pytest.raises(ValueError, match="opset version"):
            import_onnx(model, contract=_contract(), out=tmp_path / "artifact")


def test_import_onnx_rejects_ml_opset_above_runtime_cap(tmp_path):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)

    def add_ml5(graph, onnx):
        graph.opset_import.append(onnx.helper.make_opsetid("ai.onnx.ml", 5))

    _rewrite_model(model, add_ml5)
    with pytest.raises(ValueError, match="opset version"):
        import_onnx(model, contract=_contract(), out=tmp_path / "artifact")


def test_import_onnx_rejects_custom_domain_even_without_opset_declaration(tmp_path):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)
    _rewrite_model(
        model,
        lambda graph, _onnx: setattr(graph.graph.node[0], "domain", "vendor.custom"),
    )

    with pytest.raises(ValueError, match="custom ONNX operator domain"):
        import_onnx(model, contract=_contract(), out=tmp_path / "artifact")


def test_import_onnx_checker_rejects_unknown_standard_operator(tmp_path):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)
    _rewrite_model(
        model,
        lambda graph, _onnx: setattr(graph.graph.node[0], "op_type", "NotARealOnnxOp"),
    )

    with pytest.raises(ValueError, match="schema/operator validation"):
        import_onnx(model, contract=_contract(), out=tmp_path / "artifact")


def test_import_onnx_rejects_extra_non_initializer_input(tmp_path):
    onnx = pytest.importorskip("onnx")
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)

    def add_input(graph, _onnx):
        graph.graph.input.append(
            onnx.helper.make_tensor_value_info("context", onnx.TensorProto.FLOAT, [None, 1])
        )

    _rewrite_model(model, add_input)
    with pytest.raises(ValueError, match="exactly one"):
        import_onnx(model, contract=_contract(), out=tmp_path / "artifact")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda graph, onnx: setattr(
                graph.graph.output[0].type.tensor_type,
                "elem_type",
                onnx.TensorProto.INT64,
            ),
            "floating point",
        ),
        (
            lambda graph, _onnx: graph.graph.output[0]
            .type.tensor_type.shape.dim.add()
            .__setattr__("dim_value", 2),
            "score output must",
        ),
        (
            lambda graph, _onnx: (
                setattr(graph.graph.input[0].type.tensor_type.shape.dim[0], "dim_param", "batch"),
                setattr(
                    graph.graph.output[0].type.tensor_type.shape.dim[0],
                    "dim_param",
                    "different_batch",
                ),
            ),
            "batch symbol",
        ),
    ],
)
def test_import_onnx_rejects_invalid_score_metadata(tmp_path, mutation, message):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)
    _rewrite_model(model, mutation)
    with pytest.raises(ValueError, match=message):
        import_onnx(model, contract=_contract(), out=tmp_path / "artifact")


@pytest.mark.parametrize(
    ("axis", "index", "message"),
    [(0, 1, "non-batch"), (-1, 3, "index")],
)
def test_import_onnx_rejects_invalid_score_selection(tmp_path, axis, index, message):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_class_score_model(model)
    contract = _selection_contract()
    contract["outputs"]["score"].update(axis=axis, index=index)
    with pytest.raises(ValueError, match=message):
        import_onnx(model, contract=contract, out=tmp_path / "artifact")


@pytest.mark.parametrize(
    ("mutation", "contract_mutation", "message"),
    [
        (
            lambda graph, onnx: setattr(
                graph.graph.output[1].type.tensor_type,
                "elem_type",
                onnx.TensorProto.INT64,
            ),
            lambda contract: None,
            "floating point",
        ),
        (
            lambda graph, _onnx: None,
            lambda contract: contract["outputs"]["anomaly_map"].update(layout="NHW"),
            "does not match shape",
        ),
        (
            lambda graph, _onnx: None,
            lambda contract: contract["outputs"]["anomaly_map"].update(channel=1),
            "channel",
        ),
    ],
)
def test_import_onnx_rejects_invalid_map_metadata(tmp_path, mutation, contract_mutation, message):
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_map_model(model)
    _rewrite_model(model, mutation)
    contract = _score_map_contract()
    contract_mutation(contract)
    with pytest.raises(ValueError, match=message):
        import_onnx(model, contract=contract, out=tmp_path / "artifact")


def test_imported_score_model_matches_declared_raw_score(tmp_path):
    pytest.importorskip("onnxruntime")
    from pyimgano.artifacts.importers import import_onnx

    model = tmp_path / "model.onnx"
    _write_score_model(model)
    result = import_onnx(model, contract=_contract(), out=tmp_path / "artifact")

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(Path(result["artifact_root"]) / "model" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    sample = np.ones((2, 3, 4, 4), dtype=np.float32)
    score = session.run(["score"], {"input": sample})[0]
    np.testing.assert_allclose(score, np.ones((2,), dtype=np.float32))


def test_import_onnx_real_score_and_map_graph_roundtrip(tmp_path):
    from pyimgano.artifacts import load_artifact_manifest
    from pyimgano.artifacts.importers import import_onnx
    from pyimgano.inference import load_artifact

    source = tmp_path / "score-map.onnx"
    _write_score_map_model(source)
    result = import_onnx(source, contract=_score_map_contract(), out=tmp_path / "artifact")
    manifest = load_artifact_manifest(result["artifact_root"])
    assert manifest["output_contract"]["anomaly_map"]["layout"] == "NCHW"
    assert manifest["compatibility"]["runtime_versions"]["onnxruntime"] == ">=1.17,<2"

    runtime = load_artifact(result["artifact_root"])
    try:
        scores, maps = runtime.score_and_maps(
            [np.full((4, 4, 3), 255, dtype=np.uint8)], include_maps=True
        )
        np.testing.assert_allclose(scores, np.ones((1,), dtype=np.float32))
        assert maps is not None
        np.testing.assert_allclose(maps, np.ones((1, 4, 4), dtype=np.float32))
    finally:
        runtime.close()


def test_import_onnx_real_softmax_selection_roundtrip(tmp_path):
    from pyimgano.artifacts.importers import import_onnx
    from pyimgano.inference import load_artifact

    source = tmp_path / "class-score.onnx"
    _write_class_score_model(source)
    result = import_onnx(source, contract=_selection_contract(), out=tmp_path / "artifact")

    runtime = load_artifact(result["artifact_root"])
    try:
        scores = runtime.decision_function([np.full((4, 4, 3), 255, dtype=np.uint8)])
        np.testing.assert_allclose(scores, np.asarray([1.0 / 3.0], dtype=np.float32))
    finally:
        runtime.close()


def test_import_onnx_lower_score_order_roundtrip(tmp_path):
    from pyimgano.artifacts.importers import import_onnx
    from pyimgano.inference import load_artifact

    source = tmp_path / "lower-score.onnx"
    _write_score_model(source)
    contract = _contract()
    contract["outputs"]["score"]["score_order"] = "lower_is_more_anomalous"
    result = import_onnx(source, contract=contract, out=tmp_path / "artifact")

    runtime = load_artifact(result["artifact_root"])
    try:
        scores = runtime.decision_function([np.full((4, 4, 3), 255, dtype=np.uint8)])
        np.testing.assert_allclose(scores, np.asarray([-1.0], dtype=np.float32))
    finally:
        runtime.close()


def test_external_data_import_is_relocatable_and_has_exact_verified_closure(tmp_path):
    pytest.importorskip("onnxruntime")
    from pyimgano.artifacts import (
        load_artifact_manifest,
        stage_verified_artifact,
        verify_artifact_files,
    )
    from pyimgano.artifacts.importers import import_onnx
    from pyimgano.inference import infer, load_artifact

    source = tmp_path / "source" / "model.onnx"
    source.parent.mkdir()
    _write_external_score_model(source)
    result = import_onnx(source, contract=_contract(), out=tmp_path / "artifact")

    root = Path(result["artifact_root"])
    manifest = load_artifact_manifest(root)
    external = [item for item in manifest["components"] if item["role"] == "external_data"]
    assert [item["path"] for item in external] == ["model/weights/bias.bin"]
    verify_artifact_files(root, manifest)
    with stage_verified_artifact(root, manifest) as staging:
        assert staging.path_for("model/weights/bias.bin").is_file()

    relocated = tmp_path / "已发布 模型"
    shutil.copytree(root, relocated)
    shutil.rmtree(source.parent)
    runtime = load_artifact(relocated)
    try:
        result = infer(
            runtime,
            [np.zeros((4, 4, 3), dtype=np.uint8)],
            input_format="rgb_u8_hwc",
        )[0]
        assert result.score == pytest.approx(2.0)
    finally:
        runtime.close()


def test_external_data_closure_rejects_missing_and_unreferenced_components(tmp_path):
    from pyimgano.artifacts import (
        ArtifactSecurityError,
        load_artifact_manifest,
        stage_verified_artifact,
    )
    from pyimgano.artifacts.importers import import_onnx

    source = tmp_path / "source" / "model.onnx"
    source.parent.mkdir()
    _write_external_score_model(source)
    result = import_onnx(source, contract=_contract(), out=tmp_path / "artifact")
    root = Path(result["artifact_root"])
    manifest = load_artifact_manifest(root)

    missing = dict(manifest)
    missing["components"] = [
        item for item in manifest["components"] if item["role"] != "external_data"
    ]
    with pytest.raises(ArtifactSecurityError, match="missing from manifest"):
        stage_verified_artifact(root, missing)

    extra_path = root / "model" / "unused.bin"
    extra_path.write_bytes(b"unused")
    unexpected = dict(manifest)
    unexpected["components"] = list(manifest["components"]) + [
        {
            "path": "model/unused.bin",
            "role": "external_data",
            "format": "onnx-external-data",
            "serialization": "safe-data",
            "size_bytes": 6,
            "sha256": hashlib.sha256(b"unused").hexdigest(),
        }
    ]
    with pytest.raises(ArtifactSecurityError, match="unreferenced"):
        stage_verified_artifact(root, unexpected)


@pytest.mark.parametrize(
    "location",
    ["../outside.bin", "/tmp/outside.bin", "C:/outside.bin", "..\\outside.bin"],
)
def test_import_onnx_rejects_unsafe_external_data_locations(tmp_path, location):
    onnx = pytest.importorskip("onnx")
    from pyimgano.artifacts.importers import import_onnx

    source = tmp_path / "source" / "model.onnx"
    source.parent.mkdir()
    _write_external_score_model(source)
    model = onnx.load_model(str(source), load_external_data=False)
    for entry in model.graph.initializer[0].external_data:
        if entry.key == "location":
            entry.value = location
    source.write_bytes(model.SerializeToString())

    with pytest.raises(ValueError, match="Unsafe ONNX external-data location"):
        import_onnx(source, contract=_contract(), out=tmp_path / "artifact")


def test_import_onnx_rejects_symlinked_external_data_ancestor(tmp_path):
    from pyimgano.artifacts.importers import import_onnx

    source = tmp_path / "source" / "model.onnx"
    source.parent.mkdir()
    _write_external_score_model(source)
    outside = tmp_path / "outside-weights"
    (source.parent / "weights").rename(outside)
    try:
        (source.parent / "weights").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symlink"):
        import_onnx(source, contract=_contract(), out=tmp_path / "artifact")


def test_import_onnx_rejects_symlink_model_source(tmp_path):
    from pyimgano.artifacts.importers import import_onnx

    real_source = tmp_path / "real.onnx"
    source = tmp_path / "model.onnx"
    _write_score_model(real_source)
    try:
        source.symlink_to(real_source)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symlink"):
        import_onnx(source, contract=_contract(), out=tmp_path / "artifact")


def test_import_onnx_rejects_non_nfc_external_data_location(tmp_path):
    from pyimgano.artifacts.importers import import_onnx

    source = tmp_path / "source" / "model.onnx"
    source.parent.mkdir()
    _write_external_score_model(source, location="weights/e\u0301.bin")

    with pytest.raises(ValueError, match="NFC"):
        import_onnx(source, contract=_contract(), out=tmp_path / "artifact")


def test_import_onnx_fails_closed_without_secure_openat(monkeypatch, tmp_path):
    from pyimgano.artifacts import security
    from pyimgano.artifacts.importers import import_onnx

    source = tmp_path / "model.onnx"
    _write_score_model(source)
    monkeypatch.setattr(security, "_secure_source_openat_available", lambda: False)

    with pytest.raises(ValueError, match="fails closed"):
        import_onnx(source, contract=_contract(), out=tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX rename of an open file")
def test_import_onnx_validates_the_open_model_descriptor(monkeypatch, tmp_path):
    from pyimgano.artifacts import security
    from pyimgano.artifacts.importers import import_onnx

    source = tmp_path / "model.onnx"
    saved_source = tmp_path / "saved-model.onnx"
    _write_score_model(source)
    expected = source.read_bytes()
    real_copy = security._copy_open_source_file
    swapped = False

    def replace_after_open(descriptor, destination, *, maximum_bytes, label):
        nonlocal swapped
        if label == source.name and not swapped:
            swapped = True
            source.rename(saved_source)
            source.write_bytes(b"replacement that is not an ONNX model")
        return real_copy(
            descriptor,
            destination,
            maximum_bytes=maximum_bytes,
            label=label,
        )

    monkeypatch.setattr(security, "_copy_open_source_file", replace_after_open)
    result = import_onnx(source, contract=_contract(), out=tmp_path / "artifact")

    assert swapped
    assert (Path(result["artifact_root"]) / "model" / "model.onnx").read_bytes() == expected


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX directory descriptors")
def test_import_onnx_external_copy_resists_ancestor_symlink_swap(monkeypatch, tmp_path):
    from pyimgano.artifacts import security
    from pyimgano.artifacts.importers import import_onnx

    source = tmp_path / "source" / "model.onnx"
    source.parent.mkdir()
    _write_external_score_model(source)
    weights = source.parent / "weights"
    saved_weights = source.parent / "saved-weights"
    expected = (weights / "bias.bin").read_bytes()
    outside = tmp_path / "outside-weights"
    outside.mkdir()
    outside_bytes = np.full((1, 3, 4, 4), 5.0, dtype=np.float32).tobytes()
    (outside / "bias.bin").write_bytes(outside_bytes)
    real_open = security._open_source_regular_at
    swapped = False

    def swap_ancestor_after_open(directory_fd, name, *, display):
        nonlocal swapped
        if display == "weights/bias.bin" and not swapped:
            swapped = True
            weights.rename(saved_weights)
            try:
                weights.symlink_to(outside, target_is_directory=True)
            except OSError:
                pytest.skip("symlinks are unavailable on this platform")
        return real_open(directory_fd, name, display=display)

    monkeypatch.setattr(security, "_open_source_regular_at", swap_ancestor_after_open)
    result = import_onnx(source, contract=_contract(), out=tmp_path / "artifact")

    copied = Path(result["artifact_root"]) / "model" / "weights" / "bias.bin"
    assert swapped
    assert copied.read_bytes() == expected
    assert copied.read_bytes() != outside_bytes


def test_artifact_cli_import_delegates(monkeypatch, tmp_path, capsys):
    import pyimgano.artifact_cli as artifact_cli

    calls = []
    monkeypatch.setattr(
        artifact_cli,
        "import_onnx",
        lambda *args, **kwargs: calls.append((args, dict(kwargs)))
        or {"artifact_root": str(tmp_path / "artifact"), "artifact_id": "sha256:test"},
    )

    rc = artifact_cli.main(
        [
            "import",
            "--format",
            "onnx",
            "--model",
            "model.onnx",
            "--contract",
            "contract.json",
            "--out",
            str(tmp_path / "artifact"),
            "--json",
        ]
    )

    assert rc == 0
    assert calls[0][0] == ("model.onnx",)
    assert calls[0][1]["contract"] == "contract.json"
    assert json.loads(capsys.readouterr().out)["artifact_id"] == "sha256:test"


def test_artifact_cli_bind_policy_accepts_policy_path_and_emits_bound_identity(
    monkeypatch, tmp_path, capsys
):
    import pyimgano.artifact_cli as artifact_cli
    import pyimgano.artifacts as artifacts

    calls = []
    bound = tmp_path / "bound"
    monkeypatch.setattr(
        artifacts,
        "bind_policy",
        lambda source, policy, out, trust_checkpoint: calls.append(
            (source, policy, out, trust_checkpoint)
        )
        or bound,
    )
    monkeypatch.setattr(
        artifacts,
        "load_artifact_manifest",
        lambda path: {
            "artifact_id": "sha256:new",
            "runtime_id": "sha256:runtime",
            "policy_id": "sha256:policy",
            "verification": {"level": "reference_parity"},
        },
    )

    rc = artifact_cli.main(
        [
            "bind-policy",
            "--artifact",
            "source",
            "--policy",
            "policy.json",
            "--out",
            str(bound),
            "--json",
        ]
    )

    assert rc == 0
    assert calls == [("source", "policy.json", str(bound), False)]
    assert json.loads(capsys.readouterr().out) == {
        "artifact_id": "sha256:new",
        "artifact_root": str(bound),
        "policy_id": "sha256:policy",
        "runtime_id": "sha256:runtime",
        "verification_level": "reference_parity",
    }


def test_artifact_cli_validate_checks_the_full_file_closure(monkeypatch, tmp_path, capsys):
    import pyimgano.artifact_cli as artifact_cli
    import pyimgano.artifacts as artifacts

    root = tmp_path / "artifact"
    root.mkdir()
    calls = []
    manifest = {"artifact_id": "sha256:test"}
    monkeypatch.setattr(artifacts, "load_artifact_manifest", lambda path: manifest)
    monkeypatch.setattr(
        artifacts,
        "verify_artifact_files",
        lambda artifact_root, payload: calls.append((artifact_root, payload)),
    )

    rc = artifact_cli.main(["validate", str(root), "--json"])

    assert rc == 0
    assert calls == [(root, manifest)]
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
