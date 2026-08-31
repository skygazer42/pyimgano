from __future__ import annotations

import copy
import math

import pytest


def _graph_input() -> dict[str, object]:
    return {
        "kind": "image_batch",
        "name": "input",
        "dtype": "float32",
        "layout": "NCHW",
        "color_space": "RGB",
        "size": [224, 224],
        "dynamic_axes": {"batch": True},
        "resize": {"mode": "stretch", "interpolation": "bilinear"},
        "scale": {"divisor": 255.0},
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def _score() -> dict[str, object]:
    return {
        "name": "score",
        "transform": "identity",
        "score_order": "higher_is_more_anomalous",
    }


def test_current_native_autoencoder_and_import_contracts_are_valid() -> None:
    from pyimgano.artifacts.io_contract import (
        validate_artifact_input_contract,
        validate_artifact_output_contract,
    )

    native = {"kind": "image_batch", "dtype": "uint8", "layout": "HWC"}
    assert (
        validate_artifact_input_contract(
            native,
            layout="native_detector",
            backend="pyimgano",
        )
        == native
    )

    autoencoder = _graph_input()
    normalized = validate_artifact_input_contract(
        autoencoder,
        layout="single_graph",
        backend="onnxruntime",
    )
    assert normalized == autoencoder
    assert normalized is not autoencoder

    imported = _graph_input()
    imported.update(
        {
            "dtype": "uint8",
            "layout": "NHWC",
            "color_space": "GRAY",
            "dynamic_axes": {"batch": False, "spatial": True},
            "normalize": {"mean": [0.0], "std": [1.0]},
        }
    )
    assert (
        validate_artifact_input_contract(
            imported,
            layout="composite",
            backend="pyimgano",
        )
        == imported
    )

    outputs = {
        "score": {
            **_score(),
            "output_index": 0,
        },
        "anomaly_map": {
            "name": "anomaly_map",
            "output_index": 1,
            "layout": "NHW",
            "transform": "identity",
            "resize_to_source": True,
        },
    }
    assert validate_artifact_output_contract(outputs) == outputs


def test_native_input_rejects_tensor_preprocessing_and_non_rgb_color() -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_input_contract,
    )

    native = {"kind": "image_batch", "dtype": "uint8", "layout": "HWC"}
    for field, value in (
        ("name", "input"),
        ("size", [224, 224]),
        ("dynamic_axes", {"batch": True}),
        ("resize", {"mode": "stretch", "interpolation": "bilinear"}),
        ("scale", {"divisor": 255.0}),
        ("normalize", {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}),
    ):
        invalid = {**native, field: value}
        with pytest.raises(ArtifactIOContractError, match=r"input_contract: unknown keys"):
            validate_artifact_input_contract(
                invalid,
                layout="native_detector",
                backend="pyimgano",
            )

    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.color_space"):
        validate_artifact_input_contract(
            {**native, "color_space": "BGR"},
            layout="native_detector",
            backend="pyimgano",
        )


@pytest.mark.parametrize(
    ("field", "value", "error_path"),
    [
        ("dtype", "int8", r"input_contract\.dtype"),
        ("layout", "HWC", r"input_contract\.layout"),
        ("color_space", "RGBA", r"input_contract\.color_space"),
        ("size", [True, 224], r"input_contract\.size\[0\]"),
        ("size", [0, 224], r"input_contract\.size\[0\]"),
        ("size", [224], r"input_contract\.size"),
    ],
)
def test_graph_input_rejects_invalid_required_tensor_fields(
    field: str,
    value: object,
    error_path: str,
) -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_input_contract,
    )

    contract = _graph_input()
    contract[field] = value
    with pytest.raises(ArtifactIOContractError, match=error_path):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="onnxruntime",
        )


def test_graph_input_bounds_size_and_requires_strict_dynamic_booleans() -> None:
    from pyimgano.artifacts.io_contract import (
        MAX_IMAGE_DIMENSION,
        ArtifactIOContractError,
        validate_artifact_input_contract,
    )

    contract = _graph_input()
    contract["size"] = [MAX_IMAGE_DIMENSION + 1, 224]
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.size\[0\]"):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="onnxruntime",
        )

    contract = _graph_input()
    contract["dynamic_axes"] = {"batch": 1, "spatial": False}
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.dynamic_axes\.batch"):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="onnxruntime",
        )

    contract["dynamic_axes"] = {"channels": True}
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.dynamic_axes"):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="onnxruntime",
        )


def test_graph_input_validates_resize_fill_and_rejects_unknown_keys() -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_input_contract,
    )

    contract = _graph_input()
    contract["resize"] = {
        "mode": "letterbox",
        "interpolation": "area",
        "fill": [0, 127, 255],
    }
    assert (
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="openvino",
        )["resize"]
        == contract["resize"]
    )

    invalid = copy.deepcopy(contract)
    invalid["resize"]["fill"] = [0, True, 255]
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.resize\.fill\[1\]"):
        validate_artifact_input_contract(
            invalid,
            layout="single_graph",
            backend="openvino",
        )

    invalid = _graph_input()
    invalid["resize"]["fill"] = 0
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.resize\.fill"):
        validate_artifact_input_contract(
            invalid,
            layout="single_graph",
            backend="onnxruntime",
        )

    invalid = _graph_input()
    invalid["ignored"] = True
    with pytest.raises(ArtifactIOContractError, match=r"input_contract: unknown keys"):
        validate_artifact_input_contract(
            invalid,
            layout="single_graph",
            backend="onnxruntime",
        )


@pytest.mark.parametrize("value", [0, True, math.nan, math.inf, -math.inf])
def test_scale_divisor_is_finite_numeric_and_nonzero(value: object) -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_input_contract,
    )

    contract = _graph_input()
    contract["scale"] = {"divisor": value, "multiplier": 1.0, "offset": 0.0}
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.scale\.divisor"):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="torchscript",
        )


def test_scale_and_normalize_reject_nonfinite_bool_and_wrong_channel_count() -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_input_contract,
    )

    contract = _graph_input()
    contract["scale"] = {"multiplier": math.nan}
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.scale\.multiplier"):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="onnxruntime",
        )

    contract = _graph_input()
    contract["normalize"] = {"mean": [0.0, True, 0.0], "std": [1.0, 1.0, 1.0]}
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.normalize\.mean\[1\]"):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="onnxruntime",
        )

    contract["normalize"] = {"mean": [0.0], "std": [1.0]}
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.normalize\.mean"):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="onnxruntime",
        )

    contract["color_space"] = "GRAY"
    contract["normalize"] = {"mean": [0.0], "std": [0.0]}
    with pytest.raises(ArtifactIOContractError, match=r"input_contract\.normalize\.std\[0\]"):
        validate_artifact_input_contract(
            contract,
            layout="single_graph",
            backend="onnxruntime",
        )


def test_output_score_selection_rules_are_strict() -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_output_contract,
    )

    selected = {
        "score": {
            "name": "probabilities",
            "transform": "softmax_select",
            "score_order": "lower_is_more_anomalous",
            "axis": -1,
            "index": 1,
            "output_index": 0,
        }
    }
    assert validate_artifact_output_contract(selected) == selected

    missing = copy.deepcopy(selected)
    del missing["score"]["axis"]
    with pytest.raises(ArtifactIOContractError, match=r"output_contract\.score\.axis"):
        validate_artifact_output_contract(missing)

    invalid = copy.deepcopy(selected)
    invalid["score"]["index"] = True
    with pytest.raises(ArtifactIOContractError, match=r"output_contract\.score\.index"):
        validate_artifact_output_contract(invalid)

    stray = {"score": {**_score(), "axis": 1}}
    with pytest.raises(ArtifactIOContractError, match=r"output_contract\.score\.axis"):
        validate_artifact_output_contract(stray)

    negative_output = {"score": {**_score(), "output_index": -1}}
    with pytest.raises(ArtifactIOContractError, match=r"output_contract\.score\.output_index"):
        validate_artifact_output_contract(negative_output)


def test_output_map_contract_is_strict_and_names_are_distinct() -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_output_contract,
    )

    output = {
        "score": _score(),
        "anomaly_map": {
            "name": "map",
            "layout": "NCHW",
            "channel": 0,
            "resize_to_source": False,
            "transform": "sigmoid",
            "output_index": 1,
        },
    }
    assert validate_artifact_output_contract(output) == output

    invalid = copy.deepcopy(output)
    invalid["anomaly_map"]["resize_to_source"] = 1
    with pytest.raises(
        ArtifactIOContractError,
        match=r"output_contract\.anomaly_map\.resize_to_source",
    ):
        validate_artifact_output_contract(invalid)

    invalid = copy.deepcopy(output)
    del invalid["anomaly_map"]["resize_to_source"]
    with pytest.raises(
        ArtifactIOContractError,
        match=r"output_contract\.anomaly_map\.resize_to_source",
    ):
        validate_artifact_output_contract(invalid)

    invalid = copy.deepcopy(output)
    invalid["anomaly_map"].update({"layout": "NHW", "channel": 0})
    with pytest.raises(ArtifactIOContractError, match=r"output_contract\.anomaly_map\.channel"):
        validate_artifact_output_contract(invalid)

    invalid = copy.deepcopy(output)
    invalid["anomaly_map"]["name"] = "score"
    with pytest.raises(ArtifactIOContractError, match=r"output_contract\.anomaly_map\.name"):
        validate_artifact_output_contract(invalid)

    invalid = copy.deepcopy(output)
    invalid["anomaly_map"]["channel_index"] = 0
    with pytest.raises(ArtifactIOContractError, match=r"output_contract\.anomaly_map"):
        validate_artifact_output_contract(invalid)


def test_contract_top_levels_reject_unknown_keys_and_non_objects() -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_input_contract,
        validate_artifact_output_contract,
    )

    with pytest.raises(ArtifactIOContractError, match=r"input_contract"):
        validate_artifact_input_contract(
            [],
            layout="single_graph",
            backend="onnxruntime",
        )
    with pytest.raises(ArtifactIOContractError, match=r"output_contract: unknown keys"):
        validate_artifact_output_contract({"score": _score(), "metadata": {}})
    with pytest.raises(ArtifactIOContractError, match=r"output_contract\.score"):
        validate_artifact_output_contract({})


def test_layout_and_backend_arguments_are_validated_with_field_paths() -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_input_contract,
    )

    with pytest.raises(ArtifactIOContractError, match=r"layout"):
        validate_artifact_input_contract(_graph_input(), layout="graph", backend="onnxruntime")
    with pytest.raises(ArtifactIOContractError, match=r"runtime\.backend"):
        validate_artifact_input_contract(
            _graph_input(),
            layout="single_graph",
            backend="pyimgano",
        )


def test_contract_error_is_a_value_error() -> None:
    from pyimgano.artifacts.io_contract import (
        ArtifactIOContractError,
        validate_artifact_output_contract,
    )

    with pytest.raises(ValueError) as exc_info:
        validate_artifact_output_contract({})
    assert isinstance(exc_info.value, ArtifactIOContractError)
    assert str(exc_info.value).startswith("output_contract.score:")
