from __future__ import annotations

import copy

import pytest


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
            "dynamic_spatial": False,
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


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("input",),
        ("input", "resize"),
        ("input", "scale"),
        ("input", "normalize"),
        ("outputs",),
        ("outputs", "score"),
        ("outputs", "anomaly_map"),
    ],
)
def test_contract_rejects_unknown_keys_at_every_schema_level(path) -> None:
    from pyimgano.artifacts.onnx_contract import (
        ONNXImportContractError,
        normalize_onnx_import_contract,
    )

    payload = _contract()
    payload["outputs"]["anomaly_map"] = {
        "name": "map",
        "layout": "NHW",
        "resize_to_source": True,
    }
    target = payload
    for part in path:
        target = target[part]
    target["unexpected"] = "value"

    with pytest.raises(ONNXImportContractError, match="unknown keys"):
        normalize_onnx_import_contract(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("input", "dynamic_batch"), "false"),
        (("input", "dynamic_spatial"), 0),
        (("outputs", "anomaly_map", "resize_to_source"), "true"),
    ],
)
def test_contract_requires_real_json_booleans(path, value) -> None:
    from pyimgano.artifacts.onnx_contract import (
        ONNXImportContractError,
        normalize_onnx_import_contract,
    )

    payload = _contract()
    payload["outputs"]["anomaly_map"] = {
        "name": "map",
        "layout": "NHW",
        "resize_to_source": True,
    }
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ONNXImportContractError, match="boolean"):
        normalize_onnx_import_contract(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["input"].update(size=[True, 4]), "integer"),
        (
            lambda value: value["outputs"]["score"].update(
                transform="select_index", axis=True, index=0
            ),
            "integer",
        ),
        (
            lambda value: value["outputs"]["score"].update(
                transform="select_index", axis=-1, index=-1
            ),
            ">= 0",
        ),
        (
            lambda value: value["outputs"].update(
                anomaly_map={
                    "name": "map",
                    "layout": "NCHW",
                    "resize_to_source": True,
                    "channel": True,
                }
            ),
            "integer",
        ),
    ],
)
def test_contract_rejects_bool_or_negative_selection_integers(mutate, message) -> None:
    from pyimgano.artifacts.onnx_contract import (
        ONNXImportContractError,
        normalize_onnx_import_contract,
    )

    payload = _contract()
    mutate(payload)
    with pytest.raises(ONNXImportContractError, match=message):
        normalize_onnx_import_contract(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["input"]["scale"].update(divisor=float("nan")),
        lambda value: value["input"]["scale"].update(divisor=float("inf")),
        lambda value: value["input"]["normalize"].update(mean=[0.0, float("nan"), 0.0]),
        lambda value: value["input"]["normalize"].update(std=[1.0, float("inf"), 1.0]),
    ],
)
def test_contract_rejects_non_finite_numbers(mutate) -> None:
    from pyimgano.artifacts.onnx_contract import (
        ONNXImportContractError,
        normalize_onnx_import_contract,
    )

    payload = _contract()
    mutate(payload)
    with pytest.raises(ONNXImportContractError, match="finite"):
        normalize_onnx_import_contract(payload)


def test_contract_enforces_nonzero_scale_std_and_color_channel_lengths() -> None:
    from pyimgano.artifacts.onnx_contract import (
        ONNXImportContractError,
        normalize_onnx_import_contract,
    )

    zero_divisor = _contract()
    zero_divisor["input"]["scale"]["divisor"] = 0.0
    with pytest.raises(ONNXImportContractError, match="divisor"):
        normalize_onnx_import_contract(zero_divisor)

    zero_std = _contract()
    zero_std["input"]["normalize"]["std"][1] = 0.0
    with pytest.raises(ONNXImportContractError, match="std"):
        normalize_onnx_import_contract(zero_std)

    wrong_channels = _contract()
    wrong_channels["input"]["color_space"] = "GRAY"
    with pytest.raises(ONNXImportContractError, match="1 values"):
        normalize_onnx_import_contract(wrong_channels)


def test_contract_allows_negative_selection_axis_but_not_ambiguous_fields() -> None:
    from pyimgano.artifacts.onnx_contract import (
        ONNXImportContractError,
        normalize_onnx_import_contract,
    )

    payload = _contract()
    payload["outputs"]["score"].update(
        transform="softmax_select",
        axis=-1,
        index=1,
    )
    normalized = normalize_onnx_import_contract(payload)
    assert normalized["outputs"]["score"]["axis"] == -1

    ambiguous = _contract()
    ambiguous["outputs"]["score"]["axis"] = 0
    with pytest.raises(ONNXImportContractError, match="valid only"):
        normalize_onnx_import_contract(ambiguous)


def test_contract_rejects_score_map_name_collision_and_non_integer_version() -> None:
    from pyimgano.artifacts.onnx_contract import (
        ONNXImportContractError,
        normalize_onnx_import_contract,
    )

    collision = _contract()
    collision["outputs"]["anomaly_map"] = {
        "name": "score",
        "layout": "NHW",
        "resize_to_source": True,
    }
    with pytest.raises(ONNXImportContractError, match="distinct"):
        normalize_onnx_import_contract(collision)

    bad_version = copy.deepcopy(_contract())
    bad_version["schema_version"] = "1"
    with pytest.raises(ONNXImportContractError, match="schema_version=1"):
        normalize_onnx_import_contract(bad_version)
