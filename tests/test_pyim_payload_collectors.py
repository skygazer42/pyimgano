from __future__ import annotations

import pytest


def test_empty_pyim_payload_kwargs_covers_all_shared_fields() -> None:
    from pyimgano.pyim_list_spec import ALL_PAYLOAD_FIELDS
    from pyimgano.services.pyim_payload_collectors import empty_pyim_payload_kwargs

    payload_kwargs = empty_pyim_payload_kwargs()

    assert tuple(payload_kwargs.keys()) == ALL_PAYLOAD_FIELDS
    assert payload_kwargs == {field_name: [] for field_name in ALL_PAYLOAD_FIELDS}


def test_collect_pyim_payload_field_builds_typed_dataset_items(monkeypatch) -> None:
    import pyimgano.datasets.converters as dataset_converters
    from pyimgano.pyim_contracts import PyimDatasetSummary, PyimListRequest
    from pyimgano.services.pyim_payload_collectors import collect_pyim_payload_field

    class _Converter:
        name = "custom"
        description = "Custom dataset"
        requires_category = True

    monkeypatch.setattr(dataset_converters, "list_dataset_converters", lambda: [_Converter()])

    datasets = collect_pyim_payload_field("datasets", PyimListRequest(list_kind="datasets"))

    assert datasets == [
        PyimDatasetSummary(
            name="manifest",
            description="JSONL manifest dataset with explicit paths/splits for industrial workflows.",
            requires_category=True,
        ),
        PyimDatasetSummary(
            name="mvtec",
            description="MVTec AD public benchmark dataset layout.",
            requires_category=True,
        ),
        PyimDatasetSummary(
            name="custom",
            description="Custom dataset",
            requires_category=True,
        )
    ]


def test_collect_pyim_payload_field_rejects_unknown_field() -> None:
    from pyimgano.pyim_contracts import PyimListRequest
    from pyimgano.services.pyim_payload_collectors import collect_pyim_payload_field

    with pytest.raises(KeyError, match="Unsupported pyim payload field: unknown"):
        collect_pyim_payload_field("unknown", PyimListRequest())
