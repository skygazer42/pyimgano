from __future__ import annotations


def test_metadata_contract_lists_expected_fields() -> None:
    from pyimgano.models.metadata_contract import metadata_contract_fields

    fields = metadata_contract_fields()
    names = [item["name"] for item in fields]
    assert names == [
        "paper",
        "year",
        "family",
        "type",
        "supervision",
        "supports_pixel_map",
        "requires_checkpoint",
        "weights_source",
    ]


def test_audit_metadata_contract_flags_missing_and_invalid_fields() -> None:
    from pyimgano.models.metadata_contract import audit_metadata_contract
    from pyimgano.models.registry import ModelRegistry

    registry = ModelRegistry()

    class _PixelModel:
        def get_anomaly_map(self, X):  # noqa: ANN001, ANN201 - test helper
            return X

    registry.register(
        "toy_pixel_model",
        _PixelModel,
        tags=("vision", "deep", "patchcore", "memory_bank"),
        metadata={
            "paper": "Toy Paper",
            "year": 3024,
            "supervision": "mystery-mode",
            "requires_checkpoint": True,
        },
    )

    payload = audit_metadata_contract(registry)
    assert payload["summary"]["total_models"] == 1
    assert payload["required_missing_by_model"]["toy_pixel_model"] == ["weights_source"]
    assert "toy_pixel_model" not in payload["recommended_missing_by_model"]
    invalid = payload["invalid_fields_by_model"]["toy_pixel_model"]
    assert any(item["field"] == "year" for item in invalid)
    assert any(item["field"] == "supervision" for item in invalid)


def test_audit_metadata_contract_accepts_default_weights_source_for_known_wrappers() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.metadata_contract import audit_metadata_contract
    from pyimgano.models.registry import MODEL_REGISTRY

    payload = audit_metadata_contract(
        MODEL_REGISTRY,
        names=[
            "vision_patchcore_anomalib",
            "vision_patchcore_inspection_checkpoint",
            "vision_onnx_ecod",
            "vision_torchscript_ecod",
        ],
    )
    assert payload["required_missing_by_model"] == {}


def test_metadata_contract_resolves_supervision_from_explicit_tags() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.metadata_contract import resolve_metadata_contract_payload
    from pyimgano.models.registry import MODEL_REGISTRY

    assert (
        resolve_metadata_contract_payload(MODEL_REGISTRY.info("cutpaste"))["supervision"]
        == "self-supervised"
    )
    assert (
        resolve_metadata_contract_payload(MODEL_REGISTRY.info("vision_devnet"))["supervision"]
        == "weakly-supervised"
    )
    assert (
        resolve_metadata_contract_payload(MODEL_REGISTRY.info("core_ocsvm"))["supervision"]
        == "one-class"
    )
