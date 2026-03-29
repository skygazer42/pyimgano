from __future__ import annotations


def test_validate_required_presence_flag_matches_boolean_contract() -> None:
    from pyimgano.reporting.deploy_bundle_validation_helpers import (
        validate_required_presence_flag,
    )

    assert validate_required_presence_flag(True, field_name="required_flag", actual=True) == []
    assert validate_required_presence_flag(False, field_name="required_flag", actual=True) == [
        "required_flag does not match manifest artifact refs."
    ]


def test_validate_exact_mapping_rejects_non_mapping_and_mismatch() -> None:
    from pyimgano.reporting.deploy_bundle_validation_helpers import validate_exact_mapping

    assert validate_exact_mapping({}, field_name="contract", expected={}) == []
    assert validate_exact_mapping([], field_name="contract", expected={}) == [
        "contract must be a JSON object/dict."
    ]
    assert validate_exact_mapping({"a": 1}, field_name="contract", expected={"a": 2}) == [
        "contract does not match computed bundle contract."
    ]


def test_validate_artifact_refs_requires_existing_manifest_entries(tmp_path) -> None:
    from pyimgano.reporting.deploy_bundle_validation_helpers import validate_artifact_refs

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "infer_config.json").write_text("{}", encoding="utf-8")

    errors = validate_artifact_refs(
        {"infer_config": "infer_config.json", "calibration_card": "missing.json"},
        field_name="bundle_artifact_refs",
        root=bundle_root,
        entry_paths={"infer_config.json"},
    )

    assert errors == [
        "bundle_artifact_refs.calibration_card points to missing file: missing.json"
    ]


def test_validate_operator_contract_digests_skips_source_run_values_when_source_missing(
    tmp_path,
) -> None:
    from pyimgano.reporting.deploy_bundle_validation_helpers import (
        validate_operator_contract_digests_map,
    )

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    errors = validate_operator_contract_digests_map(
        {
            "source_run_operator_contract_sha256": "0" * 64,
        },
        actual={},
        source_available=False,
        key_types={
            "source_run_operator_contract_sha256": (str, type(None)),
        },
    )

    assert errors == []


def test_validate_weight_audit_files_reports_model_card_and_manifest_errors(tmp_path) -> None:
    from pyimgano.reporting.deploy_bundle_validation_helpers import validate_weight_audit_files

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "weights_manifest.json").write_text("{", encoding="utf-8")
    (bundle_root / "model_card.json").write_text("{", encoding="utf-8")

    errors = validate_weight_audit_files(
        bundle_root,
        check_hashes=False,
    )

    assert any("weights_manifest.json" in item for item in errors)
    assert any("model_card.json" in item for item in errors)
