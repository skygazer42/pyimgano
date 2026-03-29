from __future__ import annotations

from pathlib import Path


def test_collect_existing_artifact_refs_returns_only_present_files(tmp_path: Path) -> None:
    from pyimgano.reporting.deploy_bundle_contract_helpers import collect_existing_artifact_refs

    root = tmp_path / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    (root / "infer_config.json").write_text("{}", encoding="utf-8")

    refs = collect_existing_artifact_refs(
        root,
        paths={
            "infer_config": "infer_config.json",
            "calibration_card": "calibration_card.json",
        },
    )

    assert refs == {"infer_config": "infer_config.json"}


def test_build_artifact_roles_groups_paths_by_role() -> None:
    from pyimgano.reporting.deploy_bundle_contract_helpers import build_artifact_roles

    roles = build_artifact_roles(
        [
            {"path": "infer_config.json", "role": "infer_config"},
            {"path": "operator_contract.json", "role": "operator_contract"},
            {"path": "weights/model.pt", "role": "checkpoint"},
        ]
    )

    assert roles == {
        "checkpoint": ["weights/model.pt"],
        "infer_config": ["infer_config.json"],
        "operator_contract": ["operator_contract.json"],
    }


def test_build_artifact_digests_maps_paths_to_sha256() -> None:
    from pyimgano.reporting.deploy_bundle_contract_helpers import build_artifact_digests

    digests = build_artifact_digests(
        [
            {"path": "b.json", "sha256": "b" * 64},
            {"path": "a.json", "sha256": "a" * 64},
        ]
    )

    assert digests == {
        "a.json": "a" * 64,
        "b.json": "b" * 64,
    }


def test_required_artifacts_present_requires_all_named_refs() -> None:
    from pyimgano.reporting.deploy_bundle_contract_helpers import required_artifacts_present

    assert required_artifacts_present(
        {"report": "report.json", "config": "config.json"},
        required_names=("report", "config"),
    ) is True
    assert required_artifacts_present(
        {"report": "report.json"},
        required_names=("report", "config"),
    ) is False
