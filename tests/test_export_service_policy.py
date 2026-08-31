from __future__ import annotations


def test_run_infer_config_is_reduced_to_artifact_local_policy() -> None:
    from pyimgano.artifacts import validate_artifact_policy
    from pyimgano.services.export_service import _policy_without_external_reconstruction_paths

    legacy = {
        "from_run": "/private/training/run",
        "checkpoint": {"path": "checkpoints/model.pkl"},
        "threshold": 0.7,
        "postprocess": {
            "image_threshold": {
                "threshold": 0.7,
                "score_order": "higher_is_more_anomalous",
            },
            "review_policy": {"reject_confidence_below": 0.8, "reject_label": -2},
        },
        "adaptation": {"save_maps": True},
        "defects": {"enabled": False},
        "prediction": {"reject_confidence_below": 0.8, "reject_label": -2},
        "artifact_quality": {"audit_refs": {"source": "/private/report.json"}},
    }

    policy = _policy_without_external_reconstruction_paths(
        legacy,
        model_name="ae_resnet_unet",
        category="bottle",
        model_kwargs={"device": "cpu", "checkpoint_path": "/private/model.pt"},
    )

    assert policy["schema_family"] == "pyimgano-artifact-policy"
    assert policy["model"] == {
        "registry_name": "ae_resnet_unet",
        "category": "bottle",
        "constructor_kwargs": {"device": "cpu"},
    }
    assert "from_run" not in policy
    assert "checkpoint" not in policy
    assert "artifact_quality" not in policy
    assert validate_artifact_policy(policy, manifest_model=policy["model"]) == policy


def test_artifact_policy_builds_canonical_image_threshold_for_score_only_run() -> None:
    from pyimgano.artifacts import validate_artifact_policy
    from pyimgano.services.export_service import _policy_without_external_reconstruction_paths

    policy = _policy_without_external_reconstruction_paths(
        {"threshold": None},
        model_name="detector",
        category=None,
        model_kwargs={},
    )

    assert policy["postprocess"]["image_threshold"] == {
        "threshold": None,
        "score_order": "higher_is_more_anomalous",
    }
    validate_artifact_policy(policy, manifest_model=policy["model"])
