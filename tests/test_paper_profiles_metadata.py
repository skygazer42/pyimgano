from __future__ import annotations


def test_offline_defaults_disclose_separate_paper_profiles() -> None:
    from pyimgano.models.registry import MODEL_REGISTRY

    expected_flags = {
        "vision_ast": "pretrained_backbone",
        "vision_cflow": "pretrained_backbone",
        "vision_fcdd": "pretrained",
        "vision_padim": "pretrained",
        "vision_panda": "pretrained",
        "vision_patchcore": "pretrained",
        "vision_regad": "pretrained",
        "vision_simplenet": "pretrained",
        "vision_softpatch": "pretrained",
        "vision_spade": "pretrained",
        "vision_stfpm": "pretrained_teacher",
    }

    for model_name, flag in expected_flags.items():
        metadata = MODEL_REGISTRY.info(model_name).metadata
        assert str(metadata["default_profile"]).startswith("offline-safe")
        assert metadata["paper_profile"][flag] is True
