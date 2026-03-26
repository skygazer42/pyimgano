import pytest

from pyimgano.services import model_options as model_options_service
from pyimgano.services.model_options import (
    apply_onnx_session_options_shorthand,
    enforce_checkpoint_requirement,
    resolve_model_options,
    resolve_preset_kwargs,
    resolve_requested_model,
)


def test_resolve_model_options_merges_user_preset_and_auto_kwargs():
    out = resolve_model_options(
        model_name="vision_padim",
        preset="industrial-fast",
        user_kwargs={"d_reduced": 16},
        auto_kwargs={"device": "cpu", "contamination": 0.1, "pretrained": False},
        checkpoint_path=None,
    )

    assert out["d_reduced"] == 16
    assert out["contamination"] == pytest.approx(0.1)


def test_resolve_requested_model_accepts_model_preset_alias():
    model_name, preset_model_auto_kwargs, entry = resolve_requested_model(
        "industrial-structural-ecod"
    )

    assert model_name == "vision_feature_pipeline"
    assert isinstance(preset_model_auto_kwargs, dict)
    assert entry.name == "vision_feature_pipeline"


def test_enforce_checkpoint_requirement_allows_trained_checkpoint_path():
    enforce_checkpoint_requirement(
        model_name="vision_patchcore_anomalib",
        model_kwargs={},
        trained_checkpoint_path="/tmp/trained-model.pt",
    )


def test_apply_onnx_session_options_shorthand_targets_top_level_session_options():
    out = apply_onnx_session_options_shorthand(
        model_name="vision_onnx_ecod",
        user_kwargs={},
        session_options={"intra_op_num_threads": 2},
    )

    assert out["session_options"] == {"intra_op_num_threads": 2}


def test_apply_onnx_session_options_shorthand_targets_embedding_kwargs():
    out = apply_onnx_session_options_shorthand(
        model_name="vision_embedding_core",
        user_kwargs={"embedding_extractor": "onnx_embed"},
        session_options={"enable_mem_pattern": False},
    )

    assert out["embedding_kwargs"]["session_options"] == {"enable_mem_pattern": False}


def test_apply_onnx_session_options_shorthand_targets_feature_extractor_kwargs():
    out = apply_onnx_session_options_shorthand(
        model_name="vision_feature_pipeline",
        user_kwargs={"feature_extractor": "onnx_embed"},
        session_options={"graph_optimization_level": "all"},
    )

    assert out["feature_extractor"] == {
        "name": "onnx_embed",
        "kwargs": {"session_options": {"graph_optimization_level": "all"}},
    }


def test_apply_onnx_session_options_shorthand_rejects_non_onnx_route():
    with pytest.raises(ValueError, match="only supported for ONNX-based routes"):
        apply_onnx_session_options_shorthand(
            model_name="vision_padim",
            user_kwargs={},
            session_options={"intra_op_num_threads": 2},
        )


def test_apply_onnx_session_options_shorthand_returns_copy_when_session_options_missing():
    user_kwargs = {"embedding_extractor": "onnx_embed", "batch_size": 8}

    out = apply_onnx_session_options_shorthand(
        model_name="vision_embedding_core",
        user_kwargs=user_kwargs,
        session_options=None,
    )

    assert out == user_kwargs
    assert out is not user_kwargs


def test_apply_onnx_session_options_shorthand_merges_feature_extractor_mapping_kwargs(monkeypatch):
    monkeypatch.setattr(
        model_options_service,
        "get_model_signature_info",
        lambda _model_name: ({"feature_extractor"}, False),
    )

    out = apply_onnx_session_options_shorthand(
        model_name="vision_feature_pipeline",
        user_kwargs={
            "feature_extractor": {
                "name": "onnx_embed",
                "kwargs": {
                    "session_options": {"existing": 1},
                    "normalize": True,
                },
            }
        },
        session_options={"intra_op_num_threads": 2},
    )

    assert out["feature_extractor"] == {
        "name": "onnx_embed",
        "kwargs": {
            "normalize": True,
            "session_options": {
                "existing": 1,
                "intra_op_num_threads": 2,
            },
        },
    }


def test_apply_onnx_session_options_shorthand_rejects_non_onnx_embedding_target(monkeypatch):
    monkeypatch.setattr(
        model_options_service,
        "get_model_signature_info",
        lambda _model_name: ({"embedding_kwargs"}, False),
    )

    with pytest.raises(ValueError, match="requires embedding_extractor='onnx_embed'"):
        apply_onnx_session_options_shorthand(
            model_name="vision_onnx_custom",
            user_kwargs={"embedding_extractor": "torchvision"},
            session_options={"intra_op_num_threads": 2},
        )


def test_apply_onnx_session_options_shorthand_rejects_unsupported_onnx_target(monkeypatch):
    monkeypatch.setattr(
        model_options_service,
        "get_model_signature_info",
        lambda _model_name: ({"feature_extractor"}, False),
    )

    with pytest.raises(ValueError, match="could not be applied"):
        apply_onnx_session_options_shorthand(
            model_name="vision_onnx_custom",
            user_kwargs={"feature_extractor": "torchvision"},
            session_options={"intra_op_num_threads": 2},
        )


def test_apply_onnx_session_options_shorthand_rejects_onnx_route_with_no_supported_target(
    monkeypatch,
):
    monkeypatch.setattr(
        model_options_service,
        "get_model_signature_info",
        lambda _model_name: ({"providers"}, False),
    )

    with pytest.raises(ValueError, match="could not be applied"):
        apply_onnx_session_options_shorthand(
            model_name="vision_onnx_custom",
            user_kwargs={},
            session_options={"intra_op_num_threads": 2},
        )


def test_resolve_requested_model_rejects_unknown_model_or_preset(monkeypatch):
    def raise_missing_model(_name):
        raise KeyError

    monkeypatch.setattr(model_options_service.MODEL_REGISTRY, "info", raise_missing_model)
    monkeypatch.setattr(model_options_service, "resolve_model_preset", lambda _name: None)

    with pytest.raises(ValueError, match="Unknown model or model preset"):
        resolve_requested_model("not-a-real-model")


def test_resolve_requested_model_rejects_preset_pointing_to_unknown_model(monkeypatch):
    class FakePreset:
        model = "missing-model"
        kwargs = {"image_size": 256}

    def fake_info(name):
        raise KeyError(name)

    monkeypatch.setattr(model_options_service.MODEL_REGISTRY, "info", fake_info)
    monkeypatch.setattr(model_options_service, "resolve_model_preset", lambda _name: FakePreset())

    with pytest.raises(ValueError, match="refers to unknown model"):
        resolve_requested_model("broken-preset")


def test_resolve_preset_kwargs_uses_backend_for_patchcore():
    out = resolve_preset_kwargs(
        "industrial-balanced",
        "vision_patchcore",
        default_knn_backend=lambda: "custom-knn",
    )

    assert out == {
        "backbone": "resnet50",
        "coreset_sampling_ratio": 0.05,
        "feature_projection_dim": 512,
        "n_neighbors": 5,
        "knn_backend": "custom-knn",
        "memory_bank_dtype": "float16",
    }


def test_resolve_preset_kwargs_keeps_reverse_distillation_aliases_in_sync():
    a = resolve_preset_kwargs("industrial-accurate", "vision_reverse_distillation")
    b = resolve_preset_kwargs("industrial-accurate", "vision_reverse_dist")

    assert (
        a
        == b
        == {
            "epoch_num": 20,
            "batch_size": 32,
        }
    )


def test_resolve_preset_kwargs_rejects_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve_preset_kwargs("not-a-real-preset", "vision_patchcore")


@pytest.mark.parametrize(
    ("available", "expected_backend"),
    [
        (True, "faiss"),
        (False, "sklearn"),
    ],
)
def test_resolve_preset_kwargs_uses_detected_backend_when_not_overridden(
    monkeypatch,
    available,
    expected_backend,
):
    monkeypatch.setattr(
        model_options_service,
        "optional_import",
        lambda _name: (object(), None) if available else (None, ImportError("missing")),
    )

    out = resolve_preset_kwargs("industrial-fast", "vision_patchcore")

    assert out["knn_backend"] == expected_backend


def test_resolve_preset_kwargs_returns_empty_for_none_preset():
    assert resolve_preset_kwargs(None, "vision_patchcore") == {}


@pytest.mark.parametrize(
    ("preset", "model_name", "expected_subset"),
    [
        ("industrial-fast", "vision_spade", {"backbone": "resnet18", "k_neighbors": 20}),
        ("industrial-fast", "vision_anomalydino", {"knn_backend": "custom-knn", "image_size": 336}),
        (
            "industrial-fast",
            "vision_softpatch",
            {"knn_backend": "custom-knn", "train_patch_outlier_quantile": 0.1},
        ),
        ("industrial-fast", "vision_simplenet", {"backbone": "resnet50", "epochs": 5}),
        ("industrial-fast", "vision_fastflow", {"epoch_num": 5, "n_flow_steps": 4}),
        ("industrial-fast", "vision_cflow", {"epochs": 10, "n_flows": 2}),
        ("industrial-fast", "vision_stfpm", {"epochs": 10, "batch_size": 32}),
        ("industrial-fast", "vision_reverse_distillation", {"epoch_num": 5, "batch_size": 32}),
        ("industrial-fast", "vision_draem", {"image_size": 256, "epochs": 20}),
        ("industrial-fast", "not-covered-fast", {}),
        ("industrial-balanced", "vision_padim", {"d_reduced": 64, "image_size": 224}),
        ("industrial-balanced", "vision_spade", {"backbone": "resnet50", "gaussian_sigma": 4.0}),
        (
            "industrial-balanced",
            "vision_anomalydino",
            {"knn_backend": "custom-knn", "image_size": 448},
        ),
        (
            "industrial-balanced",
            "vision_softpatch",
            {"knn_backend": "custom-knn", "train_patch_outlier_quantile": 0.1},
        ),
        ("industrial-balanced", "vision_simplenet", {"backbone": "resnet50", "epochs": 10}),
        ("industrial-balanced", "vision_fastflow", {"epoch_num": 10, "n_flow_steps": 6}),
        ("industrial-balanced", "vision_cflow", {"epochs": 15, "n_flows": 4}),
        ("industrial-balanced", "vision_stfpm", {"epochs": 50, "batch_size": 32}),
        ("industrial-balanced", "vision_reverse_distillation", {"epoch_num": 10, "batch_size": 32}),
        ("industrial-balanced", "vision_draem", {"image_size": 256, "epochs": 50}),
        ("industrial-balanced", "not-covered-balanced", {}),
        ("industrial-accurate", "vision_padim", {"d_reduced": 128, "image_size": 224}),
        (
            "industrial-accurate",
            "vision_spade",
            {"backbone": "wide_resnet50", "gaussian_sigma": 4.0},
        ),
        (
            "industrial-accurate",
            "vision_anomalydino",
            {"knn_backend": "custom-knn", "image_size": 518},
        ),
        (
            "industrial-accurate",
            "vision_softpatch",
            {"knn_backend": "custom-knn", "train_patch_outlier_quantile": 0.05},
        ),
        ("industrial-accurate", "vision_simplenet", {"backbone": "wide_resnet50", "epochs": 20}),
        ("industrial-accurate", "vision_fastflow", {"epoch_num": 20, "n_flow_steps": 8}),
        ("industrial-accurate", "vision_cflow", {"epochs": 50, "n_flows": 8}),
        ("industrial-accurate", "vision_stfpm", {"epochs": 100, "batch_size": 32}),
        ("industrial-accurate", "vision_draem", {"image_size": 256, "epochs": 100}),
        ("industrial-accurate", "not-covered-accurate", {}),
    ],
)
def test_resolve_preset_kwargs_covers_supported_model_branches(preset, model_name, expected_subset):
    out = resolve_preset_kwargs(
        preset,
        model_name,
        default_knn_backend=lambda: "custom-knn",
    )

    assert out.items() >= expected_subset.items()


def test_enforce_checkpoint_requirement_includes_extra_guidance_in_error():
    with pytest.raises(ValueError, match="Run training first"):
        enforce_checkpoint_requirement(
            model_name="vision_patchcore_anomalib",
            model_kwargs={},
            trained_checkpoint_path=None,
            extra_guidance="Run training first.",
        )


def test_enforce_checkpoint_requirement_skips_non_checkpoint_models():
    enforce_checkpoint_requirement(
        model_name="vision_padim",
        model_kwargs={},
        trained_checkpoint_path=None,
    )


def test_enforce_checkpoint_requirement_raises_base_message_without_extra_guidance():
    with pytest.raises(ValueError, match="requires a checkpoint"):
        enforce_checkpoint_requirement(
            model_name="vision_patchcore_anomalib",
            model_kwargs={},
            trained_checkpoint_path=None,
            extra_guidance=None,
        )


def test_resolve_preset_kwargs_accurate_patchcore_defaults():
    out = resolve_preset_kwargs(
        "industrial-accurate",
        "vision_patchcore",
        default_knn_backend=lambda: "custom-knn",
    )

    assert out == {
        "backbone": "wide_resnet50",
        "coreset_sampling_ratio": 0.1,
        "n_neighbors": 9,
        "knn_backend": "custom-knn",
    }
