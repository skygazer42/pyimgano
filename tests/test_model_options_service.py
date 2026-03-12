from pyimgano.services.model_options import (
    apply_onnx_session_options_shorthand,
    enforce_checkpoint_requirement,
    resolve_model_options,
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
    assert out["contamination"] == 0.1


def test_resolve_requested_model_accepts_model_preset_alias():
    model_name, preset_model_auto_kwargs, entry = resolve_requested_model("industrial-structural-ecod")

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
