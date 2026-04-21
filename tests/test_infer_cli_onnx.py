from __future__ import annotations

from argparse import Namespace


def test_default_onnx_sweep_intra_values_caps_at_16(monkeypatch) -> None:
    import pyimgano.infer_cli_onnx as infer_cli_onnx

    monkeypatch.setattr(infer_cli_onnx.os, "cpu_count", lambda: 64)

    assert infer_cli_onnx.default_onnx_sweep_intra_values() == [1, 2, 4, 8, 16]


def test_extract_onnx_checkpoint_path_for_sweep_checks_known_locations() -> None:
    from pyimgano.infer_cli_onnx import extract_onnx_checkpoint_path_for_sweep

    assert extract_onnx_checkpoint_path_for_sweep({"checkpoint_path": "root.onnx"}) == "root.onnx"
    assert (
        extract_onnx_checkpoint_path_for_sweep(
            {"embedding_kwargs": {"checkpoint_path": "embed.onnx"}}
        )
        == "embed.onnx"
    )
    assert (
        extract_onnx_checkpoint_path_for_sweep(
            {
                "feature_extractor": {
                    "name": "onnx_embed",
                    "kwargs": {"checkpoint_path": "fx.onnx"},
                }
            }
        )
        == "fx.onnx"
    )


def test_extract_session_options_for_sweep_checks_known_locations() -> None:
    from pyimgano.infer_cli_onnx import extract_session_options_for_sweep

    assert extract_session_options_for_sweep({"session_options": {"intra_op_num_threads": 4}}) == {
        "intra_op_num_threads": 4
    }
    assert extract_session_options_for_sweep(
        {"embedding_kwargs": {"session_options": {"graph_optimization_level": "all"}}}
    ) == {"graph_optimization_level": "all"}
    assert extract_session_options_for_sweep(
        {
            "feature_extractor": {
                "name": "onnx_embed",
                "kwargs": {"session_options": {"execution_mode": "parallel"}},
            }
        }
    ) == {"execution_mode": "parallel"}


def test_maybe_apply_onnx_session_options_and_sweep_returns_original_without_options() -> None:
    from pyimgano.infer_cli_onnx import maybe_apply_onnx_session_options_and_sweep

    user_kwargs = {"batch_size": 8}

    resolved = maybe_apply_onnx_session_options_and_sweep(
        args=Namespace(onnx_sweep=False),
        model_name="vision_onnx",
        device="cpu",
        user_kwargs=user_kwargs,
        inputs=["a.png"],
        onnx_session_options_cli=None,
    )

    assert resolved == {"batch_size": 8}
    assert resolved is not user_kwargs


def test_maybe_apply_onnx_session_options_and_sweep_applies_cli_session_options(
    monkeypatch,
) -> None:
    import pyimgano.infer_cli_onnx as infer_cli_onnx

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        infer_cli_onnx,
        "apply_onnx_session_options_shorthand",
        lambda *, model_name, user_kwargs, session_options: calls.append(
            ("apply", (model_name, user_kwargs, session_options))
        )
        or {"resolved": True, "session_options": dict(session_options)},
    )

    resolved = infer_cli_onnx.maybe_apply_onnx_session_options_and_sweep(
        args=Namespace(onnx_sweep=False),
        model_name="vision_onnx",
        device="cpu",
        user_kwargs={"batch_size": 8},
        inputs=["a.png"],
        onnx_session_options_cli={"intra_op_num_threads": 2},
    )

    assert resolved == {"resolved": True, "session_options": {"intra_op_num_threads": 2}}
    assert calls == [
        (
            "apply",
            (
                "vision_onnx",
                {"batch_size": 8},
                {"intra_op_num_threads": 2},
            ),
        )
    ]
