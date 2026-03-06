from __future__ import annotations

from pathlib import Path


def test_vision_onnx_wrappers_accept_session_options(tmp_path: Path) -> None:
    """Industrial ONNX wrappers should expose ORT SessionOptions knobs without requiring onnxruntime."""

    ckpt = tmp_path / "model.onnx"
    ckpt.write_text("fake", encoding="utf-8")

    from pyimgano.models import create_model

    det = create_model(
        "vision_onnx_ecod",
        contamination=0.2,
        checkpoint_path=str(ckpt),
        device="cpu",
        batch_size=1,
        image_size=32,
        session_options={
            "intra_op_num_threads": 2,
            "inter_op_num_threads": 1,
            "execution_mode": "sequential",
            "graph_optimization_level": "all",
            "enable_mem_pattern": False,
        },
    )

    from pyimgano.features.onnx_embed import ONNXEmbedExtractor

    extractor = det._base_feature_extractor  # type: ignore[attr-defined]
    assert isinstance(extractor, ONNXEmbedExtractor)
    assert extractor.session_options is not None
    assert int(dict(extractor.session_options)["intra_op_num_threads"]) == 2
