from __future__ import annotations

from pathlib import Path


def test_onnx_embed_cache_fingerprint_includes_session_options(tmp_path: Path) -> None:
    from pyimgano.features.onnx_embed import ONNXEmbedExtractor

    ckpt = tmp_path / "model.onnx"
    ckpt.write_text("fake", encoding="utf-8")

    cache_dir = tmp_path / "cache"

    e1 = ONNXEmbedExtractor(
        checkpoint_path=str(ckpt),
        device="cpu",
        batch_size=1,
        image_size=224,
        cache_dir=str(cache_dir),
        session_options={"intra_op_num_threads": 1},
    )
    e2 = ONNXEmbedExtractor(
        checkpoint_path=str(ckpt),
        device="cpu",
        batch_size=1,
        image_size=224,
        cache_dir=str(cache_dir),
        session_options={"intra_op_num_threads": 2},
    )

    assert e1._cache is not None
    assert e2._cache is not None
    assert e1._cache.extractor_fingerprint != e2._cache.extractor_fingerprint
