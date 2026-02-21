import numpy as np
import pytest


def test_openclip_vit_patch_embedder_shapes(tmp_path):
    open_clip = pytest.importorskip("open_clip")

    PIL = pytest.importorskip("PIL")  # noqa: N806 - used for importorskip guard
    from PIL import Image  # noqa: E402

    from pyimgano.models.openclip_backend import OpenCLIPViTPatchEmbedder  # noqa: E402

    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    image_path = tmp_path / "img.png"
    image.save(image_path)

    embedder = OpenCLIPViTPatchEmbedder(
        open_clip_module=open_clip,
        model_name="ViT-B-32",
        pretrained=None,  # avoid weight downloads in CI
        device="cpu",
    )

    patch_embeddings, grid_shape, original_size = embedder.embed(str(image_path))
    assert patch_embeddings.ndim == 2
    assert patch_embeddings.shape[1] > 0

    grid_h, grid_w = grid_shape
    assert patch_embeddings.shape[0] == int(grid_h) * int(grid_w)
    assert original_size == (32, 32)
    assert np.isfinite(patch_embeddings).all()

