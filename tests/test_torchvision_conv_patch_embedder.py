from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_torchvision_conv_patch_embedder_smoke_offline_default(monkeypatch) -> None:
    torch = __import__("torch")

    # Guardrail: even if torch.hub download helpers are present, the default
    # embedder config (pretrained=False) must not trigger them.
    def _fail(*args, **kwargs):  # noqa: ANN001, ANN201 - test helper
        del args, kwargs
        raise AssertionError("unexpected weight download attempt")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _fail, raising=True)
    monkeypatch.setattr(torch.hub, "load", _fail, raising=True)

    from pyimgano.features.torchvision_conv_patch_embedder import TorchvisionConvPatchEmbedder

    rng = np.random.default_rng(0)
    img = (rng.integers(0, 255, size=(32, 32, 3))).astype(np.uint8)

    embedder = TorchvisionConvPatchEmbedder(
        backbone="resnet18",
        node="layer3",
        pretrained=False,
        device="cpu",
        image_size=64,
        normalize=True,
    )

    patches, grid_shape, orig_size = embedder.embed(img)
    patches = np.asarray(patches, dtype=np.float32)

    gh, gw = int(grid_shape[0]), int(grid_shape[1])
    assert patches.ndim == 2
    assert patches.shape[0] == gh * gw
    assert patches.shape[1] >= 4
    assert orig_size == (32, 32)

    assert np.all(np.isfinite(patches))
    norms = np.linalg.norm(patches, axis=1)
    assert float(np.median(norms)) == pytest.approx(1.0, abs=1e-3)
