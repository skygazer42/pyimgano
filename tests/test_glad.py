from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_glad_released_architecture_presets_and_metadata() -> None:
    from pyimgano.models.glad import (
        PAPER_DINO_LAYERS,
        PAPER_DINO_MODEL,
        PAPER_GUIDANCE_SCALE,
        PAPER_INPUT_RESOLUTION,
        PAPER_PROMPT,
        VisionGLAD,
        _uses_fine_tuned_dino,
        get_glad_preset,
    )
    from pyimgano.models.registry import MODEL_REGISTRY

    assert PAPER_DINO_MODEL == "dino_vitb8"
    assert PAPER_DINO_LAYERS == (3, 6, 9, 12)
    assert PAPER_INPUT_RESOLUTION == 512
    assert PAPER_PROMPT == "a photo of sks"
    assert PAPER_GUIDANCE_SCALE == 9
    assert not _uses_fine_tuned_dino("MVTec-AD")
    assert _uses_fine_tuned_dino("VisA")
    assert _uses_fine_tuned_dino("PCB-Bank")
    assert get_glad_preset("MVTec-AD", "transistor") == (
        850,
        0.5,
        0,
        350,
        2500,
        0,
        512,
        25,
    )
    assert get_glad_preset("VisA", "candle") == (
        450,
        0.45,
        7,
        200,
        4000,
        1,
        256,
        15,
    )
    metadata = MODEL_REGISTRY.info("vision_glad").metadata
    assert metadata["paper_fidelity"] == "paper-adaptation"
    assert metadata["requires_checkpoint"] is True
    assert "checkpoint_path" in inspect.signature(VisionGLAD.__init__).parameters

    # Construction is offline-safe; model and checkpoint loading are deferred.
    VisionGLAD(device="cpu")


def test_glad_feature_map_matches_author_nearest_cosine_formula() -> None:
    from pyimgano.models.glad import _feature_anomaly_map

    input_tokens = [
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    ]
    reconstruction_tokens = [
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])
    ]

    forward = _feature_anomaly_map(
        input_tokens, reconstruction_tokens, output_size=2, reverse=False
    )
    bidirectional = _feature_anomaly_map(
        input_tokens, reconstruction_tokens, output_size=2, reverse=True
    )

    expected_forward = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    expected_backward = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    assert torch.allclose(forward, expected_forward)
    assert torch.allclose(bidirectional, expected_forward + expected_backward)


def test_glad_adaptive_step_uses_saff_and_records_selected_step(monkeypatch) -> None:
    import pyimgano.models.glad as glad

    class _VAE:
        config = SimpleNamespace(scaling_factor=1.0)

        def decode(self, latents, return_dict=False):  # noqa: ANN001, ANN201
            del return_dict
            return (latents,)

    token = torch.tensor([[[0.0], [1.0], [1.0], [1.0], [1.0]]])
    calls = iter([[token], [-token]])
    monkeypatch.setattr(glad, "_dino_patch_tokens", lambda model, images: next(calls))

    alphas = torch.linspace(0.99, 0.5, 10)
    active = torch.tensor([True])
    chosen = torch.zeros(1, dtype=torch.long)
    thresholds = torch.zeros(1)
    previous, active = glad._adaptive_ddim_step(
        model_output=torch.zeros((1, 1, 2, 2)),
        timestep=5,
        sample=torch.ones((1, 1, 2, 2)),
        image_latents=torch.zeros((1, 1, 2, 2)),
        noise=torch.zeros((1, 1, 2, 2)),
        alphas_cumprod=alphas,
        final_alpha_cumprod=torch.tensor(1.0),
        step_ratio=2,
        vae=_VAE(),
        dino_model=object(),
        active=active,
        chosen_steps=chosen,
        thresholds=thresholds,
        input_threshold=0.5,
        min_step=0,
    )

    assert not bool(active[0])
    assert int(chosen[0]) == 5
    assert float(thresholds[0]) == pytest.approx(2.0)
    assert torch.isfinite(previous).all()


def test_glad_wrapper_calibrates_and_returns_maps() -> None:
    from pyimgano.models.glad import VisionGLAD

    class _Backend:
        def score_items(self, items, *, seed):  # noqa: ANN001, ANN201
            assert seed == 7
            values = np.asarray([np.asarray(item).mean() for item in items], dtype=np.float32)
            return values, np.repeat(values[:, None, None], 4, axis=1).reshape(-1, 2, 2), np.full(len(items), 350)

    images = np.stack(
        (np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8))
    )
    detector = VisionGLAD(backend=_Backend(), device="cpu", random_state=7)
    detector.fit(images)

    assert detector.predict(images).tolist() == [0.0, 1.0]
    assert detector.predict_anomaly_map(images).shape == (2, 2, 2)
    assert detector.get_anomaly_map(images[0]).shape == (2, 2)


def test_glad_visa_requires_the_separate_frozen_ads_dino() -> None:
    from pyimgano.models.glad import TorchGLADBackend, get_glad_preset

    pipeline = SimpleNamespace(
        vae=object(),
        unet=object(),
        text_encoder=object(),
        tokenizer=object(),
        scheduler=object(),
    )
    backend = TorchGLADBackend(
        preset=get_glad_preset("visa", "candle"),
        class_name="candle",
        dataset="visa",
        pipeline=pipeline,
        dino_model=torch.nn.Identity(),
        device="cpu",
    )

    with pytest.raises(ValueError, match="frozen pretrained DINO used by ADS"):
        backend._ensure_loaded()
