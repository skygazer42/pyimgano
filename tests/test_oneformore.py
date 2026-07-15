from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_oneformore_paper_contract_and_metadata() -> None:
    from pyimgano.models import MODEL_REGISTRY
    from pyimgano.models.oneformore import (
        OFFICIAL_COMMIT,
        PAPER_AMN_DEPTH,
        PAPER_AMN_NEIGHBOR_SIZE,
        PAPER_CHANNEL_MULT,
        PAPER_DDIM_STEPS,
        PAPER_FEATURE_LAYERS,
        PAPER_IMAGE_SIZE,
        PAPER_MODEL_CHANNELS,
        VisionOneForMore,
    )

    metadata = MODEL_REGISTRY.info("vision_oneformore").metadata
    assert metadata["paper_fidelity"] == "external-backend"
    assert metadata["requires_checkpoint"] is True
    assert metadata["supports_pixel_map"] is True
    assert metadata["official_repository"] == "https://github.com/FuNz-0/One-for-More"
    assert OFFICIAL_COMMIT == "f4eb78841dbfa5612e008570b690072b19a3d9b3"
    assert PAPER_IMAGE_SIZE == 256
    assert PAPER_MODEL_CHANNELS == 320
    assert PAPER_CHANNEL_MULT == (1, 2, 4, 4)
    assert PAPER_AMN_DEPTH == 8
    assert PAPER_AMN_NEIGHBOR_SIZE == (7, 7)
    assert PAPER_DDIM_STEPS == 10
    assert PAPER_FEATURE_LAYERS == {"mvtec": (1, 2, 3), "visa": (0, 2, 4)}

    detector = VisionOneForMore(device="cpu")
    assert detector.checkpoint_path is None


def test_author_backend_runs_released_preprocess_sampling_and_score() -> None:
    from pyimgano.models.oneformore import AuthorOneForMoreBackend

    class _Features(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            return [x[:, :1] * float(index + 1) for index in range(5)]

    class _AuthorModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pretrained_resnet50 = _Features()
            self.layers_ = [1, 2, 3]
            self.distance = "eucl"
            self.calls: list[dict[str, object]] = []

        def log_images_test(self, batch, **kwargs):  # noqa: ANN001, ANN201
            self.calls.append({"batch": batch, **kwargs})
            count = int(batch["jpg"].shape[0])
            return {
                "reconstruction": torch.zeros((count, 3, 8, 8)),
                "samples": torch.ones((count, 3, 8, 8)),
            }

    model = _AuthorModel()
    backend = AuthorOneForMoreBackend(
        repository_path=None,
        checkpoint_path=None,
        dataset="mvtec",
        device="cpu",
        batch_size=2,
        model=model,
    )
    scores, maps = backend.score_items(
        [np.zeros((12, 10, 3), dtype=np.uint8), np.full((9, 11, 3), 255, dtype=np.uint8)],
        seed=1,
    )

    assert scores == pytest.approx([9.0, 9.0])
    assert maps.shape == (2, 256, 256)
    assert np.allclose(maps, 9.0)
    call = model.calls[0]
    assert call["ddim_steps"] == 10
    assert call["ddim_eta"] == 0.0
    assert call["unconditional_guidance_scale"] == 9.0
    batch = call["batch"]
    assert batch["jpg"].shape == (2, 3, 256, 256)
    assert batch["hint"] is batch["jpg"]
    assert batch["txt"] == ["", ""]
    assert float(batch["jpg"][0, 0, 0, 0]) == pytest.approx(-0.485 / 0.229)


def test_oneformore_wrapper_calibrates_and_returns_pixel_maps(capsys) -> None:
    from pyimgano.models.oneformore import VisionOneForMore

    class _Backend:
        batch_size = 9

        def score_items(self, items, *, seed):  # noqa: ANN001, ANN201
            assert seed == 7
            maps = np.stack(
                [np.full((4, 4), float(index + 1), dtype=np.float32) for index in range(len(items))]
            )
            return maps.reshape(len(items), -1).max(axis=1), maps

    backend = _Backend()
    detector = VisionOneForMore(
        backend=backend,
        batch_size=2,
        device="cpu",
        random_state=7,
        contamination=0.25,
    )
    train = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    detector.fit(train)

    assert detector.decision_scores_ == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert detector.decision_function(train, batch_size=1) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert detector.predict_anomaly_map(train).shape == (4, 4, 4)
    assert backend.batch_size == 9
    assert capsys.readouterr().out == ""


def test_oneformore_rejects_missing_official_runtime_at_first_use() -> None:
    from pyimgano.models.oneformore import VisionOneForMore

    detector = VisionOneForMore(device="cpu")
    with pytest.raises(ValueError, match="repository_path.*checkpoint_path"):
        detector.decision_function(np.zeros((1, 8, 8, 3), dtype=np.uint8))
