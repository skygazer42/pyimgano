from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class FakeWinCLIPBackend:
    def __init__(self, scales: tuple[int, ...] = (2, 3)) -> None:
        self.scales = scales
        self.text_calls = 0

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        feature = (1.0, 0.0) if self.text_calls % 2 == 0 else (0.0, 1.0)
        self.text_calls += 1
        return torch.tensor([feature] * len(prompts), dtype=torch.float32)

    def encode_images(self, images):
        from pyimgano.models.winclip import _make_patch_masks

        means = torch.tensor(
            [float(np.asarray(image, dtype=np.float32).mean() / 255.0) for image in images]
        )
        image_embeddings = torch.stack((1 - means, means), dim=-1)
        patch_embeddings = image_embeddings[:, None, :].repeat(1, 9, 1)
        windows = [
            image_embeddings[:, None, :].repeat(
                1,
                _make_patch_masks((3, 3), scale).shape[1],
                1,
            )
            for scale in self.scales
        ]
        return image_embeddings, windows, patch_embeddings, (3, 3)


def test_winclip_uses_complete_paper_prompt_ensemble() -> None:
    from pyimgano.models.winclip import PROMPT_TEMPLATES, _create_prompt_ensemble

    normal, anomalous = _create_prompt_ensemble("metal_nut")

    assert len(PROMPT_TEMPLATES) == len(set(PROMPT_TEMPLATES)) == 22
    assert len(normal) == 7 * 22
    assert len(anomalous) == 4 * 22
    assert "a cropped photo of a metal nut." in normal
    assert "a jpeg corrupted photo of the damaged metal nut." in anomalous


def test_winclip_paper_mask_and_harmonic_equation() -> None:
    from pyimgano.models.winclip import _harmonic_aggregation, _make_patch_masks

    masks = _make_patch_masks((3, 3), 2)
    scores = torch.tensor([[1.0, 0.75, 0.5, 0.25]])
    expected = torch.tensor(
        [[[1.0, 0.8571429, 0.75], [0.6666667, 0.48, 0.375], [0.5, 0.3333333, 0.25]]]
    )

    assert torch.equal(
        masks,
        torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]]),
    )
    assert torch.allclose(_harmonic_aggregation(scores, (3, 3), masks), expected)


def test_winclip_class_and_visual_association_equations() -> None:
    from pyimgano.models.winclip import _class_scores, _visual_association_score

    image = torch.tensor([[1.0, 0.0]])
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(
        _class_scores(image, text, temperature=1.0),
        torch.tensor([[0.7310586, 0.2689414]]),
    )

    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    references = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])
    assert torch.allclose(
        _visual_association_score(query, references),
        torch.tensor([[0.1464466, 0.0]]),
        atol=1e-6,
    )


def test_winclip_supplement_non_square_tiling() -> None:
    from pyimgano.models.winclip import _square_tiles

    image = np.zeros((4, 10, 3), dtype=np.uint8)
    tiles = _square_tiles(image)

    assert [(y, x) for _, y, x in tiles] == [(0, 0), (0, 3), (0, 6)]
    assert all(tile.shape == (4, 4, 3) for tile, _, _ in tiles)


def test_winclip_paper_defaults_and_end_to_end_fake_backend() -> None:
    from pyimgano.models.registry import MODEL_REGISTRY
    from pyimgano.models.winclip import WinCLIPDetector

    detector = WinCLIPDetector(
        class_name="bottle",
        k_shot=2,
        backend=FakeWinCLIPBackend(),
        device="cpu",
        random_state=0,
    )
    dark = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    bright = np.full((1, 8, 8, 3), 255, dtype=np.uint8)
    detector.fit(dark)

    scores = detector.decision_function(np.concatenate((dark[:1], bright)))
    anomaly_map = detector.get_anomaly_map(bright[0])

    assert detector.openclip_model_name == "ViT-B-16-plus-240"
    assert detector.openclip_pretrained == "laion400m_e31"
    assert detector.image_size == 240
    assert detector.scales == (2, 3)
    assert detector.temperature == pytest.approx(0.07)
    assert MODEL_REGISTRY.info("vision_winclip").metadata["paper_fidelity"] == "paper-adaptation"
    assert detector.reference_patches_.shape == (2, 9, 2)
    assert [memory.shape for memory in detector.reference_windows_] == [(2, 4, 2), (2, 1, 2)]
    assert scores.shape == (2,)
    assert scores[1] > scores[0]
    assert anomaly_map.shape == (8, 8)
    assert np.isfinite(anomaly_map).all()


def test_winclip_rejects_removed_crop_proxy_backbone() -> None:
    from pyimgano.models.winclip import WinCLIPDetector

    with pytest.raises(ValueError, match="removed crop proxy"):
        WinCLIPDetector(clip_model="ViT-B/32", backend=FakeWinCLIPBackend(), device="cpu")


def test_winclip_few_shot_sampling_is_repeatable() -> None:
    from pyimgano.models.winclip import WinCLIPDetector

    images = np.stack(
        [np.full((8, 8, 3), value, dtype=np.uint8) for value in (0, 32, 64, 96, 128, 160)]
    )
    first = WinCLIPDetector(
        k_shot=3,
        backend=FakeWinCLIPBackend(),
        device="cpu",
        random_state=5,
    )
    second = WinCLIPDetector(
        k_shot=3,
        backend=FakeWinCLIPBackend(),
        device="cpu",
        random_state=5,
    )

    np.random.default_rng(9).random(100)
    first.fit(images)
    np.random.default_rng(27).random(100)
    second.fit(images)

    assert torch.equal(first.reference_patches_, second.reference_patches_)
