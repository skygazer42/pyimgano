from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _ResidualBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, 2)
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, width * 2),
            torch.nn.GELU(),
            torch.nn.Linear(width * 2, width),
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:  # noqa: ANN001
        normalized = self.ln_1(x)
        x = (
            x
            + self.attn(
                normalized,
                normalized,
                normalized,
                attn_mask=attn_mask,
                need_weights=False,
            )[0]
        )
        return x + self.mlp(self.ln_2(x))


class _Transformer(torch.nn.Module):
    def __init__(self, blocks: int, width: int) -> None:
        super().__init__()
        self.resblocks = torch.nn.ModuleList([_ResidualBlock(width) for _ in range(blocks)])


class _Visual(torch.nn.Module):
    def __init__(self, blocks: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=False)
        self.class_embedding = torch.nn.Parameter(torch.randn(8))
        self.positional_embedding = torch.nn.Parameter(torch.randn(26, 8) * 0.01)
        self.patch_dropout = torch.nn.Identity()
        self.ln_pre = torch.nn.LayerNorm(8)
        self.transformer = _Transformer(blocks, 8)
        self.ln_post = torch.nn.LayerNorm(8)
        self.input_patchnorm = False


class _TinyCLIP(torch.nn.Module):
    def __init__(self, *, visual_blocks: int = 4, text_blocks: int = 4) -> None:
        super().__init__()
        self.visual = _Visual(visual_blocks)
        self.token_embedding = torch.nn.Embedding(256, 8)
        self.positional_embedding = torch.nn.Parameter(torch.randn(16, 8) * 0.01)
        self.transformer = _Transformer(text_blocks, 8)
        self.ln_final = torch.nn.LayerNorm(8)
        self.text_projection = torch.nn.Parameter(torch.randn(8, 6) * 0.1)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(100.0)))
        self.register_buffer(
            "attn_mask",
            torch.full((16, 16), float("-inf")).triu_(1),
            persistent=False,
        )


def _tokenize(prompts: list[str]) -> torch.Tensor:
    rows = []
    for prompt in prompts:
        words = prompt.replace(".", " .").split()
        ids = [1] + [2 + sum(map(ord, word)) % 200 for word in words] + [255]
        assert len(ids) <= 16
        rows.append(ids + [0] * (16 - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


def _preprocess(image) -> torch.Tensor:  # noqa: ANN001
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class _FakeBackend:
    def initialize(self):  # noqa: ANN201
        return self

    def score_image(
        self,
        image: np.ndarray,
        class_description: str,
        domain: str,
    ):  # noqa: ANN201
        assert class_description == "dark bottle"
        assert domain == "industrial"
        score = float(image.mean() / 255.0)
        return score, np.full((5, 5), score, dtype=np.float32)


def _small_network():  # noqa: ANN201
    from pyimgano.models.aaclip import AAClipNetwork

    return AAClipNetwork(
        _TinyCLIP(),
        tokenizer=_tokenize,
        image_size=20,
        output_layers=(1, 2, 3, 4),
        text_adapt_until=2,
        image_adapt_until=2,
    )


def test_aaclip_paper_defaults_prompts_and_metadata() -> None:
    from pyimgano.models.aaclip import (
        PAPER_IMAGE_ADAPT_UNTIL,
        PAPER_IMAGE_SIZE,
        PAPER_MODEL,
        PAPER_OUTPUT_LAYERS,
        PAPER_TEXT_ADAPT_UNTIL,
        _create_text_ensemble,
    )
    from pyimgano.models.registry import MODEL_REGISTRY

    normal, anomaly = _create_text_ensemble("metal_nut")

    assert PAPER_MODEL == "ViT-L-14-336"
    assert PAPER_IMAGE_SIZE == 518
    assert PAPER_OUTPUT_LAYERS == (6, 12, 18, 24)
    assert PAPER_TEXT_ADAPT_UNTIL == 3
    assert PAPER_IMAGE_ADAPT_UNTIL == 6
    assert len(normal) == 3 * 2
    assert len(anomaly) == 5 * 2
    assert "a photo of a metal nut which has four notched edges." in normal
    assert "a photo of a broken metal nut which has four notched edges." in anomaly
    metadata = MODEL_REGISTRY.info("vision_aaclip").metadata
    assert metadata["paper_fidelity"] == "paper-adaptation"
    assert metadata["requires_checkpoint"] is True


def test_aaclip_network_matches_released_adapter_layout_and_outputs() -> None:
    torch.manual_seed(7)
    network = _small_network().eval()
    anchors = network.encode_text_anchors("dark bottle")
    anomaly_map, anomaly_score = network(torch.rand(1, 3, 20, 20), anchors, domain="industrial")
    state = network.state_dict()
    trainable = {name for name, parameter in network.named_parameters() if parameter.requires_grad}

    assert anchors.shape == (6, 2)
    assert anomaly_map.shape == (1, 20, 20)
    assert anomaly_score.shape == (1,)
    assert torch.isfinite(anomaly_map).all()
    assert torch.isfinite(anomaly_score).all()
    assert "image_adapter.layer_adapters.0.fc.0.weight" in state
    assert "image_adapter.seg_proj.3.fc.weight" in state
    assert "image_adapter.det_proj.fc.weight" in state
    assert "text_adapter.2.fc.0.weight" in state
    assert not any(name.startswith("clip_model.") for name in trainable)


def test_aaclip_backend_loads_author_checkpoint_directory(tmp_path) -> None:
    from pyimgano.models.aaclip import OpenCLIPAAClipBackend

    source = _small_network()
    torch.save(
        {"epoch": 5, "text_adapter": source.text_adapter.state_dict()},
        tmp_path / "text_adapter.pth",
    )
    torch.save(
        {"epoch": 20, "image_adapter": source.image_adapter.state_dict()},
        tmp_path / "image_adapter.pth",
    )
    backend = OpenCLIPAAClipBackend(
        checkpoint_path=tmp_path,
        model=_TinyCLIP(),
        preprocess=_preprocess,
        tokenizer=_tokenize,
        image_size=20,
        output_layers=(1, 2, 3, 4),
        text_adapt_until=2,
        image_adapt_until=2,
        device="cpu",
    ).initialize()
    score, anomaly_map = backend.score_image(
        np.full((13, 9, 3), 128, dtype=np.uint8),
        "dark bottle",
        "industrial",
    )

    assert np.isfinite(score)
    assert anomaly_map.shape == (20, 20)
    assert np.isfinite(anomaly_map).all()
    assert backend.network is not None
    for name, expected in source.text_adapter.state_dict().items():
        torch.testing.assert_close(backend.network.text_adapter.state_dict()[name], expected)
    for name, expected in source.image_adapter.state_dict().items():
        torch.testing.assert_close(backend.network.image_adapter.state_dict()[name], expected)


def test_aaclip_detector_calibrates_without_replacing_paper_training() -> None:
    from pyimgano.models.aaclip import VisionAAClip

    detector = VisionAAClip(class_name="bottle", backend=_FakeBackend())
    dark = np.zeros((8, 10, 3), dtype=np.uint8)
    bright = np.full((8, 10, 3), 255, dtype=np.uint8)
    detector.fit([dark])

    assert detector.decision_function([dark, bright]).tolist() == [0.0, 1.0]
    assert detector.predict([dark, bright]).tolist() == [0, 1]
    assert detector.get_anomaly_map(bright).shape == (8, 10)


def test_aaclip_requires_two_stage_author_checkpoints() -> None:
    from pyimgano.models.aaclip import OpenCLIPAAClipBackend

    with pytest.raises(ValueError, match="checkpoint_path is required"):
        OpenCLIPAAClipBackend(checkpoint_path=None, device="cpu").initialize()
