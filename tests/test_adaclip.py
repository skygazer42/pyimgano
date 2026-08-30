from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional


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
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=False)
        self.class_embedding = torch.nn.Parameter(torch.randn(8))
        self.positional_embedding = torch.nn.Parameter(torch.randn(5, 8) * 0.01)
        self.patch_dropout = torch.nn.Identity()
        self.ln_pre = torch.nn.LayerNorm(8)
        self.transformer = _Transformer(4, 8)
        self.ln_post = torch.nn.LayerNorm(8)
        self.proj = torch.nn.Parameter(torch.randn(8, 6) * 0.1)
        self.input_patchnorm = False
        self.global_average_pool = False
        self.attn_pool = None


class _TinyCLIP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _Visual()
        self.token_embedding = torch.nn.Embedding(256, 8)
        self.positional_embedding = torch.nn.Parameter(torch.randn(16, 8) * 0.01)
        self.transformer = _Transformer(4, 8)
        self.ln_final = torch.nn.LayerNorm(8)
        self.text_projection = torch.nn.Parameter(torch.randn(8, 6) * 0.1)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(100.0)))
        mask = torch.full((16, 16), float("-inf")).triu_(1)
        self.register_buffer("attn_mask", mask, persistent=False)


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

    def score_image(self, image: np.ndarray, class_name: str):  # noqa: ANN201
        assert class_name == "bottle"
        score = float(image.mean() / 255.0)
        return score, np.full(image.shape[:2], score, dtype=np.float32)


def test_adaclip_paper_defaults_prompts_and_metadata() -> None:
    from pyimgano.models.adaclip import (
        PAPER_IMAGE_SIZE,
        PAPER_K_CLUSTERS,
        PAPER_MODEL,
        PAPER_OUTPUT_LAYERS,
        PAPER_PROMPT_DEPTH,
        PAPER_PROMPT_LENGTH,
        _create_text_ensemble,
    )
    from pyimgano.models.registry import MODEL_REGISTRY

    normal, anomaly = _create_text_ensemble("metal_nut")

    assert PAPER_MODEL == "ViT-L-14-336"
    assert PAPER_IMAGE_SIZE == 518
    assert PAPER_OUTPUT_LAYERS == (6, 12, 18, 24)
    assert PAPER_PROMPT_DEPTH == 4
    assert PAPER_PROMPT_LENGTH == 5
    assert PAPER_K_CLUSTERS == 20
    assert len(normal) == 7 * 4
    assert len(anomaly) == 5 * 4
    assert "a bad photo of a flawless metal nut." in normal
    assert "a cropped photo of the broken metal nut." in anomaly
    metadata = MODEL_REGISTRY.info("vision_adaclip").metadata
    assert metadata["paper_fidelity"] == "paper-adaptation"
    assert metadata["requires_checkpoint"] is True


def test_adaclip_block_runner_supports_openclip_batch_first_attention() -> None:
    from pyimgano.models.adaclip import _run_block

    class BatchFirstBlock(_ResidualBlock):
        def __init__(self) -> None:
            super().__init__(8)
            self.attn = torch.nn.MultiheadAttention(8, 2, batch_first=True)

    torch.manual_seed(3)
    block = BatchFirstBlock().eval()
    sequence_first = torch.randn(5, 2, 8)
    expected = block(sequence_first.permute(1, 0, 2)).permute(1, 0, 2)

    torch.testing.assert_close(_run_block(block, sequence_first), expected)


def test_adaclip_block_runner_detects_current_openclip_custom_attention() -> None:
    from pyimgano.models.adaclip import _run_block

    class CurrentBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = SimpleNamespace(use_sdpa=True)

        def forward(self, tokens, attn_mask=None):  # noqa: ANN001, ANN201
            del attn_mask
            assert tuple(tokens.shape[:2]) == (2, 5)
            return tokens + 2.0

    sequence_first = torch.randn(5, 2, 8)
    torch.testing.assert_close(
        _run_block(CurrentBlock(), sequence_first),
        sequence_first + 2.0,
    )


def test_adaclip_network_has_released_prompt_structure_and_returns_both_outputs() -> None:
    from pyimgano.models.adaclip import AdaCLIPNetwork

    torch.manual_seed(7)
    network = AdaCLIPNetwork(
        _TinyCLIP(),
        image_size=8,
        output_layers=(1, 2, 3, 4),
        prompting_depth=4,
        prompting_length=5,
        k_clusters=1,
        tokenizer=_tokenize,
    )
    anomaly_map, anomaly_score = network(torch.rand(1, 3, 8, 8), "bottle")
    trainable = network.trainable_state_dict()

    assert anomaly_map.shape == (1, 8, 8)
    assert anomaly_score.shape == (1,)
    assert torch.isfinite(anomaly_map).all()
    assert torch.isfinite(anomaly_score).all()
    assert len(trainable) == 38
    assert "text_prompter.static_prompts.3" in trainable
    assert "visual_prompter.static_prompts.3" in trainable
    assert "patch_token_layer.head.3.weight" in trainable
    assert "dynamic_visual_prompt_generator.head.4.weight" in trainable
    assert "dynamic_text_prompt_generator.head.4.weight" in trainable
    assert not any(name.startswith("freeze_clip.") for name in trainable)


def test_adaclip_backend_loads_author_checkpoint_key_layout(tmp_path) -> None:
    from pyimgano.models.adaclip import AdaCLIPNetwork, OpenCLIPAdaCLIPBackend

    source = AdaCLIPNetwork(
        _TinyCLIP(),
        image_size=8,
        output_layers=(1, 2, 3, 4),
        prompting_depth=4,
        prompting_length=2,
        k_clusters=1,
        tokenizer=_tokenize,
    )
    checkpoint = tmp_path / "adaclip.pth"
    torch.save(
        {f"clip_model.{name}": value for name, value in source.trainable_state_dict().items()},
        checkpoint,
    )
    backend = OpenCLIPAdaCLIPBackend(
        checkpoint_path=checkpoint,
        model=_TinyCLIP(),
        preprocess=_preprocess,
        tokenizer=_tokenize,
        image_size=8,
        output_layers=(1, 2, 3, 4),
        prompting_depth=4,
        prompting_length=2,
        k_clusters=1,
        device="cpu",
    ).initialize()
    score, anomaly_map = backend.score_image(
        np.full((11, 7, 3), 128, dtype=np.uint8),
        "bottle",
    )

    assert np.isfinite(score)
    assert anomaly_map.shape == (8, 8)
    assert np.isfinite(anomaly_map).all()
    assert backend.network is not None
    for name, expected in source.trainable_state_dict().items():
        torch.testing.assert_close(backend.network.state_dict()[name], expected)


def test_adaclip_detector_calibrates_threshold_without_retraining_proxy() -> None:
    from pyimgano.models.adaclip import VisionAdaCLIP

    detector = VisionAdaCLIP(
        class_name="bottle",
        backend=_FakeBackend(),
        gaussian_sigma=0,
    )
    dark = np.zeros((8, 8, 3), dtype=np.uint8)
    bright = np.full((8, 8, 3), 255, dtype=np.uint8)
    detector.fit([dark])

    assert detector.decision_function([dark, bright]).tolist() == [0.0, 1.0]
    assert detector.predict([dark, bright]).tolist() == [0, 1]
    assert detector.get_anomaly_map(bright).shape == (8, 8)


def test_adaclip_requires_auxiliary_trained_checkpoint() -> None:
    from pyimgano.models.adaclip import OpenCLIPAdaCLIPBackend

    with pytest.raises(ValueError, match="checkpoint_path is required"):
        OpenCLIPAdaCLIPBackend(checkpoint_path=None, device="cpu").initialize()
