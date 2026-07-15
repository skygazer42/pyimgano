from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_inctrl_released_architecture_and_prompts() -> None:
    from pyimgano.models.inctrl import (
        PAPER_FEATURE_LAYERS,
        PAPER_GLOBAL_DIM,
        PAPER_IMAGE_SIZE,
        PAPER_PATCH_GRID,
        PAPER_PRETRAINED,
        PAPER_SHOTS,
        PAPER_TRAIN_BATCH_SIZE,
        PAPER_TRAIN_EPOCHS,
        PAPER_TRAIN_LEARNING_RATE,
        InCTRLHeads,
        VisionInCTRL,
        _prompt_ensemble,
    )
    from pyimgano.models.registry import MODEL_REGISTRY

    heads = InCTRLHeads()
    normal, anomaly = _prompt_ensemble("metal_nut")
    natural_normal, natural_anomaly = _prompt_ensemble("airplane")

    assert PAPER_PRETRAINED == "laion400m_e32"
    assert PAPER_IMAGE_SIZE == 240
    assert PAPER_FEATURE_LAYERS == (7, 9, 11)
    assert PAPER_PATCH_GRID == (15, 15)
    assert PAPER_SHOTS == (2, 4, 8)
    assert (PAPER_TRAIN_EPOCHS, PAPER_TRAIN_BATCH_SIZE) == (10, 48)
    assert PAPER_TRAIN_LEARNING_RATE == pytest.approx(1e-3)
    assert tuple(heads.adapter.fc[0].weight.shape) == (160, PAPER_GLOBAL_DIM)
    assert tuple(heads.adapter.fc[2].weight.shape) == (PAPER_GLOBAL_DIM, 160)
    assert tuple(heads.diff_head.projection1.weight.shape) == (128, 225)
    assert tuple(heads.diff_head_ref.projection1.weight.shape) == (128, PAPER_GLOBAL_DIM)
    assert "diff_head.bn1.weight" in heads.state_dict()
    assert (len(normal), len(anomaly)) == (154, 88)
    assert natural_normal == ["a photo of airplane for anomaly detection."]
    assert natural_anomaly == ["a photo without airplane for anomaly detection."]
    assert MODEL_REGISTRY.info("vision_inctrl").metadata["paper_fidelity"] == ("paper-adaptation")

    # Construction is offline-safe; loading is deferred until fit.
    VisionInCTRL(device="cpu")


class _IdentityBlock(torch.nn.Module):
    def __init__(self, *, batch_first: bool) -> None:
        super().__init__()
        self.attn = SimpleNamespace(batch_first=batch_first)

    def forward(self, tokens):  # noqa: ANN001, ANN201
        expected_axis = 1 if self.attn.batch_first else 0
        assert tokens.shape[expected_axis] == 226
        return tokens if self.attn.batch_first else (tokens, None)


class _TinyOpenCLIP(torch.nn.Module):
    def __init__(self, *, batch_first: bool) -> None:
        super().__init__()
        visual = torch.nn.Module()
        visual.conv1 = torch.nn.Conv2d(3, 896, 16, 16, bias=False)
        visual.class_embedding = torch.nn.Parameter(torch.zeros(896))
        position = torch.zeros(226, 896)
        position[:, 1] = 1.0
        visual.positional_embedding = torch.nn.Parameter(position)
        visual.patch_dropout = torch.nn.Identity()
        visual.ln_pre = torch.nn.Identity()
        visual.transformer = SimpleNamespace(
            resblocks=torch.nn.ModuleList(
                [_IdentityBlock(batch_first=batch_first) for _ in range(12)]
            )
        )
        visual.ln_post = torch.nn.Identity()
        visual.proj = torch.nn.Parameter(torch.zeros(896, 640))
        self.visual = visual
        with torch.no_grad():
            visual.conv1.weight.zero_()
            visual.conv1.weight[0].fill_(1.0 / (3 * 16 * 16))

    def encode_text(self, tokens):  # noqa: ANN001, ANN201
        features = torch.zeros(len(tokens), 640, device=tokens.device)
        features[:, 0] = 1.0
        features[:, 1] = tokens[:, 0].float()
        return features


def _preprocess(image):  # noqa: ANN001, ANN201
    array = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _tokenize(prompts):  # noqa: ANN001, ANN201
    anomalous = [
        int("damaged" in prompt or " with " in prompt or "without airplane" in prompt)
        for prompt in prompts
    ]
    return torch.tensor(anomalous, dtype=torch.long).unsqueeze(1)


@pytest.mark.parametrize("batch_first", [True, False])
def test_inctrl_backend_runs_released_residual_equations(batch_first: bool) -> None:
    from pyimgano.models.inctrl import InCTRLHeads, OpenCLIPInCTRLBackend

    heads = InCTRLHeads()
    with torch.no_grad():
        for parameter in heads.parameters():
            parameter.zero_()
    backend = OpenCLIPInCTRLBackend(
        model=_TinyOpenCLIP(batch_first=batch_first),
        preprocess=_preprocess,
        tokenizer=_tokenize,
        heads=heads,
        device="cpu",
        batch_size=2,
    )
    normal = np.zeros((240, 240, 3), dtype=np.uint8)
    anomaly = np.full((240, 240, 3), 255, dtype=np.uint8)

    backend.fit([normal], "bottle")
    normal_score = backend.score(normal)
    anomaly_score = backend.score(anomaly)

    assert [tuple(memory.shape) for memory in backend._patch_memories] == [
        (225, 896),
        (225, 896),
        (225, 896),
    ]
    assert backend.last_residual_map_.shape == (15, 15)
    assert normal_score == pytest.approx(0.25, abs=1e-6)
    assert anomaly_score > normal_score
    assert all(not parameter.requires_grad for parameter in backend.model.parameters())
    assert all(not parameter.requires_grad for parameter in backend.heads.parameters())


def test_inctrl_fit_uses_prompts_without_target_training() -> None:
    from pyimgano.models.inctrl import VisionInCTRL

    class Backend:
        def fit(self, items, class_name):  # noqa: ANN001, ANN201
            self.items = list(items)
            self.class_name = class_name

        def score(self, item):  # noqa: ANN001, ANN201
            return float(np.asarray(item).mean())

    backend = Backend()
    supports = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
        np.full((4, 4, 3), 2, dtype=np.uint8),
    ]
    detector = VisionInCTRL(
        backend=backend,
        class_name="metal_nut",
        k_shot=2,
        contamination=0.25,
        device="cpu",
    ).fit(supports)

    assert backend.items == supports[:2]
    assert backend.class_name == "metal nut"
    assert detector.decision_scores_.tolist() == [0.0, 1.0]
    assert detector.decision_function([supports[2]]).tolist() == [2.0]
