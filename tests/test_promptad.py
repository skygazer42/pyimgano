from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional


class _TinyPromptADBackend(torch.nn.Module):
    def __init__(self, width: int = 8, context_length: int = 32) -> None:
        super().__init__()
        self.width = int(width)
        self.context_length = int(context_length)
        self.embedding = torch.nn.Embedding(256, self.width)
        with torch.no_grad():
            values = torch.arange(256 * self.width, dtype=torch.float32).reshape(256, self.width)
            self.embedding.weight.copy_((values.remainder(31) - 15) / 31)
        self.embedding.weight.requires_grad_(False)
        self.register_buffer("_scale", torch.tensor(10.0))

    def initialize(self):  # noqa: ANN201
        return self

    def tokenize(self, prompts: list[str]) -> torch.Tensor:
        rows = []
        for prompt in prompts:
            words = prompt.replace(".", " .").split()
            ids = [1] + [2 + sum(map(ord, word)) % 200 for word in words] + [255]
            if len(ids) > self.context_length:
                raise ValueError("tiny tokenizer context overflow")
            rows.append(ids + [0] * (self.context_length - len(ids)))
        return torch.tensor(rows, dtype=torch.long, device=self._scale.device)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    def encode_text_embeddings(
        self, embeddings: torch.Tensor, eot_indices: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)[None, :]
        mask = (positions <= eot_indices[:, None]).to(embeddings.dtype).unsqueeze(-1)
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self._scale

    def preprocess_images(self, images: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(images).transpose(0, 3, 1, 2).copy()).float()
        if tensor.max() > 1:
            tensor /= 255.0
        return F.interpolate(tensor, size=(8, 8), mode="bilinear", align_corners=False)

    def encode_image(self, images: torch.Tensor):  # noqa: ANN201
        batch = int(images.shape[0])
        scalar = images.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        basis = torch.arange(1, self.width + 1, device=images.device, dtype=images.dtype)[None]
        global_feature = F.normalize(basis + scalar, dim=-1)
        offsets = torch.arange(4, device=images.device, dtype=images.dtype)[None, :, None] / 10
        local = F.normalize(basis[:, None, :] + scalar[:, None, :] + offsets, dim=-1)
        memory1 = F.normalize(local + 0.1, dim=-1)
        memory2 = F.normalize(local - 0.1, dim=-1)
        assert local.shape == (batch, 4, self.width)
        return global_feature, local, memory1, memory2, (2, 2)


class _TinyBlock(torch.nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, 2)
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, width * 2),
            torch.nn.GELU(),
            torch.nn.Linear(width * 2, width),
        )


class _TinyTransformer(torch.nn.Module):
    def __init__(self, blocks: int = 3, width: int = 8) -> None:
        super().__init__()
        self.resblocks = torch.nn.ModuleList([_TinyBlock(width) for _ in range(blocks)])


class _IdentityTextTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.cast_dtype = torch.float32

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:  # noqa: ANN001
        return x

    def get_cast_dtype(self) -> torch.dtype:
        return self.cast_dtype


class _TinyVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=False)
        self.class_embedding = torch.nn.Parameter(torch.randn(8))
        self.positional_embedding = torch.nn.Parameter(torch.randn(5, 8) * 0.01)
        self.patch_dropout = torch.nn.Identity()
        self.ln_pre = torch.nn.LayerNorm(8)
        self.transformer = _TinyTransformer()
        self.ln_post = torch.nn.LayerNorm(8)
        self.proj = torch.nn.Parameter(torch.randn(8, 6) * 0.1)
        self.input_patchnorm = False
        self.global_average_pool = False
        self.attn_pool = None


class _TinyCLIP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _TinyVisual()
        self.token_embedding = torch.nn.Embedding(256, 8)
        self.positional_embedding = torch.nn.Parameter(torch.randn(32, 8) * 0.01)
        self.transformer = _IdentityTextTransformer()
        self.ln_final = torch.nn.LayerNorm(8)
        self.text_projection = torch.nn.Parameter(torch.randn(8, 6) * 0.1)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))
        self.register_buffer("attn_mask", torch.empty(0), persistent=False)


def _tiny_tokenizer(prompts: list[str]) -> torch.Tensor:
    backend = _TinyPromptADBackend(context_length=32)
    return backend.tokenize(prompts).cpu()


def _ordinary_visual(model: _TinyCLIP, images: torch.Tensor):  # noqa: ANN202
    visual = model.visual
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.reshape(1, 1, -1).expand(x.shape[0], -1, -1)
    x = visual.ln_pre(torch.cat((cls, x), dim=1) + visual.positional_embedding)
    x = x.permute(1, 0, 2)
    memories = []
    for index, block in enumerate(visual.transformer.resblocks):
        normalized = block.ln_1(x)
        x = x + block.attn(normalized, normalized, normalized, need_weights=False)[0]
        if index in (1, 2):
            memories.append(x.permute(1, 0, 2)[:, 1:])
        x = x + block.mlp(block.ln_2(x))
    x = x.permute(1, 0, 2)
    pooled = F.normalize(visual.ln_post(x[:, 0]) @ visual.proj, dim=-1)
    tokens = F.normalize(visual.ln_post(x[:, 1:]) @ visual.proj, dim=-1)
    return pooled, tokens, memories


def test_prompt_learner_matches_paper_prompt_shapes_and_suffix_order() -> None:
    from pyimgano.models.promptad import PromptADPromptLearner

    backend = _TinyPromptADBackend()
    learner = PromptADPromptLearner(backend, class_name="carpet")
    normal, manual, learned = learner()

    assert normal.shape == (1, 32, 8)
    assert manual.shape == (14, 32, 8)  # 8 generic + 6 carpet-specific MAPs
    assert learned.shape == (4, 32, 8)
    assert sum(parameter.numel() for parameter in learner.parameters()) == 64
    assert learner.anomaly_start > 1 + learner.n_ctx
    torch.testing.assert_close(
        learned[:, learner.anomaly_start : learner.anomaly_start + learner.n_ctx_ab],
        learner.anomaly_context,
    )
    torch.testing.assert_close(
        manual[:, 1 : 1 + learner.n_ctx], learner.normal_context.repeat(14, 1, 1)
    )


def test_promptad_objective_implements_clip_eam_and_alignment_equations() -> None:
    from pyimgano.models.promptad import _promptad_objective

    visual = torch.tensor([[0.0, 1.0]])
    normal = torch.tensor([[1.0, 0.0]])
    manual = torch.tensor([[0.0, 1.0]])
    learned = torch.tensor([[0.0, 1.0]])
    total, clip_loss, margin_loss, alignment_loss = _promptad_objective(
        visual,
        normal,
        manual,
        learned,
        logit_scale=torch.tensor(1.0),
    )

    expected_clip = F.cross_entropy(torch.tensor([[0.0, 1.0, 1.0]]), torch.tensor([0]))
    torch.testing.assert_close(clip_loss, expected_clip)
    torch.testing.assert_close(margin_loss, torch.tensor(math.sqrt(2.0)))
    torch.testing.assert_close(alignment_loss, torch.tensor(0.0))
    torch.testing.assert_close(total, expected_clip + math.sqrt(2.0))


def test_openclip_backend_preserves_cls_path_and_uses_vv_local_path() -> None:
    from pyimgano.models.promptad import OpenCLIPPromptADBackend

    torch.manual_seed(4)
    model = _TinyCLIP()
    backend = OpenCLIPPromptADBackend(
        model=model,
        tokenizer=_tiny_tokenizer,
        preprocess=lambda image: torch.zeros(3, 8, 8),
        memory_layers=(0, 1),
        precision="fp32",
        device="cpu",
    ).initialize()
    images = torch.randn(2, 3, 8, 8)
    global_feature, local, memory1, memory2, grid = backend.encode_image(images)
    expected_global, ordinary_tokens, memories = _ordinary_visual(model, images)

    assert grid == (2, 2)
    assert global_feature.shape == (2, 6)
    assert local.shape == (2, 4, 6)
    assert memory1.shape == memory2.shape == (2, 4, 8)
    torch.testing.assert_close(global_feature, expected_global, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(memory1, F.normalize(memories[0], dim=-1))
    torch.testing.assert_close(memory2, F.normalize(memories[1], dim=-1))
    assert not torch.allclose(local, ordinary_tokens)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_openclip_backend_loads_published_backbone_and_weights() -> None:
    from pyimgano.models.promptad import OpenCLIPPromptADBackend

    calls = []
    model = _TinyCLIP()

    class _FakeOpenCLIP:
        @staticmethod
        def create_model_and_transforms(name, **kwargs):  # noqa: ANN001, ANN202
            calls.append((name, kwargs))
            return model, object(), lambda image: torch.zeros(3, 8, 8)

        @staticmethod
        def get_tokenizer(name):  # noqa: ANN001, ANN205
            assert name == "ViT-B-16-plus-240"
            return _tiny_tokenizer

    backend = OpenCLIPPromptADBackend(
        open_clip_module=_FakeOpenCLIP,
        precision="fp32",
        device="cpu",
        memory_layers=(0, 1),
    ).initialize()

    assert backend.model is model
    assert calls == [
        (
            "ViT-B-16-plus-240",
            {
                "pretrained": "laion400m_e32",
                "precision": "fp32",
                "device": torch.device("cpu"),
                "force_image_size": 240,
            },
        )
    ]


def test_openclip_backend_uses_transformer_cast_dtype_for_prompt_embeddings() -> None:
    from pyimgano.models.promptad import OpenCLIPPromptADBackend

    model = _TinyCLIP()
    model.transformer.cast_dtype = torch.float16
    backend = OpenCLIPPromptADBackend(
        model=model,
        tokenizer=_tiny_tokenizer,
        preprocess=lambda image: torch.zeros(3, 8, 8),
        memory_layers=(0, 1),
        precision="fp16",
        device="cpu",
    ).initialize()

    tokens = backend.tokenize(["N object."])
    assert backend.embed_tokens(tokens).dtype == torch.float16


def test_promptad_fit_scores_and_localizes_with_paper_components() -> None:
    from pyimgano.models.promptad import VisionPromptAD

    rng = np.random.default_rng(12)
    train = rng.integers(0, 255, size=(4, 16, 16, 3), dtype=np.uint8)
    test = rng.integers(0, 255, size=(2, 16, 16, 3), dtype=np.uint8)
    detector = VisionPromptAD(
        class_name="carpet",
        backend=_TinyPromptADBackend(),
        epochs=1,
        batch_size=2,
        gaussian_sigma=0,
        precision="fp32",
        device="cpu",
    ).fit(train)

    scores = detector.decision_function(test)
    maps = detector.predict_anomaly_map(test)
    assert scores.shape == (2,)
    assert maps.shape == (2, 16, 16)
    assert np.isfinite(scores).all()
    assert np.isfinite(maps).all()
    assert detector.text_features_.shape == (2, 8)
    assert detector.feature_gallery1_.shape == (16, 8)
    assert detector.feature_gallery2_.shape == (16, 8)
    assert sum(parameter.numel() for parameter in detector.prompt_learner_.parameters()) == 64


def test_promptad_published_defaults() -> None:
    from pyimgano.models.promptad import VisionPromptAD, _paper_harmonic_fusion

    signature = inspect.signature(VisionPromptAD.__init__)
    expected = {
        "openclip_model_name": "ViT-B-16-plus-240",
        "openclip_pretrained": "laion400m_e32",
        "image_size": 240,
        "n_ctx": 4,
        "n_ctx_ab": 1,
        "n_pro": 1,
        "n_pro_ab": 4,
        "learning_rate": 0.002,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "alignment_weight": 0.001,
        "anomaly_margin": 0.0,
        "epochs": 100,
        "batch_size": 400,
        "gaussian_sigma": 4.0,
    }
    assert {name: signature.parameters[name].default for name in expected} == expected
    torch.testing.assert_close(
        _paper_harmonic_fusion(torch.tensor([0.5]), torch.tensor([0.25])),
        torch.tensor([1.0 / 6.0]),
    )
