"""PromptAD paper adaptation for one-class few-shot anomaly detection.

This implementation follows the CVPR 2024 method: a frozen LAION-400M
ViT-B/16+ CLIP, V-V attention for local features, semantic concatenation,
explicit anomaly margin, and two-layer normal visual memories.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, TensorDataset

from pyimgano.models._image_batch import coerce_rgb_image_batch
from pyimgano.utils.optional_deps import require
from pyimgano.utils.random_state import isolated_random_state_method

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)

_CLASS_NAME_MAP = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut",
}

_GENERIC_ANOMALY_TEMPLATES = (
    "damaged {}",
    "flawed {}",
    "abnormal {}",
    "imperfect {}",
    "blemished {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
)

# The object-specific descriptions published in the authors' supplement/code.
_OBJECT_ANOMALY_TEMPLATES = {
    "bottle": ("{} with large breakage", "{} with small breakage", "{} with contamination"),
    "toothbrush": ("{} with defect", "{} with anomaly"),
    "carpet": (
        "{} with hole",
        "{} with color stain",
        "{} with metal contamination",
        "{} with thread residue",
        "{} with thread",
        "{} with cut",
    ),
    "hazelnut": ("{} with crack", "{} with cut", "{} with hole", "{} with print"),
    "leather": (
        "{} with color stain",
        "{} with cut",
        "{} with fold",
        "{} with glue",
        "{} with poke",
    ),
    "cable": (
        "{} with bent wire",
        "{} with missing part",
        "{} with missing wire",
        "{} with cut",
        "{} with poke",
    ),
    "capsule": (
        "{} with crack",
        "{} with faulty imprint",
        "{} with poke",
        "{} with scratch",
        "{} squeezed with compression",
    ),
    "grid": (
        "{} with breakage",
        "{} with thread residue",
        "{} with thread",
        "{} with metal contamination",
        "{} with glue",
        "{} with a bent shape",
    ),
    "pill": (
        "{} with color stain",
        "{} with contamination",
        "{} with crack",
        "{} with faulty imprint",
        "{} with scratch",
        "{} with abnormal type",
    ),
    "transistor": (
        "{} with bent lead",
        "{} with cut lead",
        "{} with damage",
        "{} with misplaced transistor",
    ),
    "metal_nut": (
        "{} with a bent shape",
        "{} with color stain",
        "{} with a flipped orientation",
        "{} with scratch",
    ),
    "screw": (
        "{} with manipulated front",
        "{} with scratch neck",
        "{} with scratch head",
    ),
    "zipper": (
        "{} with broken teeth",
        "{} with fabric border",
        "{} with defect fabric",
        "{} with broken fabric",
        "{} with split teeth",
        "{} with squeezed teeth",
    ),
    "tile": (
        "{} with crack",
        "{} with glue strip",
        "{} with gray stroke",
        "{} with oil",
        "{} with rough surface",
    ),
    "wood": ("{} with color stain", "{} with hole", "{} with scratch", "{} with liquid"),
    "candle": (
        "{} with melded wax",
        "{} with foreign particals",
        "{} with extra wax",
        "{} with chunk of wax missing",
        "{} with weird candle wick",
        "{} with damaged corner of packaging",
        "{} with different colour spot",
    ),
    "capsules": (
        "{} with scratch",
        "{} with discolor",
        "{} with misshape",
        "{} with leak",
        "{} with bubble",
    ),
    "cashew": (
        "{} with breakage",
        "{} with small scratches",
        "{} with burnt",
        "{} with stuck together",
        "{} with spot",
    ),
    "chewinggum": (
        "{} with corner missing",
        "{} with scratches",
        "{} with chunk of gum missing",
        "{} with colour spot",
        "{} with cracks",
    ),
    "fryum": (
        "{} with breakage",
        "{} with scratches",
        "{} with burnt",
        "{} with colour spot",
        "{} with fryum stuck together",
        "{} with colour spot",
    ),
    "macaroni1": (
        "{} with color spot",
        "{} with small chip around edge",
        "{} with small scratches",
        "{} with breakage",
        "{} with cracks",
    ),
    "macaroni2": (
        "{} with color spot",
        "{} with small chip around edge",
        "{} with small scratches",
        "{} with breakage",
        "{} with cracks",
    ),
    "pcb1": ("{} with bent", "{} with scratch", "{} with missing", "{} with melt"),
    "pcb2": ("{} with bent", "{} with scratch", "{} with missing", "{} with melt"),
    "pcb3": ("{} with bent", "{} with scratch", "{} with missing", "{} with melt"),
    "pcb4": (
        "{} with scratch",
        "{} with extra",
        "{} with missing",
        "{} with wrong place",
        "{} with damage",
        "{} with burnt",
        "{} with dirt",
    ),
    "pipe_fryum": (
        "{} with breakage",
        "{} with small scratches",
        "{} with burnt",
        "{} with stuck together",
        "{} with colour spot",
        "{} with cracks",
    ),
}


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-12)


def _paper_harmonic_fusion(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Equations 17/18: ``1 / (1/a + 1/b)`` (without an extra factor of two)."""
    denominator = first + second
    return torch.where(denominator > 0, first * second / denominator, torch.zeros_like(first))


def _promptad_objective(
    visual_features: torch.Tensor,
    normal_text_features: torch.Tensor,
    manual_anomaly_features: torch.Tensor,
    learned_anomaly_features: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    alignment_weight: float = 0.001,
    anomaly_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PromptAD equations 12, 13, and 15."""
    visual = _normalize(visual_features).reshape(-1, visual_features.shape[-1])
    normal = _normalize(normal_text_features)
    manual = _normalize(manual_anomaly_features)
    learned = _normalize(learned_anomaly_features)
    anomaly = torch.cat((manual, learned), dim=0)

    normal_prototype = _normalize(normal.mean(dim=0, keepdim=True))
    anomaly_prototype = _normalize(anomaly.mean(dim=0, keepdim=True))
    logits = torch.cat((visual @ normal_prototype.T, visual @ anomaly.T), dim=1)
    clip_loss = F.cross_entropy(
        logits * logit_scale.to(device=logits.device, dtype=logits.dtype),
        torch.zeros(len(logits), dtype=torch.long, device=logits.device),
    )

    normal_distance = torch.linalg.vector_norm(visual - normal_prototype, dim=-1)
    anomaly_distance = torch.linalg.vector_norm(visual - anomaly_prototype, dim=-1)
    margin_loss = F.relu(normal_distance - anomaly_distance + float(anomaly_margin)).mean()

    manual_mean = _normalize(manual.mean(dim=0, keepdim=True))
    learned_mean = _normalize(learned.mean(dim=0, keepdim=True))
    alignment_loss = (manual_mean - learned_mean).square().sum()
    total = clip_loss + margin_loss + float(alignment_weight) * alignment_loss
    return total, clip_loss, margin_loss, alignment_loss


class PromptADPromptLearner(nn.Module):
    """Semantic-concatenation prompt learner from PromptAD equations 9--11."""

    def __init__(
        self,
        backend: Any,
        *,
        class_name: str,
        n_ctx: int = 4,
        n_pro: int = 1,
        n_ctx_ab: int = 1,
        n_pro_ab: int = 4,
        anomaly_templates: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if min(n_ctx, n_pro, n_ctx_ab, n_pro_ab) <= 0:
            raise ValueError("Prompt counts and context lengths must be positive")

        raw_class_name = str(class_name).strip().lower().replace(" ", "_")
        if not raw_class_name:
            raise ValueError("class_name must be non-empty")
        display_name = _CLASS_NAME_MAP.get(raw_class_name, raw_class_name.replace("_", " "))
        descriptions = list(_GENERIC_ANOMALY_TEMPLATES)
        descriptions.extend(_OBJECT_ANOMALY_TEMPLATES.get(raw_class_name, ()))
        if anomaly_templates is not None:
            descriptions = [str(item) for item in anomaly_templates]
        if not descriptions:
            raise ValueError("anomaly_templates must contain at least one description")
        try:
            anomaly_phrases = [description.format(display_name) for description in descriptions]
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("Each anomaly template must accept one '{}' class-name field") from exc

        self.n_ctx = int(n_ctx)
        self.n_pro = int(n_pro)
        self.n_ctx_ab = int(n_ctx_ab)
        self.n_pro_ab = int(n_pro_ab)
        self.n_manual = len(anomaly_phrases)
        prefix = " ".join(["N"] * self.n_ctx)
        anomaly_prefix = " ".join(["A"] * self.n_ctx_ab)

        normal_strings = [f"{prefix} {display_name}." for _ in range(self.n_pro)]
        manual_strings = [
            f"{prefix} {phrase}."
            for phrase in anomaly_phrases
            for _ in range(self.n_pro)
        ]
        # The learnable anomaly tokens are a suffix of the complete normal prompt,
        # as specified by equation 11 (the upstream code places them before obj.).
        learned_strings = [
            f"{prefix} {display_name} {anomaly_prefix}."
            for _ in range(self.n_pro)
            for _ in range(self.n_pro_ab)
        ]

        normal_template, normal_eot = self._embed_templates(backend, normal_strings)
        manual_template, manual_eot = self._embed_templates(backend, manual_strings)
        learned_template, learned_eot = self._embed_templates(backend, learned_strings)
        base_tokens = backend.tokenize([f"{prefix} {display_name}"])
        anomaly_start = int(base_tokens.argmax(dim=-1).item())
        if anomaly_start + self.n_ctx_ab >= int(learned_eot.min().item()):
            raise ValueError("Prompt exceeds the CLIP text context length")

        width = int(normal_template.shape[-1])
        dtype = normal_template.dtype
        self.normal_context = nn.Parameter(
            torch.empty(
                self.n_pro,
                self.n_ctx,
                width,
                dtype=dtype,
                device=normal_template.device,
            ).normal_(std=0.02)
        )
        self.anomaly_context = nn.Parameter(
            torch.empty(
                self.n_pro_ab,
                self.n_ctx_ab,
                width,
                dtype=dtype,
                device=normal_template.device,
            ).normal_(std=0.02)
        )
        self.register_buffer("normal_template", normal_template)
        self.register_buffer("manual_template", manual_template)
        self.register_buffer("learned_template", learned_template)
        self.register_buffer("normal_eot", normal_eot)
        self.register_buffer("manual_eot", manual_eot)
        self.register_buffer("learned_eot", learned_eot)
        self.anomaly_start = anomaly_start

    @staticmethod
    def _embed_templates(backend: Any, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = backend.tokenize(prompts)
        with torch.no_grad():
            embeddings = backend.embed_tokens(tokens).detach()
        return embeddings, tokens.argmax(dim=-1)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal = self.normal_template.clone()
        normal[:, 1 : 1 + self.n_ctx] = self.normal_context

        manual = self.manual_template.clone()
        manual[:, 1 : 1 + self.n_ctx] = self.normal_context.repeat(self.n_manual, 1, 1)

        learned = self.learned_template.clone()
        learned[:, 1 : 1 + self.n_ctx] = self.normal_context.repeat_interleave(
            self.n_pro_ab, dim=0
        )
        learned[:, self.anomaly_start : self.anomaly_start + self.n_ctx_ab] = (
            self.anomaly_context.repeat(self.n_pro, 1, 1)
        )
        return normal, manual, learned


class OpenCLIPPromptADBackend(nn.Module):
    """OpenCLIP ViT-B/16+ with PromptAD's frozen dual-path V-V attention."""

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-16-plus-240",
        pretrained: str = "laion400m_e32",
        image_size: int = 240,
        precision: str = "fp16",
        device: str = "cuda",
        memory_layers: tuple[int, int] = (2, 7),
        open_clip_module: Any = None,
        model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        preprocess: Any = None,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.image_size = int(image_size)
        self.precision = str(precision)
        requested_device = torch.device(device)
        self.device = (
            requested_device
            if requested_device.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.memory_layers = tuple(int(index) for index in memory_layers)
        if len(self.memory_layers) != 2 or min(self.memory_layers) < 0:
            raise ValueError("memory_layers must contain exactly two nonnegative block indexes")
        self._open_clip = open_clip_module
        self.model = model
        self._tokenizer = tokenizer
        self._preprocess = preprocess
        self._initialized = False

    def initialize(self) -> "OpenCLIPPromptADBackend":
        if self._initialized:
            return self
        if self.model is None:
            open_clip = self._open_clip or require(
                "open_clip",
                extra="clip",
                purpose="PromptAD's frozen OpenCLIP backbone",
            )
            result = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                precision=self.precision if self.device.type == "cuda" else "fp32",
                device=self.device,
                force_image_size=self.image_size,
            )
            if not isinstance(result, tuple) or len(result) not in (2, 3):
                raise RuntimeError("Unexpected OpenCLIP create_model_and_transforms result")
            if len(result) == 3:
                self.model, _, self._preprocess = result
            else:
                self.model, self._preprocess = result
            self._tokenizer = open_clip.get_tokenizer(self.model_name)

        self.model = self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._validate_model()
        self._initialized = True
        return self

    def _validate_model(self) -> None:
        if self.model is None:  # pragma: no cover - initialize guards this
            raise RuntimeError("OpenCLIP model is not initialized")
        required = (
            "token_embedding",
            "positional_embedding",
            "transformer",
            "ln_final",
            "text_projection",
            "logit_scale",
            "visual",
        )
        missing = [name for name in required if not hasattr(self.model, name)]
        visual = getattr(self.model, "visual", None)
        visual_required = (
            "conv1",
            "class_embedding",
            "positional_embedding",
            "ln_pre",
            "transformer",
            "ln_post",
            "proj",
        )
        missing.extend(
            f"visual.{name}" for name in visual_required if visual is None or not hasattr(visual, name)
        )
        if missing:
            raise TypeError(
                "PromptAD requires a classic OpenCLIP ViT text/visual tower; missing "
                + ", ".join(missing)
            )
        blocks = getattr(visual.transformer, "resblocks", None)
        if blocks is None or len(blocks) <= max(self.memory_layers) + 1:
            raise TypeError("PromptAD requires OpenCLIP visual.transformer.resblocks")
        if getattr(visual, "input_patchnorm", False):
            raise TypeError("PromptAD's published ViT-B/16+ path does not use input patch norm")
        if getattr(visual, "global_average_pool", False) or getattr(visual, "attn_pool", None):
            raise TypeError("PromptAD requires the ViT CLS-token pooling path")
        for block in blocks:
            attention = getattr(block, "attn", None)
            if not isinstance(attention, nn.MultiheadAttention) or attention.in_proj_weight is None:
                raise TypeError("PromptAD V-V attention requires nn.MultiheadAttention blocks")

    def tokenize(self, prompts: list[str]) -> torch.Tensor:
        self.initialize()
        if self._tokenizer is None:
            raise RuntimeError("OpenCLIP tokenizer is unavailable")
        return self._tokenizer(prompts).to(self.device)

    def _text_cast_dtype(self) -> torch.dtype:
        transformer = self.model.transformer  # type: ignore[union-attr]
        if hasattr(transformer, "get_cast_dtype"):
            return transformer.get_cast_dtype()
        return self.model.text_projection.dtype  # type: ignore[union-attr]

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        self.initialize()
        return self.model.token_embedding(tokens).to(  # type: ignore[union-attr]
            dtype=self._text_cast_dtype()
        )

    @property
    def logit_scale(self) -> torch.Tensor:
        self.initialize()
        return self.model.logit_scale.exp()  # type: ignore[union-attr]

    def encode_text_embeddings(
        self,
        embeddings: torch.Tensor,
        eot_indices: torch.Tensor,
    ) -> torch.Tensor:
        self.initialize()
        model = self.model
        cast_dtype = self._text_cast_dtype()
        x = embeddings.to(dtype=cast_dtype) + model.positional_embedding.to(dtype=cast_dtype)
        x = x.permute(1, 0, 2)
        mask = getattr(model, "attn_mask", None)
        x = model.transformer(x, attn_mask=mask)
        if isinstance(x, tuple):
            x = x[0]
        x = model.ln_final(x.permute(1, 0, 2))
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, eot_indices.to(x.device)] @ model.text_projection

    @staticmethod
    def _dual_attention(
        attention: nn.MultiheadAttention, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the author's V-V and original Q-K attention branches."""
        length, batch, width = x.shape
        heads = int(attention.num_heads)
        head_width = width // heads
        bias = attention.in_proj_bias
        qkv = F.linear(
            x,
            attention.in_proj_weight,
            bias,
        )
        qkv = qkv.permute(1, 0, 2).reshape(batch, length, 3, heads, head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scale = head_width**-0.5
        original_weights = ((query @ key.transpose(-2, -1)) * scale).softmax(dim=-1)
        value_weights = ((value @ value.transpose(-2, -1)) * scale).softmax(dim=-1)
        if float(attention.dropout):
            original_weights = F.dropout(
                original_weights, p=float(attention.dropout), training=attention.training
            )
            value_weights = F.dropout(
                value_weights, p=float(attention.dropout), training=attention.training
            )

        def project(weights: torch.Tensor) -> torch.Tensor:
            output = (weights @ value).transpose(1, 2).reshape(batch, length, width)
            output = F.linear(output, attention.out_proj.weight, attention.out_proj.bias)
            return output.permute(1, 0, 2)

        return project(value_weights), project(original_weights)

    def _encode_vv_visual(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        visual = self.model.visual  # type: ignore[union-attr]
        dtype = visual.conv1.weight.dtype
        x = visual.conv1(images.to(dtype=dtype))
        grid = (int(x.shape[-2]), int(x.shape[-1]))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(dtype=x.dtype).reshape(1, 1, -1)
        x = torch.cat((class_token.expand(x.shape[0], -1, -1), x), dim=1)
        position = visual.positional_embedding.to(dtype=x.dtype)
        if position.ndim == 2:
            position = position.unsqueeze(0)
        if position.shape[1] != x.shape[1]:
            raise RuntimeError(
                f"PromptAD expected {x.shape[1]} positional tokens, got {position.shape[1]}"
            )
        x = visual.ln_pre(visual.patch_dropout(x + position))
        local = x.permute(1, 0, 2)
        original = local
        memories: dict[int, torch.Tensor] = {}
        for index, block in enumerate(visual.transformer.resblocks):
            normalized = block.ln_1(original)
            value_residual, original_residual = self._dual_attention(block.attn, normalized)
            local = local + value_residual
            original = original + original_residual
            # The authors' hooks retain a view which the next block updates via
            # ``+=`` before its MLP. Reproduce those published gallery tensors.
            previous_index = index - 1
            if previous_index in self.memory_layers:
                memories[previous_index] = original.permute(1, 0, 2)[:, 1:, :]
            original = original + block.mlp(block.ln_2(original))

        local = local.permute(1, 0, 2).clone()
        original = original.permute(1, 0, 2)
        local[:, 0] = original[:, 0]
        pooled = visual.ln_post(local[:, 0])
        tokens = visual.ln_post(local[:, 1:])
        if visual.proj is not None:
            pooled = pooled @ visual.proj
            tokens = tokens @ visual.proj
        return pooled, tokens, memories[self.memory_layers[0]], memories[self.memory_layers[1]], grid

    def encode_image(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        self.initialize()
        global_feature, local, memory1, memory2, grid = self._encode_vv_visual(
            images.to(self.device)
        )
        return (
            _normalize(global_feature),
            _normalize(local),
            _normalize(memory1),
            _normalize(memory2),
            grid,
        )

    def preprocess_images(self, images: NDArray) -> torch.Tensor:
        self.initialize()
        if self._preprocess is None:
            raise RuntimeError("OpenCLIP preprocessing transform is unavailable")
        from PIL import Image

        tensors = []
        for image in images:
            array = np.asarray(image)
            if not np.isfinite(array).all():
                raise ValueError("PromptAD images must contain only finite values")
            if float(array.min()) < 0 or float(array.max()) > 255:
                raise ValueError("PromptAD image values must be in [0, 1] or [0, 255]")
            if np.issubdtype(array.dtype, np.floating):
                if float(array.max()) <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
            tensors.append(self._preprocess(Image.fromarray(array, mode="RGB")))
        return torch.stack(tensors).to(self.device)


@register_model(
    "vision_promptad",
    tags=(
        "vision",
        "deep",
        "clip",
        "openclip",
        "promptad",
        "few-shot",
        "one-class",
        "prompt",
        "pixel_map",
        "cvpr2024",
    ),
    metadata={
        "description": "PromptAD semantic-concatenation/EAM method with frozen VV-CLIP",
        "paper": "PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html",
        "official_code_url": "https://github.com/FuNz-0/PromptAD",
        "year": 2024,
        "implementation_status": "native-paper-method-openclip-adaptation",
        "paper_fidelity": "paper-adaptation",
        "conference": "CVPR",
        "type": "prompt-learning",
        "supervision": "one-class",
        "supports_pixel_map": True,
        "requires_checkpoint": True,
        "weights_source": "OpenCLIP ViT-B-16-plus-240 laion400m_e32",
    },
)
class VisionPromptAD(BaseVisionDeepDetector):
    """PromptAD with paper defaults and an optional injectable CLIP backend.

    ``fit`` expects only normal images. Set ``training_task='classification'``
    for the global CLS objective or ``'segmentation'`` for the local-token
    objective used by the authors' separate pixel-level training script.
    """

    def __init__(
        self,
        *,
        class_name: str = "object",
        openclip_model_name: str = "ViT-B-16-plus-240",
        openclip_pretrained: str = "laion400m_e32",
        image_size: int = 240,
        n_ctx: int = 4,
        n_ctx_ab: int = 1,
        n_pro: int = 1,
        n_pro_ab: int = 4,
        learning_rate: float = 0.002,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        alignment_weight: float = 0.001,
        anomaly_margin: float = 0.0,
        epochs: int = 100,
        batch_size: int = 400,
        training_task: str = "classification",
        gaussian_sigma: float = 4.0,
        precision: str = "fp16",
        device: str = "cuda",
        random_state: Optional[int] = 111,
        anomaly_templates: Optional[Sequence[str]] = None,
        backend: Any = None,
        open_clip_module: Any = None,
        **kwargs: Any,
    ) -> None:
        if min(image_size, n_ctx, n_ctx_ab, n_pro, n_pro_ab, batch_size) <= 0:
            raise ValueError("PromptAD sizes and prompt counts must be positive")
        if (
            epochs < 0
            or learning_rate <= 0
            or momentum < 0
            or weight_decay < 0
            or alignment_weight < 0
            or anomaly_margin < 0
            or gaussian_sigma < 0
        ):
            raise ValueError("PromptAD optimizer parameters are invalid")
        if precision not in {"fp16", "fp32", "bf16"}:
            raise ValueError("precision must be 'fp16', 'fp32', or 'bf16'")
        task = {"cls": "classification", "seg": "segmentation"}.get(
            str(training_task).lower(), str(training_task).lower()
        )
        if task not in {"classification", "segmentation"}:
            raise ValueError("training_task must be 'classification' or 'segmentation'")
        requested_device = torch.device(device)
        resolved_device = (
            requested_device
            if requested_device.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        super().__init__(
            batch_size=int(batch_size),
            device=str(resolved_device),
            random_state=random_state,
            **kwargs,
        )
        self.class_name = str(class_name)
        self.openclip_model_name = str(openclip_model_name)
        self.openclip_pretrained = str(openclip_pretrained)
        self.image_size = int(image_size)
        self.n_ctx = int(n_ctx)
        self.n_ctx_ab = int(n_ctx_ab)
        self.n_pro = int(n_pro)
        self.n_pro_ab = int(n_pro_ab)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.alignment_weight = float(alignment_weight)
        self.anomaly_margin = float(anomaly_margin)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.training_task = task
        self.gaussian_sigma = float(gaussian_sigma)
        self.precision = str(precision)
        self.device = resolved_device
        self.random_state = random_state
        self.anomaly_templates = anomaly_templates
        self._open_clip_module = open_clip_module
        self.backend_ = backend
        self.prompt_learner_: Optional[PromptADPromptLearner] = None
        self.text_features_: Optional[torch.Tensor] = None
        self.feature_gallery1_: Optional[torch.Tensor] = None
        self.feature_gallery2_: Optional[torch.Tensor] = None

    def _ensure_backend(self) -> Any:
        if self.backend_ is None:
            self.backend_ = OpenCLIPPromptADBackend(
                model_name=self.openclip_model_name,
                pretrained=self.openclip_pretrained,
                image_size=self.image_size,
                precision=self.precision,
                device=str(self.device),
                open_clip_module=self._open_clip_module,
            )
        if hasattr(self.backend_, "to"):
            self.backend_.to(self.device)
        if hasattr(self.backend_, "device"):
            self.backend_.device = self.device
        self.backend_.initialize()
        return self.backend_

    def _encode_prompt_features(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.prompt_learner_ is None:
            raise RuntimeError("PromptAD prompt learner is not initialized")
        normal, manual, learned = self.prompt_learner_()
        backend = self._ensure_backend()
        return (
            backend.encode_text_embeddings(normal, self.prompt_learner_.normal_eot),
            backend.encode_text_embeddings(manual, self.prompt_learner_.manual_eot),
            backend.encode_text_embeddings(learned, self.prompt_learner_.learned_eot),
        )

    def _extract_visual_features(
        self, images: NDArray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        backend = self._ensure_backend()
        outputs: list[list[torch.Tensor]] = [[], [], [], []]
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = backend.preprocess_images(images[start : start + self.batch_size])
                encoded = backend.encode_image(batch)
                for index in range(4):
                    outputs[index].append(encoded[index].detach().cpu())
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple(torch.cat(parts) for parts in outputs),
        )

    def _build_text_gallery(self) -> None:
        with torch.no_grad():
            normal, manual, learned = self._encode_prompt_features()
            normal_prototype = _normalize(_normalize(normal).mean(dim=0, keepdim=True))
            anomaly = torch.cat((_normalize(manual), _normalize(learned)), dim=0)
            anomaly_prototype = _normalize(anomaly.mean(dim=0, keepdim=True))
            self.text_features_ = torch.cat((normal_prototype, anomaly_prototype), dim=0)

    @isolated_random_state_method
    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionPromptAD":
        """Learn prompts and normal visual memories from few-shot normal images."""
        del y
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        images = coerce_rgb_image_batch(x_value)
        if len(images) == 0:
            raise ValueError("PromptAD requires at least one normal support image")
        backend = self._ensure_backend()
        self.prompt_learner_ = PromptADPromptLearner(
            backend,
            class_name=self.class_name,
            n_ctx=self.n_ctx,
            n_pro=self.n_pro,
            n_ctx_ab=self.n_ctx_ab,
            n_pro_ab=self.n_pro_ab,
            anomaly_templates=self.anomaly_templates,
        ).to(self.device)

        global_features, local_features, memory1, memory2 = self._extract_visual_features(images)
        self.feature_gallery1_ = _normalize(memory1.reshape(-1, memory1.shape[-1])).to(
            self.device
        )
        self.feature_gallery2_ = _normalize(memory2.reshape(-1, memory2.shape[-1])).to(
            self.device
        )
        training_features = (
            global_features if self.training_task == "classification" else local_features
        )
        loader = DataLoader(
            TensorDataset(training_features),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.SGD(
            self.prompt_learner_.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.epochs),
            eta_min=1e-5,
        )
        self.prompt_learner_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (visual,) in loader:
                normal, manual, learned = self._encode_prompt_features()
                loss, _, _, _ = _promptad_objective(
                    visual.to(self.device),
                    normal,
                    manual,
                    learned,
                    logit_scale=backend.logit_scale,
                    alignment_weight=self.alignment_weight,
                    anomaly_margin=self.anomaly_margin,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "PromptAD epoch %d/%d loss %.6f",
                    epoch + 1,
                    self.epochs,
                    epoch_loss / max(1, len(loader)),
                )

        self.prompt_learner_.eval()
        self._build_text_gallery()
        self.is_fitted_ = True
        self.decision_scores_ = self._score_images(images)[0]
        self._process_decision_scores()
        self._set_n_classes(None)
        return self

    def _check_promptad_fitted(self) -> None:
        if (
            not bool(getattr(self, "is_fitted_", False))
            or self.text_features_ is None
            or self.feature_gallery1_ is None
            or self.feature_gallery2_ is None
        ):
            raise RuntimeError("Call fit() before PromptAD inference")

    def _score_images(self, images: NDArray) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        self._check_promptad_fitted()
        if len(images) == 0:
            raise ValueError("PromptAD requires at least one image for inference")
        backend = self._ensure_backend()
        scores: list[torch.Tensor] = []
        maps: list[torch.Tensor] = []
        original_size = (int(images.shape[1]), int(images.shape[2]))
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = backend.preprocess_images(images[start : start + self.batch_size])
                global_feature, local, memory1, memory2, grid = backend.encode_image(batch)
                text = self.text_features_.to(device=global_feature.device, dtype=global_feature.dtype)
                scale = backend.logit_scale.to(
                    device=global_feature.device, dtype=global_feature.dtype
                )
                image_prompt_score = (scale * global_feature @ text.T).softmax(dim=-1)[:, 1]
                pixel_prompt_score = (scale * local @ text.T).softmax(dim=-1)[..., 1]

                gallery1 = self.feature_gallery1_.to(memory1.device, dtype=memory1.dtype)
                gallery2 = self.feature_gallery2_.to(memory2.device, dtype=memory2.dtype)
                visual1 = ((1.0 - memory1 @ gallery1.T).amin(dim=-1) / 2.0).clamp(0, 1)
                visual2 = ((1.0 - memory2 @ gallery2.T).amin(dim=-1) / 2.0).clamp(0, 1)
                visual_map = 0.5 * (visual1 + visual2)
                image_score = _paper_harmonic_fusion(
                    visual_map.amax(dim=-1), image_prompt_score
                )
                pixel_map = _paper_harmonic_fusion(visual_map, pixel_prompt_score)
                pixel_map = pixel_map.reshape(-1, 1, grid[0], grid[1])
                pixel_map = F.interpolate(
                    pixel_map,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                scores.append(image_score.cpu())
                maps.append(pixel_map.cpu())

        score_array = torch.cat(scores).numpy().astype(np.float64, copy=False)
        map_array = torch.cat(maps).numpy().astype(np.float32, copy=False)
        if self.gaussian_sigma:
            for index in range(len(map_array)):
                map_array[index] = gaussian_filter(map_array[index], sigma=self.gaussian_sigma)
        return score_array, map_array

    def predict_anomaly_map(
        self, x: object = MISSING, **kwargs: object
    ) -> NDArray[np.float32]:
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        return self._score_images(coerce_rgb_image_batch(x_value))[1]

    def get_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="get_anomaly_map")
        return self.predict_anomaly_map(x_value)[0]

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        return self._score_images(coerce_rgb_image_batch(x_value))[0]

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        if batch_size is None:
            return self.predict(x_value)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        previous = self.batch_size
        try:
            self.batch_size = batch_size
            return self.predict(x_value)
        finally:
            self.batch_size = previous
