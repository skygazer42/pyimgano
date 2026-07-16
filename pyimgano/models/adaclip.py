"""AdaCLIP paper inference adaptation on a frozen OpenAI CLIP ViT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

from pyimgano.utils.optional_deps import require

from ._image_batch import _coerce_single_rgb_image
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .deep_io import safe_torch_load
from .openclip_backend import _load_openclip_model_and_preprocess
from .registry import register_model

PAPER_MODEL = "ViT-L-14-336"
PAPER_PRETRAINED = "openai"
PAPER_IMAGE_SIZE = 518
PAPER_OUTPUT_LAYERS = (6, 12, 18, 24)
PAPER_PROMPT_DEPTH = 4
PAPER_PROMPT_LENGTH = 5
PAPER_K_CLUSTERS = 20
PAPER_LOGIT_SCALE = 100.0
PAPER_HSF_ALPHA = 0.2
PAPER_GAUSSIAN_SIGMA = 4.0
PAPER_RANDOM_STATE = 111

NORMAL_STATES = (
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
)
ANOMALOUS_STATES = (
    "damaged {}",
    "broken {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
)
PROMPT_TEMPLATES = (
    "a bad photo of a {}.",
    "a low resolution photo of the {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-12)


def _create_text_ensemble(class_name: str) -> tuple[list[str], list[str]]:
    object_name = " ".join(str(class_name).replace("-", " ").replace("_", " ").split())
    if not object_name:
        raise ValueError("class_name must be non-empty")

    def expand(states: Sequence[str]) -> list[str]:
        prompted_states = [state.format(object_name) for state in states]
        return [
            template.format(state) for state in prompted_states for template in PROMPT_TEMPLATES
        ]

    return expand(NORMAL_STATES), expand(ANOMALOUS_STATES)


def _run_block(
    block: nn.Module,
    x: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_first = bool(getattr(getattr(block, "attn", None), "batch_first", False))
    block_input = x.permute(1, 0, 2) if batch_first else x
    try:
        output = block(block_input, attn_mask=attn_mask)
    except TypeError:
        output = block(block_input)
    result = output[0] if isinstance(output, tuple) else output
    return result.permute(1, 0, 2) if batch_first else result


class ProjectLayer(nn.Module):
    """Independent linear projections used by the authors for each replica."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_replicas: int,
        *,
        stack: bool = False,
        drop_cls: bool = False,
    ) -> None:
        super().__init__()
        self.head = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(int(num_replicas))]
        )
        self.stack = bool(stack)
        self.drop_cls = bool(drop_cls)

    def forward(self, tokens: torch.Tensor | Sequence[torch.Tensor]):
        inputs = tokens if isinstance(tokens, Sequence) else [tokens] * len(self.head)
        if len(inputs) != len(self.head):
            raise ValueError("Projection input count does not match its replicas")
        outputs = [
            layer(value[:, 1:] if self.drop_cls else value)
            for layer, value in zip(self.head, inputs)
        ]
        return torch.stack(outputs, dim=1) if self.stack else outputs


class PromptLayer(nn.Module):
    def __init__(
        self,
        channel: int,
        length: int,
        depth: int,
        *,
        prompting_type: str,
        enabled: bool,
    ) -> None:
        super().__init__()
        self.length = int(length)
        self.depth = int(depth)
        self.prompting_type = str(prompting_type)
        self.enabled = bool(enabled)
        if self.enabled and "S" in self.prompting_type:
            self.static_prompts = nn.ParameterList(
                [nn.Parameter(torch.empty(self.length, channel)) for _ in range(self.depth)]
            )
            for prompt in self.static_prompts:
                nn.init.normal_(prompt, std=0.02)

    def context(
        self,
        layer_index: int,
        batch_size: int,
        dynamic_prompts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        if "S" in self.prompting_type:
            pieces.append(self.static_prompts[layer_index].unsqueeze(0).expand(batch_size, -1, -1))
        if "D" in self.prompting_type:
            if dynamic_prompts is None:
                raise RuntimeError("Dynamic AdaCLIP prompts were not generated")
            pieces.append(dynamic_prompts.expand(batch_size, -1, -1))
        if not pieces:
            raise RuntimeError("AdaCLIP prompting requires static or dynamic prompts")
        return torch.stack(pieces).sum(dim=0)


def _hybrid_semantic_fusion(
    patch_tokens: Sequence[torch.Tensor],
    anomaly_logits: Sequence[torch.Tensor],
    *,
    k_clusters: int,
    random_state: Optional[int],
) -> torch.Tensor:
    """Author-code HSF: top anomaly tokens, KMeans, then multi-level centroids."""

    anomaly_probability = torch.stack(anomaly_logits, dim=1).mean(dim=1).softmax(dim=-1)[..., 1]
    selected_count = min(anomaly_probability.shape[1], int(k_clusters) * 5)
    if selected_count < int(k_clusters):
        raise ValueError(
            f"k_clusters={k_clusters} exceeds the {selected_count} available patch tokens"
        )
    top_indices = anomaly_probability.topk(selected_count, dim=1).indices
    selected = [
        tokens.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        for tokens in patch_tokens
    ]
    clustering_features = torch.cat(selected, dim=-1)

    fused: list[torch.Tensor] = []
    for batch_index in range(clustering_features.shape[0]):
        labels_np = KMeans(
            n_clusters=int(k_clusters),
            n_init=1,
            random_state=random_state,
        ).fit_predict(clustering_features[batch_index].detach().cpu().numpy())
        labels = torch.as_tensor(labels_np, device=clustering_features.device)
        centers = [
            torch.cat(
                [tokens[batch_index, labels == cluster] for tokens in selected],
                dim=0,
            ).mean(dim=0)
            for cluster in range(int(k_clusters))
        ]
        fused.append(torch.stack(centers).mean(dim=0))
    return _normalize(torch.stack(fused))


class AdaCLIPNetwork(nn.Module):
    """Hybrid visual/text prompting, projections, and HSF from AdaCLIP."""

    def __init__(
        self,
        freeze_clip: nn.Module,
        *,
        image_size: int = PAPER_IMAGE_SIZE,
        output_layers: Sequence[int] = PAPER_OUTPUT_LAYERS,
        prompting_depth: int = PAPER_PROMPT_DEPTH,
        prompting_length: int = PAPER_PROMPT_LENGTH,
        prompting_branch: str = "VL",
        prompting_type: str = "SD",
        use_hsf: bool = True,
        k_clusters: int = PAPER_K_CLUSTERS,
        random_state: Optional[int] = PAPER_RANDOM_STATE,
        tokenizer: Any,
    ) -> None:
        super().__init__()
        self.freeze_clip = freeze_clip.requires_grad_(False).eval()
        self.image_size = int(image_size)
        self.output_layers = tuple(int(layer) for layer in output_layers)
        self.prompting_depth = int(prompting_depth)
        self.prompting_length = int(prompting_length)
        self.prompting_branch = str(prompting_branch)
        self.prompting_type = str(prompting_type)
        self.use_hsf = bool(use_hsf)
        self.k_clusters = int(k_clusters)
        self.random_state = random_state
        self.tokenizer = tokenizer
        self._validate_backbone()

        visual = self.freeze_clip.visual
        visual_width = int(visual.conv1.out_channels)
        text_width = int(self.freeze_clip.token_embedding.embedding_dim)
        text_projection = self.freeze_clip.text_projection
        embed_dim = int(text_projection.shape[-1])
        self.text_prompter = PromptLayer(
            text_width,
            self.prompting_length,
            self.prompting_depth,
            prompting_type=self.prompting_type,
            enabled="L" in self.prompting_branch,
        )
        self.visual_prompter = PromptLayer(
            visual_width,
            self.prompting_length,
            self.prompting_depth,
            prompting_type=self.prompting_type,
            enabled="V" in self.prompting_branch,
        )
        self.patch_token_layer = ProjectLayer(
            visual_width,
            embed_dim,
            len(self.output_layers),
            drop_cls=True,
        )
        self.cls_token_layer = ProjectLayer(embed_dim, embed_dim, 1)
        if "D" in self.prompting_type:
            self.dynamic_visual_prompt_generator = ProjectLayer(
                embed_dim,
                visual_width,
                self.prompting_length,
                stack=True,
            )
            self.dynamic_text_prompt_generator = ProjectLayer(
                embed_dim,
                text_width,
                self.prompting_length,
                stack=True,
            )

    def _validate_backbone(self) -> None:
        if self.image_size <= 0 or self.prompting_depth <= 0 or self.prompting_length <= 0:
            raise ValueError("AdaCLIP image and prompt sizes must be positive")
        if self.k_clusters <= 0:
            raise ValueError("k_clusters must be positive")
        if self.prompting_branch not in {"", "V", "L", "VL"}:
            raise ValueError("prompting_branch must be '', 'V', 'L', or 'VL'")
        if self.prompting_type not in {"S", "D", "SD"}:
            raise ValueError("prompting_type must be 'S', 'D', or 'SD'")
        required = (
            "visual",
            "token_embedding",
            "positional_embedding",
            "transformer",
            "ln_final",
            "text_projection",
        )
        missing = [name for name in required if not hasattr(self.freeze_clip, name)]
        visual = getattr(self.freeze_clip, "visual", None)
        visual_required = (
            "conv1",
            "class_embedding",
            "positional_embedding",
            "patch_dropout",
            "ln_pre",
            "transformer",
            "ln_post",
            "proj",
        )
        missing.extend(
            f"visual.{name}"
            for name in visual_required
            if visual is None or not hasattr(visual, name)
        )
        if missing:
            raise TypeError(
                "AdaCLIP requires a classic OpenCLIP ViT; missing " + ", ".join(missing)
            )
        visual_blocks = getattr(visual.transformer, "resblocks", None)
        text_blocks = getattr(self.freeze_clip.transformer, "resblocks", None)
        if visual_blocks is None or text_blocks is None:
            raise TypeError("AdaCLIP requires OpenCLIP transformer.resblocks")
        if (
            not self.output_layers
            or min(self.output_layers) <= 0
            or max(self.output_layers) > len(visual_blocks)
        ):
            raise ValueError("output_layers must be valid 1-based visual block indexes")
        if self.prompting_depth > min(len(visual_blocks), len(text_blocks)):
            raise ValueError("prompting_depth exceeds the CLIP transformer depth")
        if getattr(visual, "input_patchnorm", False):
            raise TypeError("AdaCLIP's published ViT-L/14 path does not use input patch norm")
        if getattr(visual, "global_average_pool", False) or getattr(visual, "attn_pool", None):
            raise TypeError("AdaCLIP requires the CLIP CLS-token pooling path")

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        return {
            name: state[name]
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def _visual_input(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        visual = self.freeze_clip.visual
        x = visual.conv1(images.to(dtype=visual.conv1.weight.dtype))
        grid = (int(x.shape[-2]), int(x.shape[-1]))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(dtype=x.dtype).reshape(1, 1, -1)
        x = torch.cat((class_token.expand(x.shape[0], -1, -1), x), dim=1)
        position = visual.positional_embedding.to(dtype=x.dtype)
        if position.ndim == 2:
            position = position.unsqueeze(0)
        if position.shape[1] != x.shape[1]:
            raise RuntimeError(
                f"AdaCLIP expected {x.shape[1]} positional tokens, got {position.shape[1]}"
            )
        x = visual.ln_pre(visual.patch_dropout(x + position))
        return x.permute(1, 0, 2), grid

    def _pool_visual(self, x: torch.Tensor) -> torch.Tensor:
        visual = self.freeze_clip.visual
        pooled = visual.ln_post(x.permute(1, 0, 2)[:, 0])
        return pooled @ visual.proj if visual.proj is not None else pooled

    def _encode_unprompted_image(self, images: torch.Tensor) -> torch.Tensor:
        x, _ = self._visual_input(images)
        for block in self.freeze_clip.visual.transformer.resblocks:
            x = _run_block(block, x)
        return self._pool_visual(x)

    def _encode_prompted_image(
        self,
        images: torch.Tensor,
        dynamic_prompts: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], tuple[int, int]]:
        x, grid = self._visual_input(images)
        patch_tokens: list[torch.Tensor] = []
        for index, block in enumerate(self.freeze_clip.visual.transformer.resblocks):
            if self.visual_prompter.enabled and index < self.prompting_depth:
                context = (
                    self.visual_prompter.context(
                        index,
                        x.shape[1],
                        dynamic_prompts,
                    )
                    .permute(1, 0, 2)
                    .to(dtype=x.dtype)
                )
                x = (
                    torch.cat((x, context), dim=0)
                    if index == 0
                    else torch.cat((x[: -self.prompting_length], context), dim=0)
                )
            x = _run_block(block, x)
            if index + 1 in self.output_layers:
                vanilla = x[: -self.prompting_length] if self.visual_prompter.enabled else x
                patch_tokens.append(vanilla.permute(1, 0, 2))
        return self._pool_visual(x), patch_tokens, grid

    def _encode_text_tokens(
        self,
        tokens: torch.Tensor,
        dynamic_prompts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        model = self.freeze_clip
        dtype = model.token_embedding.weight.dtype
        x = model.token_embedding(tokens).to(dtype=dtype) + model.positional_embedding.to(
            dtype=dtype
        )
        x = x.permute(1, 0, 2)
        mask = getattr(model, "attn_mask", None)
        for index, block in enumerate(model.transformer.resblocks):
            # The released AdaCLIP path replaces positions after SOT from block 2 onward.
            if self.text_prompter.enabled and 0 < index < self.prompting_depth:
                context = (
                    self.text_prompter.context(
                        index,
                        x.shape[1],
                        dynamic_prompts,
                    )
                    .permute(1, 0, 2)
                    .to(dtype=x.dtype)
                )
                x = torch.cat((x[:1], context, x[1 + self.prompting_length :]), dim=0)
            x = _run_block(block, x, attn_mask=mask)
        x = model.ln_final(x.permute(1, 0, 2))
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, tokens.argmax(dim=-1)] @ model.text_projection

    def _text_features(
        self,
        class_name: str,
        dynamic_prompts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        normal_prompts, anomaly_prompts = _create_text_ensemble(class_name)
        features = []
        for prompts in (normal_prompts, anomaly_prompts):
            tokens = self.tokenizer(prompts).to(next(self.parameters()).device)
            current_dynamic = (
                None if dynamic_prompts is None else dynamic_prompts.expand(len(prompts), -1, -1)
            )
            encoded = _normalize(self._encode_text_tokens(tokens, current_dynamic))
            features.append(_normalize(encoded.mean(dim=0, keepdim=True)).squeeze(0))
        return torch.stack(features)

    def forward(
        self,
        images: torch.Tensor,
        class_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic_visual: Optional[torch.Tensor] = None
        dynamic_text: Optional[torch.Tensor] = None
        if "D" in self.prompting_type:
            with torch.no_grad():
                image_condition = self._encode_unprompted_image(images)
            dynamic_visual = self.dynamic_visual_prompt_generator(image_condition)
            dynamic_text = self.dynamic_text_prompt_generator(image_condition)

        image_features, patch_tokens, grid = self._encode_prompted_image(images, dynamic_visual)
        projected_patches = [_normalize(value) for value in self.patch_token_layer(patch_tokens)]
        projected_image = _normalize(self.cls_token_layer(image_features)[0])
        text_features = torch.stack(
            [
                self._text_features(
                    class_name,
                    None if dynamic_text is None else dynamic_text[index : index + 1],
                )
                for index in range(images.shape[0])
            ]
        )
        anomaly_logits = [
            PAPER_LOGIT_SCALE * torch.einsum("bld,bcd->blc", patches, text_features)
            for patches in projected_patches
        ]

        if self.use_hsf:
            clustered = _hybrid_semantic_fusion(
                projected_patches,
                anomaly_logits,
                k_clusters=self.k_clusters,
                random_state=self.random_state,
            )
            projected_image = _normalize(
                PAPER_HSF_ALPHA * clustered + (1.0 - PAPER_HSF_ALPHA) * projected_image
            )
        image_logits = PAPER_LOGIT_SCALE * torch.einsum(
            "bd,bcd->bc", projected_image, text_features
        )
        anomaly_score = image_logits.softmax(dim=-1)[:, 1]

        maps = []
        for logits in anomaly_logits:
            logits = logits.permute(0, 2, 1).reshape(-1, 2, grid[0], grid[1])
            maps.append(
                F.interpolate(
                    logits,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=True,
                )
            )
        probability = torch.stack(maps, dim=1).mean(dim=1).softmax(dim=1)
        anomaly_map = (probability[:, 1] + 1.0 - probability[:, 0]) / 2.0
        return anomaly_map, anomaly_score


def _read_checkpoint(path: str | Path) -> Mapping[str, torch.Tensor]:
    raw = safe_torch_load(path, map_location="cpu")
    if isinstance(raw, Mapping) and isinstance(raw.get("state_dict"), Mapping):
        raw = raw["state_dict"]
    if not isinstance(raw, Mapping):
        raise TypeError("AdaCLIP checkpoint must contain a tensor state mapping")
    state: dict[str, torch.Tensor] = {}
    for raw_name, value in raw.items():
        if not isinstance(raw_name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("AdaCLIP checkpoint must contain only named tensors")
        name = raw_name
        for prefix in ("module.", "clip_model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        state[name] = value
    return state


def _load_prompt_checkpoint(network: AdaCLIPNetwork, path: str | Path) -> None:
    state = dict(_read_checkpoint(path))
    model_keys = set(network.state_dict())
    expected = set(network.trainable_state_dict())
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - model_keys)
    if missing or unexpected:
        raise ValueError(
            "AdaCLIP checkpoint does not match the configured paper network: "
            f"missing={missing}, unexpected={unexpected}"
        )
    network.load_state_dict(state, strict=False)


class OpenCLIPAdaCLIPBackend:
    """Lazy OpenCLIP loader for the released AdaCLIP prompt checkpoints."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None,
        model_name: str = PAPER_MODEL,
        pretrained: str = PAPER_PRETRAINED,
        image_size: int = PAPER_IMAGE_SIZE,
        output_layers: Sequence[int] = PAPER_OUTPUT_LAYERS,
        prompting_depth: int = PAPER_PROMPT_DEPTH,
        prompting_length: int = PAPER_PROMPT_LENGTH,
        prompting_branch: str = "VL",
        prompting_type: str = "SD",
        use_hsf: bool = True,
        k_clusters: int = PAPER_K_CLUSTERS,
        random_state: Optional[int] = PAPER_RANDOM_STATE,
        device: str = "cuda",
        open_clip_module: Any = None,
        model: Optional[nn.Module] = None,
        preprocess: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.image_size = int(image_size)
        self.output_layers = tuple(int(layer) for layer in output_layers)
        self.prompting_depth = int(prompting_depth)
        self.prompting_length = int(prompting_length)
        self.prompting_branch = str(prompting_branch)
        self.prompting_type = str(prompting_type)
        self.use_hsf = bool(use_hsf)
        self.k_clusters = int(k_clusters)
        self.random_state = random_state
        requested = torch.device(device)
        self.device = (
            requested
            if requested.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._open_clip = open_clip_module
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.network: Optional[AdaCLIPNetwork] = None

    def initialize(self) -> "OpenCLIPAdaCLIPBackend":
        if self.network is not None:
            return self
        if self.checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required: AdaCLIP prompts are trained on annotated auxiliary data"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"AdaCLIP checkpoint not found: {self.checkpoint_path}")
        if self.model is None or self.preprocess is None:
            self._open_clip = self._open_clip or require(
                "open_clip",
                extra="clip",
                purpose="AdaCLIP's frozen OpenAI CLIP backbone",
            )
            self.model, self.preprocess, self.device = _load_openclip_model_and_preprocess(
                open_clip_module=self._open_clip,
                model_name=self.model_name,
                pretrained=self.pretrained,
                device=str(self.device),
                force_image_size=self.image_size,
            )
        else:
            self.model = self.model.to(self.device).eval()
        if self.tokenizer is None:
            self._open_clip = self._open_clip or require(
                "open_clip",
                extra="clip",
                purpose="AdaCLIP text tokenization",
            )
            self.tokenizer = self._open_clip.get_tokenizer(self.model_name)
        self.network = AdaCLIPNetwork(
            self.model,
            image_size=self.image_size,
            output_layers=self.output_layers,
            prompting_depth=self.prompting_depth,
            prompting_length=self.prompting_length,
            prompting_branch=self.prompting_branch,
            prompting_type=self.prompting_type,
            use_hsf=self.use_hsf,
            k_clusters=self.k_clusters,
            random_state=self.random_state,
            tokenizer=self.tokenizer,
        ).to(self.device)
        _load_prompt_checkpoint(self.network, self.checkpoint_path)
        self.network.eval()
        return self

    def _preprocess_image(self, image: NDArray[Any]) -> torch.Tensor:
        self.initialize()
        from PIL import Image

        from pyimgano.utils.image_ops import Resampling

        array = np.asarray(image)
        if not np.isfinite(array).all():
            raise ValueError("AdaCLIP images must contain only finite values")
        if float(array.min()) < 0 or float(array.max()) > 255:
            raise ValueError("AdaCLIP image values must be in [0, 1] or [0, 255]")
        if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array, mode="RGB").resize(
            (self.image_size, self.image_size),
            Resampling.BICUBIC,
        )
        return self.preprocess(pil).unsqueeze(0).to(self.device)

    def score_image(
        self,
        image: NDArray[Any],
        class_name: str,
    ) -> tuple[float, NDArray[np.float32]]:
        batch = self._preprocess_image(image)
        if self.network is None:  # pragma: no cover - initialize guards this
            raise RuntimeError("AdaCLIP network is not initialized")
        amp_context = torch.cuda.amp.autocast if self.device.type == "cuda" else nullcontext
        with torch.inference_mode(), amp_context():
            anomaly_map, anomaly_score = self.network(batch, class_name)
        return (
            float(anomaly_score.item()),
            anomaly_map[0].float().cpu().numpy().astype(np.float32, copy=False),
        )


@register_model(
    "vision_adaclip",
    tags=(
        "vision",
        "deep",
        "clip",
        "openclip",
        "zero-shot",
        "prompt",
        "pixel_map",
        "adaclip",
        "eccv2024",
    ),
    metadata={
        "description": "Native AdaCLIP hybrid visual/text prompt and HSF inference path",
        "paper": "AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection",
        "paper_url": "https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5096_ECCV_2024_paper.php",
        "official_code_url": "https://github.com/caoyunkang/AdaCLIP",
        "official_weights_url": "https://huggingface.co/spaces/Caoyunkang/AdaCLIP/tree/main/weights",
        "year": 2024,
        "conference": "ECCV",
        "implementation_status": "native-paper-inference-openclip-adaptation",
        "paper_fidelity": "paper-adaptation",
        "supervision": "zero-shot",
        "supports_pixel_map": True,
        "requires_checkpoint": True,
        "weights_source": "OpenAI CLIP ViT-L/14@336 plus official AdaCLIP prompt checkpoint",
    },
)
class VisionAdaCLIP:
    """AdaCLIP detector using an official auxiliary-trained prompt checkpoint.

    ``fit`` calibrates a deployment threshold; it does not replace the paper's
    annotated auxiliary-data training protocol.
    """

    def __init__(
        self,
        *,
        class_name: str = "object",
        checkpoint_path: str | Path | None = None,
        openclip_model_name: str = PAPER_MODEL,
        openclip_pretrained: str = PAPER_PRETRAINED,
        image_size: int = PAPER_IMAGE_SIZE,
        output_layers: Sequence[int] = PAPER_OUTPUT_LAYERS,
        prompting_depth: int = PAPER_PROMPT_DEPTH,
        prompting_length: int = PAPER_PROMPT_LENGTH,
        prompting_branch: str = "VL",
        prompting_type: str = "SD",
        use_hsf: bool = True,
        k_clusters: int = PAPER_K_CLUSTERS,
        gaussian_sigma: float = PAPER_GAUSSIAN_SIGMA,
        contamination: float = 0.1,
        random_state: Optional[int] = PAPER_RANDOM_STATE,
        device: Optional[str] = None,
        backend: Any = None,
        open_clip_module: Any = None,
    ) -> None:
        self.class_name = str(class_name)
        _create_text_ensemble(self.class_name)
        self.gaussian_sigma = float(gaussian_sigma)
        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")
        self.contamination = float(contamination)
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend or OpenCLIPAdaCLIPBackend(
            checkpoint_path=checkpoint_path,
            model_name=openclip_model_name,
            pretrained=openclip_pretrained,
            image_size=image_size,
            output_layers=output_layers,
            prompting_depth=prompting_depth,
            prompting_length=prompting_length,
            prompting_branch=prompting_branch,
            prompting_type=prompting_type,
            use_hsf=use_hsf,
            k_clusters=k_clusters,
            random_state=random_state,
            device=resolved_device,
            open_clip_module=open_clip_module,
        )
        self.openclip_model_name = str(openclip_model_name)
        self.openclip_pretrained = str(openclip_pretrained)
        self.image_size = int(image_size)
        self.output_layers = tuple(int(layer) for layer in output_layers)
        self.prompting_depth = int(prompting_depth)
        self.prompting_length = int(prompting_length)
        self.prompting_branch = str(prompting_branch)
        self.prompting_type = str(prompting_type)
        self.use_hsf = bool(use_hsf)
        self.k_clusters = int(k_clusters)

    @staticmethod
    def _coerce_images(x: Any) -> list[NDArray[np.uint8]]:
        if isinstance(x, (str, Path)) or (isinstance(x, np.ndarray) and x.ndim in (2, 3)):
            items = [x]
        elif isinstance(x, np.ndarray) and x.ndim == 4:
            items = list(x)
        else:
            items = list(x)
        if not items:
            raise ValueError("AdaCLIP requires at least one image")
        images = []
        for item in items:
            array = np.asarray(_coerce_single_rgb_image(item))
            if not np.isfinite(array).all():
                raise ValueError("AdaCLIP images must contain only finite values")
            if float(array.min()) < 0 or float(array.max()) > 255:
                raise ValueError("AdaCLIP image values must be in [0, 1] or [0, 255]")
            if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
                array = array * 255.0
            images.append(np.clip(array, 0, 255).astype(np.uint8))
        return images

    def set_class_name(self, class_name: str) -> "VisionAdaCLIP":
        _create_text_ensemble(class_name)
        self.class_name = str(class_name)
        return self

    def _score_image(self, image: NDArray[np.uint8]) -> tuple[float, NDArray[np.float32]]:
        if hasattr(self.backend, "initialize"):
            self.backend.initialize()
        score, anomaly_map = self.backend.score_image(image, self.class_name)
        result = np.asarray(anomaly_map, dtype=np.float32)
        if self.gaussian_sigma:
            result = gaussian_filter(result, sigma=self.gaussian_sigma).astype(
                np.float32,
                copy=False,
            )
        if result.shape != image.shape[:2]:
            result = (
                F.interpolate(
                    torch.from_numpy(result).reshape(1, 1, *result.shape),
                    size=tuple(int(value) for value in image.shape[:2]),
                    mode="bilinear",
                    align_corners=True,
                )
                .reshape(*image.shape[:2])
                .numpy()
                .astype(np.float32, copy=False)
            )
        return float(score), result

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray[Any]] = None,
        **kwargs: object,
    ) -> "VisionAdaCLIP":
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        self.decision_scores_ = self.decision_function(x_value)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        self.is_fitted_ = True
        del y
        return self

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        del batch_size
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        return np.asarray(
            [self._score_image(image)[0] for image in self._coerce_images(x_value)],
            dtype=np.float64,
        )

    def predict(
        self,
        x: object = MISSING,
        **kwargs: object,
    ) -> NDArray[np.int64]:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        return (self.decision_function(x_value) > float(self.threshold_)).astype(np.int64)

    def predict_anomaly_map(
        self,
        x: object = MISSING,
        **kwargs: object,
    ) -> list[NDArray[np.float32]]:
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        return [self._score_image(image)[1] for image in self._coerce_images(x_value)]

    def get_anomaly_map(
        self,
        x: object = MISSING,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="get_anomaly_map")
        return self.predict_anomaly_map(x_value)[0]
