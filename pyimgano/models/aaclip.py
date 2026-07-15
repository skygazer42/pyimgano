"""AA-CLIP paper inference on an OpenAI CLIP ViT-L/14@336 backbone."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import require

from ._image_batch import _coerce_single_rgb_image
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .adaclip import _run_block
from .openclip_backend import _load_openclip_model_and_preprocess
from .registry import register_model

PAPER_MODEL = "ViT-L-14-336"
PAPER_PRETRAINED = "openai"
PAPER_IMAGE_SIZE = 518
PAPER_OUTPUT_LAYERS = (6, 12, 18, 24)
PAPER_TEXT_ADAPT_UNTIL = 3
PAPER_IMAGE_ADAPT_UNTIL = 6
PAPER_TEXT_ADAPT_WEIGHT = 0.1
PAPER_IMAGE_ADAPT_WEIGHT = 0.1
PAPER_LOGIT_SCALE = 100.0

NORMAL_STATES = ("{}", "a {}", "the {}")
ANOMALY_STATES = (
    "a damaged {}",
    "a broken {}",
    "a {} with flaw",
    "a {} with defect",
    "a {} with damage",
)
PROMPT_TEMPLATES = ("{}.", "a photo of {}.")

# Class descriptions released with the authors' evaluation code. Unknown names
# remain usable as literal class descriptions.
_OFFICIAL_CLASS_DESCRIPTIONS = {
    "brain": "scan",
    "liver": "scan",
    "retina": "scan",
    "colon_clinicdb": "colon endoscopy image",
    "colon_colondb": "colon endoscopy image",
    "colon_cvc300": "colon endoscopy image",
    "cvc_300": "colon endoscopy image",
    "colon_kvasir": "colon endoscopy image",
    "kvasir": "colon endoscopy image",
    "bottle": "dark bottle",
    "cable": "top view of three cables",
    "capsule": "black and orange capsule",
    "carpet": "gray carpet",
    "grid": "metal or plastic mesh",
    "hazelnut": "single brown hazelnut",
    "leather": "brown leather",
    "metal_nut": "metal nut which has four notched edges",
    "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
    "screw": "screw",
    "tile": "speckled tile surface",
    "transistor": "a three-legged transistor placed vertically",
    "toothbrush": "toothbrush head",
    "wood": "wood surface",
    "zipper": "a black zipper",
    "candle": "candle",
    "pcb3": "infrared sensor pcb module",
    "capsules": "capsules",
    "pipe_fryum": "pipe-shaped fryum",
    "pcb4": "battery charging pcb module",
    "macaroni2": "scattered yellow macaroni",
    "pcb2": "integrated circuits board",
    "chewinggum": "chewing gum",
    "macaroni1": "orange macaroni",
    "cashew": "cashew nut",
    "fryum": "wheel-shaped fryum snack",
    "pcb1": "dual ultrasonic distance sensor pcb module",
    "connector": "metal clamps with black adjustment knobs",
    "tubes": "scattered metal objects",
    "metal_plate": "blue rectangular metal plate with a notch on one side",
    "bracket_white": "white, elongated triangular metal bracket with a smooth, matte finish",
    "bracket_brown": (
        "brown L-shaped metal bracket with smooth, glossy finish and multiple mounting "
        "holes along its arms"
    ),
    "bracket_black": (
        "black ornamental metal bracket with spiral design attached to a rectangular frame"
    ),
    "01": (
        "Bright concentric rings in neon yellow and blue tones against a dark blue background, "
        "resembling a stylized wave or energy field radiating outward."
    ),
    "02": "vertical fabric lines in warm, dusty pink and beige tones",
    "03": "oval concentric circular rings in gradient shades of blue and white",
}


def _normalize_class_key(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _resolve_class_description(
    class_name: str,
    class_description: Optional[str] = None,
) -> str:
    raw = str(class_name if class_description is None else class_description).strip()
    if not raw:
        raise ValueError("class_name/class_description must be non-empty")
    if class_description is not None:
        return " ".join(raw.split())
    key = _normalize_class_key(raw)
    return _OFFICIAL_CLASS_DESCRIPTIONS.get(key, key.replace("_", " "))


def _create_text_ensemble(
    class_name: str,
    class_description: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    description = _resolve_class_description(class_name, class_description)

    def expand(states: Sequence[str]) -> list[str]:
        return [
            template.format(state.format(description))
            for state in states
            for template in PROMPT_TEMPLATES
        ]

    return expand(NORMAL_STATES), expand(ANOMALY_STATES)


class _SimpleAdapter(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channels, channels, bias=False), nn.LeakyReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _SimpleProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, relu: bool) -> None:
        super().__init__()
        linear = nn.Linear(input_dim, output_dim, bias=False)
        self.fc = nn.Sequential(linear, nn.LeakyReLU()) if relu else linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _residual_adapt(
    original: torch.Tensor,
    adapted: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    adapted = adapted * original.norm(dim=-1, keepdim=True) / adapted.norm(dim=-1, keepdim=True)
    return float(weight) * adapted + (1.0 - float(weight)) * original


def _gaussian_blur2d(
    value: torch.Tensor,
    *,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """Kornia-compatible separable Gaussian blur used by the authors."""

    radius = int(kernel_size) // 2
    coordinates = torch.arange(int(kernel_size), device=value.device, dtype=value.dtype) - float(
        radius
    )
    kernel = torch.exp(-(coordinates.square()) / (2.0 * float(sigma) ** 2))
    kernel = kernel / kernel.sum()
    channels = int(value.shape[1])
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    value = F.conv2d(
        F.pad(value, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(value, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=channels,
    )


class AAClipNetwork(nn.Module):
    """Residual text/visual adapters and projections from the AA-CLIP release."""

    def __init__(
        self,
        clip_model: nn.Module,
        *,
        tokenizer: Any,
        image_size: int = PAPER_IMAGE_SIZE,
        output_layers: Sequence[int] = PAPER_OUTPUT_LAYERS,
        text_adapt_until: int = PAPER_TEXT_ADAPT_UNTIL,
        image_adapt_until: int = PAPER_IMAGE_ADAPT_UNTIL,
        text_adapt_weight: float = PAPER_TEXT_ADAPT_WEIGHT,
        image_adapt_weight: float = PAPER_IMAGE_ADAPT_WEIGHT,
        relu: bool = False,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.image_size = int(image_size)
        self.output_layers = tuple(int(layer) for layer in output_layers)
        self.text_adapt_until = int(text_adapt_until)
        self.image_adapt_until = int(image_adapt_until)
        self.text_adapt_weight = float(text_adapt_weight)
        self.image_adapt_weight = float(image_adapt_weight)
        self.relu = bool(relu)
        self._validate_backbone()

        visual_width = int(self.clip_model.visual.conv1.out_channels)
        text_width = int(self.clip_model.token_embedding.embedding_dim)
        embed_dim = int(self.clip_model.text_projection.shape[-1])
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": nn.ModuleList(
                    [_SimpleAdapter(visual_width) for _ in range(self.image_adapt_until)]
                ),
                "seg_proj": nn.ModuleList(
                    [
                        _SimpleProjection(visual_width, embed_dim, relu=self.relu)
                        for _ in self.output_layers
                    ]
                ),
                "det_proj": _SimpleProjection(visual_width, embed_dim, relu=self.relu),
            }
        )
        self.text_adapter = nn.ModuleList(
            [_SimpleAdapter(text_width) for _ in range(self.text_adapt_until)]
            + [_SimpleProjection(text_width, embed_dim, relu=True)]
        )
        for parameter in self.image_adapter.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        for parameter in self.text_adapter.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def _validate_backbone(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= self.text_adapt_weight <= 1.0:
            raise ValueError("text_adapt_weight must be in [0, 1]")
        if not 0.0 <= self.image_adapt_weight <= 1.0:
            raise ValueError("image_adapt_weight must be in [0, 1]")
        required = (
            "visual",
            "token_embedding",
            "positional_embedding",
            "transformer",
            "ln_final",
            "text_projection",
        )
        missing = [name for name in required if not hasattr(self.clip_model, name)]
        visual = getattr(self.clip_model, "visual", None)
        if visual is None:
            raise TypeError("AA-CLIP requires a classic OpenCLIP ViT; missing visual")
        visual_required = (
            "conv1",
            "class_embedding",
            "positional_embedding",
            "ln_pre",
            "transformer",
            "ln_post",
        )
        missing.extend(f"visual.{name}" for name in visual_required if not hasattr(visual, name))
        if missing:
            raise TypeError(
                "AA-CLIP requires a classic OpenCLIP ViT; missing " + ", ".join(missing)
            )
        visual_blocks = getattr(visual.transformer, "resblocks", None)
        text_blocks = getattr(self.clip_model.transformer, "resblocks", None)
        if visual_blocks is None or text_blocks is None:
            raise TypeError("AA-CLIP requires OpenCLIP transformer.resblocks")
        if (
            not self.output_layers
            or tuple(sorted(set(self.output_layers))) != self.output_layers
            or min(self.output_layers) <= 0
            or max(self.output_layers) > len(visual_blocks)
        ):
            raise ValueError("output_layers must be unique increasing 1-based visual block indexes")
        if not 0 <= self.image_adapt_until <= len(visual_blocks):
            raise ValueError("image_adapt_until exceeds the visual transformer depth")
        if not 0 <= self.text_adapt_until <= len(text_blocks):
            raise ValueError("text_adapt_until exceeds the text transformer depth")
        if getattr(visual, "input_patchnorm", False):
            raise TypeError("AA-CLIP's published ViT-L/14 path does not use input patch norm")

    def _visual_input(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.clip_model.visual
        x = visual.conv1(images.to(dtype=visual.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(dtype=x.dtype).reshape(1, 1, -1)
        x = torch.cat((class_token.expand(x.shape[0], -1, -1), x), dim=1)
        position = visual.positional_embedding.to(dtype=x.dtype)
        if position.ndim == 2:
            position = position.unsqueeze(0)
        if position.shape[1] != x.shape[1]:
            raise RuntimeError(
                f"AA-CLIP expected {x.shape[1]} positional tokens, got {position.shape[1]}"
            )
        x = x + position
        patch_dropout = getattr(visual, "patch_dropout", None)
        if patch_dropout is not None:
            x = patch_dropout(x)
        return visual.ln_pre(x).permute(1, 0, 2)

    def encode_image(self, images: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        visual = self.clip_model.visual
        x = self._visual_input(images)
        tokens: list[torch.Tensor] = []
        for index, block in enumerate(visual.transformer.resblocks):
            x = _run_block(block, x)
            if index < self.image_adapt_until:
                x = _residual_adapt(
                    x,
                    self.image_adapter["layer_adapters"][index](x),
                    self.image_adapt_weight,
                )
            if index + 1 in self.output_layers:
                tokens.append(x[1:].permute(1, 0, 2))
        tokens = [visual.ln_post(token) for token in tokens]
        segmentation = [
            F.normalize(self.image_adapter["seg_proj"][index](token), dim=-1)
            for index, token in enumerate(tokens)
        ]
        detection = F.normalize(self.image_adapter["det_proj"](tokens[-1]), dim=-1).mean(1)
        return segmentation, detection

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        model = self.clip_model
        dtype = model.token_embedding.weight.dtype
        x = model.token_embedding(tokens).to(dtype=dtype)
        x = x + model.positional_embedding.to(dtype=dtype)
        x = x.permute(1, 0, 2)
        mask = getattr(model, "attn_mask", None)
        for index, block in enumerate(model.transformer.resblocks):
            x = _run_block(block, x, attn_mask=mask)
            if index < self.text_adapt_until:
                x = _residual_adapt(
                    x,
                    self.text_adapter[index](x),
                    self.text_adapt_weight,
                )
        x = model.ln_final(x.permute(1, 0, 2))
        rows = torch.arange(x.shape[0], device=x.device)
        return self.text_adapter[-1](x[rows, tokens.argmax(dim=-1)])

    def encode_text_anchors(self, class_description: str) -> torch.Tensor:
        normal_prompts, anomaly_prompts = _create_text_ensemble(
            class_description,
            class_description=class_description,
        )
        anchors = []
        device = next(self.parameters()).device
        for prompts in (normal_prompts, anomaly_prompts):
            tokens = self.tokenizer(prompts).to(device)
            embeddings = F.normalize(self.encode_text(tokens), dim=-1)
            anchors.append(F.normalize(embeddings.mean(dim=0), dim=0))
        return torch.stack(anchors, dim=1)

    def forward(
        self,
        images: torch.Tensor,
        text_anchors: torch.Tensor,
        *,
        domain: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        domain_key = str(domain).strip().lower()
        if domain_key not in {"industrial", "medical"}:
            raise ValueError("domain must be 'industrial' or 'medical'")
        if text_anchors.ndim != 2 or text_anchors.shape[1] != 2:
            raise ValueError("text_anchors must have shape (embedding_dim, 2)")

        patch_features, detection = self.encode_image(images)
        image_prediction = detection @ text_anchors
        image_score = (image_prediction[:, 1] + 1.0) / 2.0
        kernel_size, sigma = (7, 1.0) if domain_key == "industrial" else (9, 1.5)

        maps = []
        for features in patch_features:
            logits = PAPER_LOGIT_SCALE * (features @ text_anchors)
            side = int(np.sqrt(int(logits.shape[1])))
            if side * side != logits.shape[1]:
                raise RuntimeError("AA-CLIP requires a square visual patch grid")
            logits = logits.permute(0, 2, 1).reshape(-1, 2, side, side)
            anomaly = (logits[:, 1] + 1.0 - logits[:, 0]).unsqueeze(1) / 2.0
            anomaly = _gaussian_blur2d(anomaly, kernel_size=kernel_size, sigma=sigma)
            maps.append(
                F.interpolate(
                    anomaly,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=True,
                )[:, 0]
            )
        return torch.stack(maps, dim=0).sum(dim=0), image_score


def _read_author_state(path: Path, key: str) -> Mapping[str, torch.Tensor]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch < 2.0
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping) or not isinstance(raw.get(key), Mapping):
        raise TypeError(f"AA-CLIP checkpoint {path} must contain a '{key}' state mapping")
    state = cast(Mapping[Any, Any], raw[key])
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()
    ):
        raise TypeError(f"AA-CLIP checkpoint {path} contains a non-tensor state entry")
    return cast(Mapping[str, torch.Tensor], state)


def _load_author_checkpoints(
    network: AAClipNetwork,
    *,
    text_path: Path,
    image_path: Path,
) -> None:
    for module, path, key in (
        (network.text_adapter, text_path, "text_adapter"),
        (network.image_adapter, image_path, "image_adapter"),
    ):
        state = _read_author_state(path, key)
        expected = set(module.state_dict())
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        if missing or unexpected:
            raise ValueError(
                "AA-CLIP checkpoint does not match the configured paper network: "
                f"missing={missing}, unexpected={unexpected}"
            )
        module.load_state_dict(state, strict=True)


class OpenCLIPAAClipBackend:
    """Lazy loader for author-format AA-CLIP text and image adapter checkpoints."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None,
        model_name: str = PAPER_MODEL,
        pretrained: str = PAPER_PRETRAINED,
        image_size: int = PAPER_IMAGE_SIZE,
        output_layers: Sequence[int] = PAPER_OUTPUT_LAYERS,
        text_adapt_until: int = PAPER_TEXT_ADAPT_UNTIL,
        image_adapt_until: int = PAPER_IMAGE_ADAPT_UNTIL,
        text_adapt_weight: float = PAPER_TEXT_ADAPT_WEIGHT,
        image_adapt_weight: float = PAPER_IMAGE_ADAPT_WEIGHT,
        relu: bool = False,
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
        self.text_adapt_until = int(text_adapt_until)
        self.image_adapt_until = int(image_adapt_until)
        self.text_adapt_weight = float(text_adapt_weight)
        self.image_adapt_weight = float(image_adapt_weight)
        self.relu = bool(relu)
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
        self.network: Optional[AAClipNetwork] = None
        self._text_cache: dict[str, torch.Tensor] = {}

    def initialize(self) -> "OpenCLIPAAClipBackend":
        if self.network is not None:
            return self
        if self.checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required: AA-CLIP adapters require the paper's "
                "annotated two-stage auxiliary training"
            )
        if not self.checkpoint_path.is_dir():
            raise FileNotFoundError(
                "AA-CLIP checkpoint_path must be an author-format directory containing "
                "text_adapter.pth and image_adapter.pth"
            )
        text_path = self.checkpoint_path / "text_adapter.pth"
        image_path = self.checkpoint_path / "image_adapter.pth"
        for path in (text_path, image_path):
            if not path.is_file():
                raise FileNotFoundError(f"AA-CLIP checkpoint not found: {path}")

        if self.model is None or self.preprocess is None:
            self._open_clip = self._open_clip or require(
                "open_clip",
                extra="clip",
                purpose="AA-CLIP's frozen OpenAI CLIP backbone",
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
                "open_clip", extra="clip", purpose="AA-CLIP text tokenization"
            )
            self.tokenizer = self._open_clip.get_tokenizer(self.model_name)
        self.network = AAClipNetwork(
            self.model,
            tokenizer=self.tokenizer,
            image_size=self.image_size,
            output_layers=self.output_layers,
            text_adapt_until=self.text_adapt_until,
            image_adapt_until=self.image_adapt_until,
            text_adapt_weight=self.text_adapt_weight,
            image_adapt_weight=self.image_adapt_weight,
            relu=self.relu,
        ).to(self.device)
        _load_author_checkpoints(self.network, text_path=text_path, image_path=image_path)
        self.network.eval()
        return self

    def _preprocess_image(self, image: NDArray[Any]) -> torch.Tensor:
        self.initialize()
        from PIL import Image

        from pyimgano.utils.image_ops import Resampling

        array = np.asarray(image)
        if not np.isfinite(array).all():
            raise ValueError("AA-CLIP images must contain only finite values")
        if float(array.min()) < 0 or float(array.max()) > 255:
            raise ValueError("AA-CLIP image values must be in [0, 1] or [0, 255]")
        if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array, mode="RGB").resize(
            (self.image_size, self.image_size), Resampling.BICUBIC
        )
        return self.preprocess(pil).unsqueeze(0).to(self.device)

    def score_image(
        self,
        image: NDArray[Any],
        class_description: str,
        domain: str,
    ) -> tuple[float, NDArray[np.float32]]:
        batch = self._preprocess_image(image)
        if self.network is None:  # pragma: no cover - initialize guards this
            raise RuntimeError("AA-CLIP network is not initialized")
        with torch.inference_mode():
            anchors = self._text_cache.get(class_description)
            if anchors is None:
                anchors = self.network.encode_text_anchors(class_description)
                self._text_cache[class_description] = anchors
            anomaly_map, anomaly_score = self.network(batch, anchors, domain=domain)
        return (
            float(anomaly_score.item()),
            anomaly_map[0].float().cpu().numpy().astype(np.float32, copy=False),
        )


@register_model(
    "vision_aaclip",
    tags=(
        "vision",
        "deep",
        "clip",
        "openclip",
        "pixel_map",
        "zero-shot",
        "prompt",
        "aaclip",
        "cvpr2025",
    ),
    metadata={
        "description": "Native AA-CLIP residual text/visual adapter inference path",
        "paper": "AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2025/html/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.html",
        "official_code_url": "https://github.com/Mwxinnn/AA-CLIP",
        "year": 2025,
        "conference": "CVPR",
        "implementation_status": "native-paper-inference-openclip-adaptation",
        "paper_fidelity": "paper-adaptation",
        "supervision": "zero-shot",
        "supports_pixel_map": True,
        "requires_checkpoint": True,
        "weights_source": (
            "OpenAI CLIP ViT-L/14@336 plus user-trained author-format "
            "text_adapter.pth and image_adapter.pth"
        ),
    },
)
class VisionAAClip:
    """AA-CLIP inference with author-format two-stage adapter checkpoints.

    ``fit`` only calibrates a deployment threshold. It does not replace the
    paper's supervised two-stage training on an auxiliary anomaly dataset.
    """

    def __init__(
        self,
        *,
        class_name: str = "object",
        class_description: Optional[str] = None,
        domain: str = "industrial",
        checkpoint_path: str | Path | None = None,
        openclip_model_name: str = PAPER_MODEL,
        openclip_pretrained: str = PAPER_PRETRAINED,
        image_size: int = PAPER_IMAGE_SIZE,
        output_layers: Sequence[int] = PAPER_OUTPUT_LAYERS,
        text_adapt_until: int = PAPER_TEXT_ADAPT_UNTIL,
        image_adapt_until: int = PAPER_IMAGE_ADAPT_UNTIL,
        text_adapt_weight: float = PAPER_TEXT_ADAPT_WEIGHT,
        image_adapt_weight: float = PAPER_IMAGE_ADAPT_WEIGHT,
        relu: bool = False,
        contamination: float = 0.1,
        device: Optional[str] = None,
        backend: Any = None,
        open_clip_module: Any = None,
    ) -> None:
        self.class_name = str(class_name)
        self.class_description = _resolve_class_description(class_name, class_description)
        self.domain = str(domain).strip().lower()
        if self.domain not in {"industrial", "medical"}:
            raise ValueError("domain must be 'industrial' or 'medical'")
        self.contamination = float(contamination)
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend or OpenCLIPAAClipBackend(
            checkpoint_path=checkpoint_path,
            model_name=openclip_model_name,
            pretrained=openclip_pretrained,
            image_size=image_size,
            output_layers=output_layers,
            text_adapt_until=text_adapt_until,
            image_adapt_until=image_adapt_until,
            text_adapt_weight=text_adapt_weight,
            image_adapt_weight=image_adapt_weight,
            relu=relu,
            device=resolved_device,
            open_clip_module=open_clip_module,
        )
        self.openclip_model_name = str(openclip_model_name)
        self.openclip_pretrained = str(openclip_pretrained)
        self.image_size = int(image_size)
        self.output_layers = tuple(int(layer) for layer in output_layers)
        self.text_adapt_until = int(text_adapt_until)
        self.image_adapt_until = int(image_adapt_until)
        self.text_adapt_weight = float(text_adapt_weight)
        self.image_adapt_weight = float(image_adapt_weight)
        self.relu = bool(relu)

    @staticmethod
    def _coerce_images(x: Any) -> list[NDArray[np.uint8]]:
        if isinstance(x, (str, Path)) or (isinstance(x, np.ndarray) and x.ndim in (2, 3)):
            items = [x]
        elif isinstance(x, np.ndarray) and x.ndim == 4:
            items = list(x)
        else:
            items = list(x)
        if not items:
            raise ValueError("AA-CLIP requires at least one image")
        images = []
        for item in items:
            array = np.asarray(_coerce_single_rgb_image(item))
            if not np.isfinite(array).all():
                raise ValueError("AA-CLIP images must contain only finite values")
            if float(array.min()) < 0 or float(array.max()) > 255:
                raise ValueError("AA-CLIP image values must be in [0, 1] or [0, 255]")
            if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
                array = array * 255.0
            images.append(np.clip(array, 0, 255).astype(np.uint8))
        return images

    def set_class_name(
        self,
        class_name: str,
        *,
        class_description: Optional[str] = None,
    ) -> "VisionAAClip":
        self.class_name = str(class_name)
        self.class_description = _resolve_class_description(class_name, class_description)
        return self

    def _score_image(self, image: NDArray[np.uint8]) -> tuple[float, NDArray[np.float32]]:
        if hasattr(self.backend, "initialize"):
            self.backend.initialize()
        score, anomaly_map = self.backend.score_image(image, self.class_description, self.domain)
        result = np.asarray(anomaly_map, dtype=np.float32)
        if result.ndim != 2 or not np.isfinite(result).all():
            raise ValueError("AA-CLIP backend must return a finite 2D anomaly map")
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

    def fit(self, x: object = MISSING, _y=None, **kwargs: object) -> "VisionAAClip":
        del _y
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        self.decision_scores_ = self.decision_function(x_value)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        self.is_fitted_ = True
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

    def predict(self, x: object = MISSING, **kwargs: object) -> NDArray[np.int64]:
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
