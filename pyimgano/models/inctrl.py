"""Native InCTRL inference with the authors' frozen OpenCLIP architecture."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from ._batch_size import validate_batch_size
from ._image_batch import _coerce_single_rgb_image
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .deep_io import safe_torch_load
from .openclip_backend import _load_openclip_model_and_preprocess
from .registry import register_model
from .winclip import ANOMALOUS_STATES, NORMAL_STATES

PAPER_MODEL_NAME = "ViT-B-16-plus-240"
PAPER_PRETRAINED = "laion400m_e32"
PAPER_IMAGE_SIZE = 240
PAPER_FEATURE_LAYERS = (7, 9, 11)
PAPER_PATCH_GRID = (15, 15)
PAPER_GLOBAL_DIM = 640
PAPER_VISUAL_WIDTH = 896
PAPER_ADAPTER_REDUCTION = 4
PAPER_PATCH_DISTANCE_SCALE = 0.5
PAPER_TEXT_LOGIT_SCALE = 100.0
PAPER_TRAIN_EPOCHS = 10
PAPER_TRAIN_BATCH_SIZE = 48
PAPER_TRAIN_LEARNING_RATE = 1e-3
PAPER_SHOTS = (2, 4, 8)

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() with normal image prompts first."

_NATURAL_CLASSES = frozenset(
    {
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
        "animal",
    }
)

_PROMPT_TEMPLATES = (
    "a cropped photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of the {}.",
    "a photo of a {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for anomaly detection.",
    "a photo of the {} for anomaly detection.",
)


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, (str, Path)):
        return [value]
    if isinstance(value, np.ndarray) and value.ndim in (2, 3):
        return [value]
    return list(cast(Iterable[Any], value))


def _as_rgb_uint8(image: Any) -> NDArray[np.uint8]:
    array = np.asarray(_coerce_single_rgb_image(image))
    if not np.isfinite(array).all():
        raise ValueError("InCTRL images must contain only finite values.")
    minimum, maximum = float(array.min()), float(array.max())
    if minimum < 0 or maximum > 255:
        raise ValueError("InCTRL image values must be in [0, 1] or [0, 255].")
    if np.issubdtype(array.dtype, np.floating) and maximum <= 1.0:
        array = array * 255.0
    return np.ascontiguousarray(np.rint(array).astype(np.uint8))


def _prompt_ensemble(class_name: str) -> tuple[list[str], list[str]]:
    object_name = " ".join(str(class_name).replace("_", " ").split())
    if not object_name:
        raise ValueError("class_name must be non-empty.")
    if object_name in _NATURAL_CLASSES:
        return (
            [f"a photo of {object_name} for anomaly detection."],
            [f"a photo without {object_name} for anomaly detection."],
        )
    normal_states = [state.format(object_name) for state in NORMAL_STATES]
    anomaly_states = [state.format(object_name) for state in ANOMALOUS_STATES]
    return (
        [template.format(state) for state in normal_states for template in _PROMPT_TEMPLATES],
        [template.format(state) for state in anomaly_states for template in _PROMPT_TEMPLATES],
    )


class InCTRLAdapter(nn.Module):
    """Released 640 -> 160 -> 640 residual adapter."""

    def __init__(
        self,
        dimension: int = PAPER_GLOBAL_DIM,
        reduction: int = PAPER_ADAPTER_REDUCTION,
    ) -> None:
        super().__init__()
        if dimension <= 0 or reduction <= 0 or dimension % reduction:
            raise ValueError("dimension must be positive and divisible by reduction.")
        self.fc = nn.Sequential(
            nn.Linear(dimension, dimension // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dimension // reduction, dimension, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc(values)


class InCTRLScoreHead(nn.Module):
    """Released 128/64 sigmoid score head, including its checkpoint keys."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.projection1 = nn.Linear(dimension, 128, bias=True)
        self.projection2 = nn.Linear(128, 64, bias=True)
        self.projection3 = nn.Linear(64, 1, bias=True)
        # bn1 exists in the author checkpoint but is not called by its forward.
        self.bn1 = nn.BatchNorm1d(dimension)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.bn2(F.relu(self.projection1(values), inplace=True))
        values = self.bn3(F.relu(self.projection2(values), inplace=True))
        return torch.sigmoid(self.projection3(values))


class InCTRLHeads(nn.Module):
    """All trainable modules in the released InCTRL checkpoint."""

    def __init__(self, global_dimension: int = PAPER_GLOBAL_DIM) -> None:
        super().__init__()
        self.adapter = InCTRLAdapter(global_dimension)
        self.diff_head = InCTRLScoreHead(PAPER_PATCH_GRID[0] * PAPER_PATCH_GRID[1])
        self.diff_head_ref = InCTRLScoreHead(global_dimension)


def _checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("InCTRL checkpoint must contain a state mapping.")
    for key in ("state_dict", "model_state_dict", "model_state", "model"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            payload = nested
            break
    state: dict[str, torch.Tensor] = {}
    for raw_name, value in payload.items():
        if not isinstance(raw_name, str) or not isinstance(value, torch.Tensor):
            continue
        name = raw_name.removeprefix("module.")
        state[name] = value
    if not state:
        raise ValueError("InCTRL checkpoint does not contain named tensors.")
    return state


def _load_author_checkpoint(
    model: nn.Module,
    heads: InCTRLHeads,
    path: str | Path,
) -> None:
    state = _checkpoint_state(path)
    head_keys = set(heads.state_dict())
    missing_heads = sorted(head_keys - set(state))
    if missing_heads:
        raise ValueError(f"InCTRL checkpoint is missing trainable tensors: {missing_heads}")
    heads.load_state_dict({key: state[key] for key in head_keys}, strict=True)

    model_keys = set(model.state_dict())
    model_state = {key: value for key, value in state.items() if key in model_keys}
    incompatible = model.load_state_dict(model_state, strict=False)
    allowed_missing = {"logit_scale", "logit_bias"}
    missing_model = sorted(set(incompatible.missing_keys) - allowed_missing)
    if missing_model or incompatible.unexpected_keys:
        raise ValueError(
            "InCTRL checkpoint does not match OpenCLIP ViT-B/16+: "
            f"missing={missing_model}, unexpected={sorted(incompatible.unexpected_keys)}"
        )


class OpenCLIPInCTRLBackend:
    """Author-equivalent frozen ViT-B/16+ inference path."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        model_name: str = PAPER_MODEL_NAME,
        pretrained: str = PAPER_PRETRAINED,
        feature_layers: Sequence[int] = PAPER_FEATURE_LAYERS,
        image_size: int = PAPER_IMAGE_SIZE,
        device: str = "cuda",
        batch_size: int = 16,
        open_clip_module: Any = None,
        model: Any = None,
        preprocess: Any = None,
        tokenizer: Any = None,
        heads: InCTRLHeads | None = None,
    ) -> None:
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.feature_layers = tuple(int(layer) for layer in feature_layers)
        self.image_size = int(image_size)
        self.device = torch.device(
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        self.batch_size = int(batch_size)
        if self.feature_layers != PAPER_FEATURE_LAYERS:
            raise ValueError(f"Released InCTRL requires feature_layers={PAPER_FEATURE_LAYERS}.")
        if self.image_size != PAPER_IMAGE_SIZE:
            raise ValueError(f"Released InCTRL requires image_size={PAPER_IMAGE_SIZE}.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._open_clip = open_clip_module
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.heads = heads or InCTRLHeads()
        self._heads_injected = heads is not None
        self._initialized = False
        self._support_prototype: torch.Tensor | None = None
        self._patch_memories: tuple[torch.Tensor, ...] = ()
        self._text_features: torch.Tensor | None = None
        self.last_residual_map_: NDArray[np.float32] | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        if self.model is None or self.preprocess is None:
            load_pretrained = None if self.checkpoint_path is not None else self.pretrained
            self.model, self.preprocess, self.device = _load_openclip_model_and_preprocess(
                open_clip_module=self._open_clip,
                model_name=self.model_name,
                pretrained=load_pretrained,
                device=str(self.device),
                force_image_size=self.image_size,
            )
        else:
            self.model = self.model.to(self.device).eval()

        if self.tokenizer is None:
            if self._open_clip is None:
                from pyimgano.utils.optional_deps import require

                self._open_clip = require(
                    "open_clip", extra="clip", purpose="InCTRL text tokenization"
                )
            self.tokenizer = self._open_clip.get_tokenizer(self.model_name)

        visual = getattr(self.model, "visual", None)
        blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
        required = (
            "conv1",
            "class_embedding",
            "positional_embedding",
            "patch_dropout",
            "ln_pre",
            "ln_post",
            "proj",
        )
        if visual is None or blocks is None or any(not hasattr(visual, name) for name in required):
            raise TypeError("InCTRL requires an OpenCLIP VisionTransformer backbone.")
        if len(blocks) != 12:
            raise ValueError("Released InCTRL requires the 12-block ViT-B/16+ backbone.")
        projection_shape = tuple(getattr(visual.proj, "shape", ()))
        if int(visual.conv1.out_channels) != PAPER_VISUAL_WIDTH or projection_shape != (
            PAPER_VISUAL_WIDTH,
            PAPER_GLOBAL_DIM,
        ):
            raise ValueError("Released InCTRL requires the 896-wide, 640-D ViT-B/16+ tower.")

        if self.checkpoint_path is not None:
            _load_author_checkpoint(self.model, self.heads, self.checkpoint_path)
        elif not self._heads_injected:
            raise ValueError(
                "vision_inctrl requires checkpoint_path=... for the auxiliary-trained "
                "author heads, or explicit heads=... for offline testing."
            )

        self.model.requires_grad_(False).eval().to(self.device)
        self.heads.requires_grad_(False).eval().to(self.device)
        self._initialized = True

    def _preprocess_images(self, items: Sequence[Any]) -> torch.Tensor:
        self.initialize()
        from PIL import Image

        tensors = [
            self.preprocess(Image.fromarray(_as_rgb_uint8(item), mode="RGB")) for item in items
        ]
        batch = torch.stack(tensors)
        if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(
                "InCTRL preprocess must return tensors shaped "
                f"(3, {self.image_size}, {self.image_size})."
            )
        return batch

    def _encode_tensor_batch(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        visual = self.model.visual
        images = images.to(self.device, dtype=visual.conv1.weight.dtype)
        tokens = visual.conv1(images)
        grid = (int(tokens.shape[-2]), int(tokens.shape[-1]))
        if grid != PAPER_PATCH_GRID:
            raise RuntimeError(f"InCTRL expected a 15x15 patch grid, got {grid}.")
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(tokens.dtype).reshape(1, 1, -1)
        tokens = torch.cat((class_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        position = visual.positional_embedding.to(tokens.dtype)
        if position.ndim == 2:
            position = position.unsqueeze(0)
        if tuple(position.shape[-2:]) != tuple(tokens.shape[-2:]):
            raise RuntimeError("InCTRL positional embedding does not match the 15x15 grid.")
        tokens = visual.ln_pre(visual.patch_dropout(tokens + position))
        tokens = tokens.permute(1, 0, 2)

        selected: list[torch.Tensor] = []
        selected_layers = set(self.feature_layers)
        for index, block in enumerate(visual.transformer.resblocks, start=1):
            tokens = block(tokens)
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            if index in selected_layers:
                selected.append(tokens.permute(1, 0, 2)[:, 1:])
        if len(selected) != len(self.feature_layers):
            raise RuntimeError("InCTRL failed to collect blocks 7, 9, and 11.")

        pooled = visual.ln_post(tokens.permute(1, 0, 2)[:, 0])
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        if int(pooled.shape[-1]) != PAPER_GLOBAL_DIM:
            raise RuntimeError(f"InCTRL expected 640-D image features, got {pooled.shape[-1]}.")
        return pooled, tuple(selected)

    @torch.no_grad()
    def encode_images(self, items: Sequence[Any]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        tensors = self._preprocess_images(items)
        globals_, layer_batches = [], [[] for _ in self.feature_layers]
        for start in range(0, len(tensors), self.batch_size):
            global_batch, layers = self._encode_tensor_batch(
                tensors[start : start + self.batch_size]
            )
            globals_.append(global_batch)
            for output, layer in zip(layer_batches, layers):
                output.append(layer)
        return torch.cat(globals_), tuple(torch.cat(output) for output in layer_batches)

    @torch.no_grad()
    def _encode_prompt_prototype(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        prototype = F.normalize(self.model.encode_text(tokens), dim=-1).mean(dim=0, keepdim=True)
        return F.normalize(prototype, dim=-1)

    def fit(self, items: Sequence[Any], class_name: str) -> None:
        if not items:
            raise ValueError("InCTRL requires at least one normal image prompt.")
        self.initialize()
        with torch.no_grad():
            globals_, layers = self.encode_images(items)
            self._support_prototype = self.heads.adapter(globals_).mean(dim=0, keepdim=True)
            self._patch_memories = tuple(
                F.normalize(layer.reshape(-1, layer.shape[-1]), dim=-1) for layer in layers
            )
            normal, anomaly = _prompt_ensemble(class_name)
            self._text_features = torch.cat(
                (
                    self._encode_prompt_prototype(normal),
                    self._encode_prompt_prototype(anomaly),
                ),
                dim=0,
            )

    @torch.no_grad()
    def score(self, item: Any) -> float:
        if self._support_prototype is None or self._text_features is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        global_feature, layers = self.encode_images([item])
        residuals = []
        for query_layer, memory in zip(layers, self._patch_memories):
            query = F.normalize(query_layer[0], dim=-1)
            distance = PAPER_PATCH_DISTANCE_SCALE * (1.0 - query @ memory.T)
            residuals.append(distance.amin(dim=-1))
        patch_map = torch.stack(residuals).mean(dim=0)

        image_residual = self._support_prototype - self.heads.adapter(global_feature)
        image_score = self.heads.diff_head_ref(image_residual)
        query_global = F.normalize(global_feature, dim=-1)
        text_score = (PAPER_TEXT_LOGIT_SCALE * query_global @ self._text_features.T).softmax(
            dim=-1
        )[:, 1:2]
        holistic = self.heads.diff_head(patch_map.unsqueeze(0) + image_score + text_score)
        final_score = (holistic.squeeze(1) + patch_map.max().reshape(1)) / 2.0
        self.last_residual_map_ = (
            patch_map.reshape(PAPER_PATCH_GRID).cpu().numpy().astype(np.float32, copy=False)
        )
        return float(final_score.item())


@register_model(
    "vision_inctrl",
    tags=(
        "vision",
        "deep",
        "clip",
        "openclip",
        "few-shot",
        "generalist",
        "inctrl",
        "cvpr2024",
    ),
    metadata={
        "description": "Native InCTRL ViT-B/16+ in-context residual inference adaptation",
        "paper": "Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.html",
        "year": 2024,
        "conference": "CVPR",
        "implementation_status": "native-paper-inference-openclip-adaptation",
        "paper_fidelity": "paper-adaptation",
        "supervision": "few-shot",
        "supports_pixel_map": False,
        "requires_checkpoint": True,
        "weights_source": "Official InCTRL auxiliary-data checkpoint and OpenCLIP ViT-B/16+ LAION-400M e32",
    },
)
class VisionInCTRL(BaseDetector):
    """Checkpoint-backed InCTRL; target normal images are prompts, not training data."""

    input_mode = "images"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        class_name: str = "object",
        k_shot: int = 2,
        contamination: float = 0.1,
        batch_size: int = 16,
        device: str = "cuda",
        model_name: str = PAPER_MODEL_NAME,
        pretrained: str = PAPER_PRETRAINED,
        random_state: int | None = None,
        backend: Any = None,
        open_clip_module: Any = None,
        model: Any = None,
        preprocess: Any = None,
        tokenizer: Any = None,
        heads: InCTRLHeads | None = None,
    ) -> None:
        super().__init__(contamination=contamination)
        self.checkpoint_path = checkpoint_path
        self.class_name = " ".join(str(class_name).replace("_", " ").split())
        self.k_shot = int(k_shot)
        self.batch_size = int(batch_size)
        self.device = str(device)
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.random_state = random_state
        if not self.class_name:
            raise ValueError("class_name must be non-empty.")
        if self.k_shot <= 0:
            raise ValueError("k_shot must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.backend = backend or OpenCLIPInCTRLBackend(
            checkpoint_path=checkpoint_path,
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self.device,
            batch_size=self.batch_size,
            open_clip_module=open_clip_module,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            heads=heads,
        )
        if not hasattr(self.backend, "fit") or not hasattr(self.backend, "score"):
            raise TypeError("backend must implement .fit(items, class_name) and .score(item).")

    def fit(self, x: object = MISSING, y: Any = None, **kwargs: object) -> VisionInCTRL:
        items = _as_items(resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        if len(items) < self.k_shot:
            raise ValueError(f"InCTRL k_shot={self.k_shot} requires at least that many images.")
        self.support_images_ = items[: self.k_shot]
        self.backend.fit(self.support_images_, self.class_name)
        self.decision_scores_ = self.decision_function(self.support_images_)
        self._process_decision_scores()
        self._set_n_classes(y, warn_on_labeled_y=False)
        return self

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: int | None = None,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        validate_batch_size(batch_size)
        items = _as_items(resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
        return self.predict(items)

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
        items = _as_items(resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        return np.asarray([self.backend.score(item) for item in items], dtype=np.float64)
