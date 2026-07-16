from __future__ import annotations

"""Native WinCLIP/WinCLIP+ paper adaptation.

Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ._image_batch import _coerce_single_rgb_image
from .baseCv import BaseVisionDeepDetector
from .openclip_backend import _load_openclip_model_and_preprocess
from .registry import register_model

PAPER_MODEL = "ViT-B-16-plus-240"
PAPER_PRETRAINED = "laion400m_e31"
PAPER_IMAGE_SIZE = 240
PAPER_SCALES = (2, 3)
PAPER_TEMPERATURE = 0.07

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
    "{} with flaw",
    "{} with defect",
    "{} with damage",
)
PROMPT_TEMPLATES = (
    "a cropped photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of the {}.",
    "a dark photo of a {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
)


def _create_prompt_ensemble(class_name: str) -> tuple[list[str], list[str]]:
    """Return the exact state/template combinations from the supplement."""

    object_name = " ".join(str(class_name).replace("_", " ").split())
    if not object_name:
        raise ValueError("class_name must be non-empty")
    normal_states = [state.format(object_name) for state in NORMAL_STATES]
    anomalous_states = [state.format(object_name) for state in ANOMALOUS_STATES]
    return (
        [template.format(state) for state in normal_states for template in PROMPT_TEMPLATES],
        [template.format(state) for state in anomalous_states for template in PROMPT_TEMPLATES],
    )


def _class_scores(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = PAPER_TEMPERATURE,
) -> torch.Tensor:
    """Equation (1): two-class cosine softmax scores."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    return (image_embeddings @ text_embeddings.T / float(temperature)).softmax(dim=-1)


def _make_patch_masks(grid_size: tuple[int, int], scale: int) -> torch.Tensor:
    """Return stride-one square-window patch indices."""

    height, width = (int(grid_size[0]), int(grid_size[1]))
    if scale <= 0 or scale > min(height, width):
        raise ValueError(f"Invalid scale {scale} for patch grid {grid_size}")
    grid = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
    return F.unfold(grid, kernel_size=int(scale), stride=1).squeeze(0).to(torch.long)


def _harmonic_mean(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Paper harmonic mean, preserving exact zero-valued normal predictions."""

    values = values.clamp_min(0)
    reciprocal = torch.where(
        values > 0,
        values.reciprocal(),
        torch.full_like(values, torch.inf),
    )
    result = values.shape[dim] / reciprocal.sum(dim=dim)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _harmonic_aggregation(
    window_scores: torch.Tensor,
    grid_size: tuple[int, int],
    masks: torch.Tensor,
) -> torch.Tensor:
    """Equation (3): distribute window scores and harmonically aggregate overlaps."""

    if window_scores.ndim == 1:
        window_scores = window_scores.unsqueeze(0)
    masks = masks.to(window_scores.device)
    patch_scores = []
    for patch_index in range(int(grid_size[0] * grid_size[1])):
        covered = (masks == patch_index).any(dim=0)
        patch_scores.append(_harmonic_mean(window_scores[:, covered], dim=1))
    return torch.stack(patch_scores, dim=1).reshape(-1, *grid_size)


def _visual_association_score(
    embeddings: torch.Tensor,
    references: torch.Tensor,
) -> torch.Tensor:
    """Equation (4): minimum cosine distance to all normal memory entries."""

    if references.numel() == 0:
        raise RuntimeError("WinCLIP+ reference memory is empty")
    query = F.normalize(embeddings, dim=-1)
    memory = F.normalize(references.reshape(-1, references.shape[-1]), dim=-1)
    cosine = (query @ memory.T).clamp(-1, 1)
    return (1 - cosine.amax(dim=-1)) / 2


def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(math.floor(tile_size * (1.0 - overlap))))
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _square_tiles(
    image: NDArray[np.uint8],
    overlap: float = 0.2,
) -> list[tuple[NDArray[np.uint8], int, int]]:
    """Supplement tiling: shorter-edge squares with at least 20% overlap."""

    height, width = image.shape[:2]
    tile_size = min(height, width)
    y_starts = _tile_starts(height, tile_size, overlap)
    x_starts = _tile_starts(width, tile_size, overlap)
    return [(image[y : y + tile_size, x : x + tile_size], y, x) for y in y_starts for x in x_starts]


def _as_rgb_uint8(image: NDArray[Any]) -> NDArray[np.uint8]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"WinCLIP expects RGB HWC images, got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("WinCLIP images must have non-zero height and width")
    if not np.isfinite(array).all():
        raise ValueError("WinCLIP images must contain only finite values")
    minimum, maximum = float(array.min()), float(array.max())
    if minimum < 0 or maximum > 255:
        raise ValueError("WinCLIP image values must be in [0, 1] or [0, 255]")
    if np.issubdtype(array.dtype, np.floating) and maximum <= 1.0:
        array = array * 255.0
    return np.ascontiguousarray(np.rint(array).astype(np.uint8))


class OpenCLIPWinCLIPBackend:
    """Frozen OpenCLIP ViT with efficient masked-token window forwarding."""

    def __init__(
        self,
        *,
        model_name: str = PAPER_MODEL,
        pretrained: str = PAPER_PRETRAINED,
        image_size: int = PAPER_IMAGE_SIZE,
        scales: Sequence[int] = PAPER_SCALES,
        device: str = "cuda",
        open_clip_module: Any = None,
        model: Any = None,
        preprocess: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.image_size = int(image_size)
        self.scales = tuple(int(scale) for scale in scales)
        self.device = torch.device(device)
        self._open_clip = open_clip_module
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.grid_size: tuple[int, int] | None = None
        self.masks: list[torch.Tensor] = []

    def initialize(self) -> None:
        if self.grid_size is not None:
            return
        if self.model is None or self.preprocess is None:
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
            if self._open_clip is None:
                from pyimgano.utils.optional_deps import require

                self._open_clip = require(
                    "open_clip",
                    extra="clip",
                    purpose="WinCLIP paper implementation",
                )
            self.tokenizer = self._open_clip.get_tokenizer(self.model_name)

        visual = getattr(self.model, "visual", None)
        required = (
            "patch_dropout",
            "ln_pre",
            "transformer",
            "ln_post",
            "_global_pool",
            "grid_size",
        )
        if visual is None or not all(hasattr(visual, name) for name in required):
            raise ValueError("WinCLIP requires an OpenCLIP VisionTransformer backbone")
        visual.output_tokens = True
        grid = visual.grid_size
        self.grid_size = (
            (int(grid), int(grid)) if isinstance(grid, int) else (int(grid[0]), int(grid[1]))
        )
        self.masks = [
            _make_patch_masks(self.grid_size, scale).to(self.device) for scale in self.scales
        ]
        self.model.requires_grad_(False).eval()

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        self.initialize()
        tokens = self.tokenizer(list(prompts)).to(self.device)
        return self.model.encode_text(tokens)

    def _preprocess_images(self, images: Sequence[NDArray[np.uint8]]) -> torch.Tensor:
        self.initialize()
        from PIL import Image

        return torch.stack(
            [self.preprocess(Image.fromarray(image, mode="RGB")) for image in images]
        ).to(self.device)

    def _window_embeddings(
        self,
        feature_map: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.model.visual
        batch_size = feature_map.shape[0]
        window_count = masks.shape[1]
        indices = torch.cat(
            (
                torch.zeros((1, window_count), dtype=torch.long, device=feature_map.device),
                masks.to(feature_map.device) + 1,
            ),
            dim=0,
        ).T
        masked = torch.cat(
            [feature_map.index_select(1, index) for index in indices],
            dim=0,
        )
        masked = visual.ln_pre(visual.patch_dropout(masked))
        transformed = visual.transformer(masked.permute(1, 0, 2))
        if isinstance(transformed, tuple):
            transformed = transformed[0]
        transformed = transformed.permute(1, 0, 2)
        pooled, _ = visual._global_pool(transformed)
        pooled = visual.ln_post(pooled)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled.reshape(window_count, batch_size, -1).permute(1, 0, 2)

    @torch.no_grad()
    def encode_images(
        self,
        images: Sequence[NDArray[np.uint8]],
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
        tuple[int, int],
    ]:
        self.initialize()
        batch = self._preprocess_images(images)
        captured: dict[str, torch.Tensor] = {}

        def capture_tokens(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
            captured["feature_map"] = inputs[0].detach()

        handle = self.model.visual.patch_dropout.register_forward_pre_hook(capture_tokens)
        try:
            result = self.model.encode_image(batch)
        finally:
            handle.remove()
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("OpenCLIP visual backbone did not return image and patch tokens")
        image_embeddings, patch_embeddings = result
        feature_map = captured.get("feature_map")
        if feature_map is None:
            raise RuntimeError("Failed to capture OpenCLIP pre-transformer patch tokens")
        windows = [self._window_embeddings(feature_map, mask) for mask in self.masks]
        if self.grid_size is None:  # pragma: no cover - initialize sets it
            raise RuntimeError("OpenCLIP patch grid is unavailable")
        return image_embeddings, windows, patch_embeddings, self.grid_size


@register_model(
    "winclip",
    tags=(
        "vision",
        "deep",
        "clip",
        "openclip",
        "winclip",
        "zero-shot",
        "few-shot",
        "pixel_map",
        "cvpr2023",
    ),
    metadata={
        "description": "Native WinCLIP/WinCLIP+ CPE, masked-window, and reference-association method",
        "paper": "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html",
        "year": 2023,
        "conference": "CVPR",
        "implementation_status": "native-paper-method-openclip-adaptation",
        "paper_fidelity": "paper-adaptation",
        "type": "vision-language",
        "supervision": "zero-shot",
        "supports_pixel_map": True,
        "requires_checkpoint": True,
        "weights_source": "OpenCLIP ViT-B-16-plus-240 laion400m_e31",
    },
)
class WinCLIPDetector(BaseVisionDeepDetector):
    """WinCLIP zero-shot and WinCLIP+ few-normal-shot detector."""

    def __init__(
        self,
        clip_model: Optional[str] = None,
        window_size: Optional[int] = None,
        window_stride: Optional[int] = None,
        text_prompts: Optional[Mapping[str, Sequence[str]]] = None,
        k_shot: int = 0,
        scales: Sequence[int] = PAPER_SCALES,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        *,
        class_name: str = "object",
        openclip_model_name: str = PAPER_MODEL,
        openclip_pretrained: str = PAPER_PRETRAINED,
        image_size: int = PAPER_IMAGE_SIZE,
        temperature: float = PAPER_TEMPERATURE,
        tile_overlap: float = 0.2,
        backend: Any = None,
        open_clip_module: Any = None,
        **kwargs: Any,
    ) -> None:
        if clip_model is not None:
            legacy_name = str(clip_model).replace("/", "-")
            if legacy_name != PAPER_MODEL:
                raise ValueError(
                    "clip_model belonged to the removed crop proxy; paper WinCLIP uses "
                    f"openclip_model_name={PAPER_MODEL!r}"
                )
            openclip_model_name = legacy_name
        if window_size is not None or window_stride is not None:
            warnings.warn(
                "window_size/window_stride belonged to the removed crop proxy and are ignored; "
                "use patch scales=(2, 3) for paper WinCLIP.",
                DeprecationWarning,
                stacklevel=2,
            )
        scale_values = tuple(scales)
        parsed_scales = tuple(int(scale) for scale in scale_values)
        if not parsed_scales or any(
            float(scale) != int(scale) or int(scale) <= 0 for scale in scale_values
        ):
            raise ValueError("scales must contain positive integer patch-window sizes")
        if k_shot < 0 or image_size <= 0 or temperature <= 0:
            raise ValueError("k_shot must be non-negative and image_size/temperature positive")
        if not 0 <= tile_overlap < 1:
            raise ValueError("tile_overlap must be in [0, 1)")

        requested = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        resolved = (
            requested
            if requested.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        super().__init__(
            device=str(resolved),
            random_state=random_state,
            batch_size=1,
            **kwargs,
        )
        self.device = resolved
        self.openclip_model_name = str(openclip_model_name)
        self.openclip_pretrained = str(openclip_pretrained)
        self.image_size = int(image_size)
        self.temperature = float(temperature)
        self.k_shot = int(k_shot)
        self.scales = parsed_scales
        self.tile_overlap = float(tile_overlap)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.text_prompts = text_prompts
        self.backend = backend or OpenCLIPWinCLIPBackend(
            model_name=self.openclip_model_name,
            pretrained=self.openclip_pretrained,
            image_size=self.image_size,
            scales=self.scales,
            device=str(self.device),
            open_clip_module=open_clip_module,
        )
        self.reference_windows_: list[torch.Tensor] | None = None
        self.reference_patches_: torch.Tensor | None = None
        self.set_class_name(class_name)

    def _prompts_for_class(self, class_name: str) -> tuple[list[str], list[str]]:
        if self.text_prompts is None:
            return _create_prompt_ensemble(class_name)
        try:
            normal = self.text_prompts["normal"]
            anomalous = self.text_prompts["anomaly"]
        except (KeyError, TypeError) as exc:
            raise ValueError("text_prompts must define non-empty normal and anomaly lists") from exc
        normal_prompts = [str(prompt).format(class_name) for prompt in normal]
        anomalous_prompts = [str(prompt).format(class_name) for prompt in anomalous]
        if not normal_prompts or not anomalous_prompts:
            raise ValueError("text_prompts must define non-empty normal and anomaly lists")
        return normal_prompts, anomalous_prompts

    def set_class_name(self, class_name: str) -> WinCLIPDetector:
        self.class_name = str(class_name)
        normal, anomalous = self._prompts_for_class(self.class_name)
        with torch.no_grad():
            normal_embedding = self.backend.encode_text(normal).mean(dim=0, keepdim=True)
            anomalous_embedding = self.backend.encode_text(anomalous).mean(dim=0, keepdim=True)
        self.text_embeddings_ = torch.cat((normal_embedding, anomalous_embedding), dim=0)
        return self

    @staticmethod
    def _coerce_images(x: Any) -> list[NDArray[np.uint8]]:
        if isinstance(x, (str, Path)) or (isinstance(x, np.ndarray) and x.ndim in (2, 3)):
            items = [x]
        elif isinstance(x, np.ndarray) and x.ndim == 4:
            items = list(x)
        else:
            items = list(x)
        if not items:
            raise ValueError("Expected at least one image input")
        return [_as_rgb_uint8(_coerce_single_rgb_image(item)) for item in items]

    def _encode_tile(
        self,
        tile: NDArray[np.uint8],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, tuple[int, int]]:
        return self.backend.encode_images([tile])

    def _score_tile(self, tile: NDArray[np.uint8]) -> tuple[float, NDArray[np.float32]]:
        image_embedding, window_embeddings, patch_embeddings, grid_size = self._encode_tile(tile)
        if len(window_embeddings) != len(self.scales):
            raise RuntimeError("WinCLIP backend returned the wrong number of window scales")

        image_score = _class_scores(
            image_embedding,
            self.text_embeddings_,
            self.temperature,
        )[..., 1]
        zero_maps = [image_score.reshape(1, 1, 1).expand(1, *grid_size)]
        masks = [_make_patch_masks(grid_size, scale) for scale in self.scales]
        for embeddings, mask in zip(window_embeddings, masks):
            scores = _class_scores(embeddings, self.text_embeddings_, self.temperature)[..., 1]
            zero_maps.append(_harmonic_aggregation(scores, grid_size, mask))
        zero_map = _harmonic_mean(torch.stack(zero_maps), dim=0)

        anomaly_map = zero_map
        final_image_score = image_score
        if self.reference_windows_ is not None and self.reference_patches_ is not None:
            if len(self.reference_windows_) != len(masks):
                raise RuntimeError("WinCLIP reference windows do not match the configured scales")
            few_maps = [
                _visual_association_score(
                    patch_embeddings,
                    self.reference_patches_,
                ).reshape(1, *grid_size)
            ]
            for embeddings, references, mask in zip(
                window_embeddings,
                self.reference_windows_,
                masks,
            ):
                scores = _visual_association_score(embeddings, references)
                few_maps.append(_harmonic_aggregation(scores, grid_size, mask))
            few_map = torch.stack(few_maps).mean(dim=0)  # Equation (5)
            anomaly_map = (zero_map + few_map) / 2
            final_image_score = (image_score + few_map.amax(dim=(-2, -1))) / 2  # Equation (6)

        resized = (
            F.interpolate(
                anomaly_map.unsqueeze(1),
                size=tile.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        return (
            float(final_image_score.item()),
            resized.detach().float().cpu().numpy().astype(np.float32, copy=False),
        )

    def _score_image(self, image: NDArray[np.uint8]) -> tuple[float, NDArray[np.float32]]:
        score_sum = 0.0
        map_sum = np.zeros(image.shape[:2], dtype=np.float32)
        counts = np.zeros(image.shape[:2], dtype=np.float32)
        tiles = _square_tiles(image, self.tile_overlap)
        for tile, y, x in tiles:
            tile_score, tile_map = self._score_tile(tile)
            height, width = tile.shape[:2]
            score_sum += tile_score
            map_sum[y : y + height, x : x + width] += tile_map
            counts[y : y + height, x : x + width] += 1
        return score_sum / len(tiles), map_sum / np.maximum(counts, 1)

    def _collect_references(self, images: Sequence[NDArray[np.uint8]]) -> None:
        selected = list(images)
        if len(selected) > self.k_shot:
            indices = self.rng.choice(len(selected), self.k_shot, replace=False)
            selected = [selected[int(index)] for index in indices]

        windows_per_scale: list[list[torch.Tensor]] = [[] for _ in self.scales]
        patches: list[torch.Tensor] = []
        for image in selected:
            for tile, _, _ in _square_tiles(image, self.tile_overlap):
                _, windows, patch_embeddings, _ = self._encode_tile(tile)
                patches.append(patch_embeddings)
                for scale_index, embeddings in enumerate(windows):
                    windows_per_scale[scale_index].append(embeddings)
        self.reference_windows_ = [torch.cat(items, dim=0) for items in windows_per_scale]
        self.reference_patches_ = torch.cat(patches, dim=0)

    def fit(self, x: Any, y: Optional[NDArray[Any]] = None, **kwargs: Any) -> WinCLIPDetector:
        del kwargs
        images = self._coerce_images(x)
        if self.k_shot > 0:
            self._collect_references(images)
        else:
            self.reference_windows_ = None
            self.reference_patches_ = None
        self.decision_scores_ = self.decision_function(images)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - float(self.contamination)))
        self.is_fitted_ = True
        self._set_n_classes(y)
        return self

    def decision_function(
        self,
        x: Any,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        del batch_size, kwargs
        return np.asarray(
            [self._score_image(image)[0] for image in self._coerce_images(x)],
            dtype=np.float64,
        )

    def predict_proba(self, x: Any, **kwargs: Any) -> NDArray[np.float64]:
        return self.decision_function(x, **kwargs)

    def predict(self, x: Any, **kwargs: Any) -> NDArray[np.int64]:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")
        return (self.decision_function(x, **kwargs) > float(self.threshold_)).astype(np.int64)

    def predict_anomaly_map(self, x: Any) -> list[NDArray[np.float32]]:
        return [self._score_image(image)[1] for image in self._coerce_images(x)]

    def get_anomaly_map(self, image: Any) -> NDArray[np.float32]:
        return self.predict_anomaly_map(image)[0]


@register_model(
    "vision_winclip",
    tags=(
        "vision",
        "deep",
        "clip",
        "openclip",
        "winclip",
        "zero-shot",
        "few-shot",
        "pixel_map",
        "cvpr2023",
    ),
    metadata={
        "description": "Native WinCLIP/WinCLIP+ CPE, masked-window, and reference-association method",
        "paper": "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html",
        "year": 2023,
        "conference": "CVPR",
        "implementation_status": "native-paper-method-openclip-adaptation",
        "paper_fidelity": "paper-adaptation",
        "type": "vision-language",
        "supervision": "zero-shot",
        "supports_pixel_map": True,
        "requires_checkpoint": True,
        "weights_source": "OpenCLIP ViT-B-16-plus-240 laion400m_e31",
    },
)
class VisionWinCLIP(WinCLIPDetector):
    """Registry alias for :class:`WinCLIPDetector`."""
