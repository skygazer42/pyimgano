"""Checkpoint-backed inference for GLAD (ECCV 2024).

This module adapts the authors' released inference path.  It deliberately does
not recreate GLAD's category-specific ATP training loop; a fine-tuned GLAD UNet
checkpoint is required.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from pyimgano.utils.random_state import isolated_random_state_method

from ._batch_size import call_with_temporary_attr, validate_batch_size
from ._image_batch import _coerce_single_rgb_image
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .deep_io import safe_torch_load
from .registry import register_model

PAPER_BASE_MODEL = "CompVis/stable-diffusion-v1-4"
PAPER_PROMPT = "a photo of sks"
PAPER_GUIDANCE_SCALE = 9.0
PAPER_INPUT_RESOLUTION = 512
PAPER_DINO_LAYERS = (3, 6, 9, 12)
PAPER_DINO_MODEL = "dino_vitb8"
PAPER_GAUSSIAN_SIGMA = 6.0
PAPER_IMAGE_TOPK = 250
PAPER_ADS_TOPK = 10


class GLADPreset(NamedTuple):
    """Released category-specific GLAD inference parameters."""

    denoise_step: int
    input_threshold: float
    pixel_weight: float
    min_step: int
    unet_checkpoint_step: int
    dino_epoch: int
    dino_resolution: int
    inference_steps: int


def _presets(
    values: dict[str, tuple[int, float, float, int, int, int]],
    *,
    dino_resolution: int,
    inference_steps: int,
) -> dict[str, GLADPreset]:
    return {
        name: GLADPreset(*value, dino_resolution, inference_steps) for name, value in values.items()
    }


GLAD_PRESETS = {
    "mvtec": _presets(
        {
            "carpet": (750, 0.32, 0, 350, 2000, 0),
            "grid": (750, 0.47, 0, 350, 3000, 0),
            "leather": (750, 0.35, 0, 350, 2500, 0),
            "tile": (750, 0.35, 0, 350, 500, 0),
            "wood": (750, 0.37, 0, 350, 1000, 0),
            "bottle": (750, 0.32, 0, 350, 2500, 0),
            "cable": (750, 0.40, 0, 350, 1500, 0),
            "capsule": (600, 0.40, 0, 350, 2000, 0),
            "hazelnut": (750, 0.50, 0, 350, 3500, 0),
            "metal_nut": (750, 0.40, 0, 350, 2500, 0),
            "pill": (750, 0.35, 0, 350, 1500, 0),
            "screw": (750, 0.32, 0, 350, 3000, 0),
            "toothbrush": (750, 0.50, 0, 350, 500, 0),
            "transistor": (850, 0.50, 0, 350, 2500, 0),
            "zipper": (750, 0.35, 0, 350, 1000, 0),
        },
        dino_resolution=512,
        inference_steps=25,
    ),
    "mpdd": _presets(
        {
            "bracket_black": (500, 0.35, 0, 350, 2500, 0),
            "bracket_brown": (500, 0.35, 0, 350, 7000, 0),
            "bracket_white": (450, 0.35, 0, 200, 1000, 0),
            "connector": (500, 0.35, 0, 350, 1000, 0),
            "metal_plate": (500, 0.35, 0, 350, 2500, 0),
            "tubes": (500, 0.10, 0, 350, 500, 0),
        },
        dino_resolution=512,
        inference_steps=25,
    ),
    "visa": _presets(
        {
            "candle": (450, 0.45, 7, 200, 4000, 1),
            "capsules": (450, 0.40, 7, 200, 4000, 4),
            "cashew": (450, 0.40, 1, 200, 4000, 7),
            "chewinggum": (450, 0.45, 1, 200, 4000, 7),
            "fryum": (450, 0.35, 2, 200, 4000, 1),
            "macaroni1": (450, 0.45, 7, 200, 4000, 1),
            "macaroni2": (450, 0.45, 7, 200, 4000, 8),
            "pcb1": (450, 0.30, 1, 200, 4000, 7),
            "pcb2": (450, 0.30, 2, 200, 4000, 6),
            "pcb3": (450, 0.30, 2, 200, 4000, 9),
            "pcb4": (450, 0.30, 2, 200, 4000, 0),
            "pipe_fryum": (450, 0.45, 2, 200, 4000, 2),
        },
        dino_resolution=256,
        inference_steps=15,
    ),
    "pcbbank": _presets(
        {
            "pcb1": (450, 0.30, 1, 200, 4000, 7),
            "pcb2": (450, 0.30, 2, 200, 4000, 6),
            "pcb3": (450, 0.30, 2, 200, 4000, 9),
            "pcb4": (450, 0.30, 2, 200, 4000, 0),
            "pcb5": (450, 0.40, 2, 200, 4000, 7),
            "pcb6": (450, 0.45, 1, 200, 4000, 14),
            "pcb7": (450, 0.30, 2, 200, 4000, 1),
        },
        dino_resolution=256,
        inference_steps=15,
    ),
}

_DATASET_ALIASES = {
    "mvtec": "mvtec",
    "mvtecad": "mvtec",
    "mpdd": "mpdd",
    "visa": "visa",
    "pcb": "pcbbank",
    "pcbbank": "pcbbank",
}
_REVERSE_DISTANCE_CLASSES = {"transistor", "pcb1", "pcb4"}
_DARK_OBJECT_CLASSES = {"screw", "capsule", "bracket_black", "bracket_brown"}
_LIGHT_OBJECT_CLASSES = {"bracket_white"}


def _uses_fine_tuned_dino(dataset: str) -> bool:
    key = "".join(character for character in str(dataset).lower() if character.isalnum())
    return key in {"visa", "pcb", "pcbbank"}


def get_glad_preset(dataset: str, class_name: str) -> GLADPreset:
    """Return the exact preset used by the authors' released test script."""

    dataset_key = "".join(character for character in str(dataset).lower() if character.isalnum())
    dataset_key = _DATASET_ALIASES.get(dataset_key, dataset_key)
    class_key = str(class_name).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return GLAD_PRESETS[dataset_key][class_key]
    except KeyError as exc:
        supported = ", ".join(sorted(GLAD_PRESETS.get(dataset_key, {})))
        if not supported:
            raise ValueError(f"Unsupported GLAD dataset: {dataset!r}.") from exc
        raise ValueError(
            f"Unsupported GLAD class {class_name!r} for {dataset!r}; choose one of: {supported}."
        ) from exc


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 4:
            if not len(value):
                raise ValueError("GLAD requires at least one image.")
            return [value[index] for index in range(len(value))]
        if value.ndim in (2, 3):
            return [value]
    if isinstance(value, (str, Path)):
        return [value]
    items = list(value)
    if not items:
        raise ValueError("GLAD requires at least one image.")
    return items


def _as_rgb_uint8(image: Any) -> NDArray[np.uint8]:
    array = np.asarray(_coerce_single_rgb_image(image))
    if not np.isfinite(array).all():
        raise ValueError("GLAD images must contain only finite values.")
    minimum, maximum = float(array.min()), float(array.max())
    if minimum < 0 or maximum > 255:
        raise ValueError("GLAD image values must be in [0, 1] or [0, 255].")
    if np.issubdtype(array.dtype, np.floating) and maximum <= 1:
        array = array * 255
    return np.ascontiguousarray(np.rint(array).astype(np.uint8))


def _resize_rgb(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(image, mode="RGB").resize(
        (int(size), int(size)), resample=resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.uint8)


def _object_mask(image: NDArray[np.uint8], class_name: str) -> NDArray[np.float32]:
    import cv2
    from scipy.ndimage import gaussian_filter

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if class_name in _DARK_OBJECT_CLASSES:
        foreground = 1 - cv2.threshold(gray, 100, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif class_name in _LIGHT_OBJECT_CLASSES:
        foreground = cv2.threshold(gray, 100, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    else:
        foreground = np.ones_like(gray)
    blurred = gaussian_filter(foreground.astype(np.float32), sigma=PAPER_GAUSSIAN_SIGMA)
    return gaussian_filter((blurred > 0).astype(np.float32), sigma=PAPER_GAUSSIAN_SIGMA)


def _prepare_images(
    items: Sequence[Any], class_name: str, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    resized = [_resize_rgb(_as_rgb_uint8(item), PAPER_INPUT_RESOLUTION) for item in items]
    images = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).to(device=device)
    images = images.to(dtype=dtype).div_(127.5).sub_(1)
    masks = np.stack([_object_mask(image, class_name) for image in resized])
    return images, torch.from_numpy(masks).to(device=device, dtype=dtype)


def _dino_patch_tokens(model: Any, images: torch.Tensor) -> list[torch.Tensor]:
    """Extract raw tokens after DINO blocks 3/6/9/12, as in GLAD."""

    prepare = getattr(model, "prepare_tokens", None)
    if not callable(prepare):
        prepare = getattr(model, "prepare_tokens_with_masks", None)
    blocks = getattr(model, "blocks", None)
    if callable(prepare) and blocks is not None:
        try:
            tokens = prepare(images)
        except TypeError:
            tokens = prepare(images, None)
        layers = []
        selected = set(PAPER_DINO_LAYERS)
        for index, block in enumerate(blocks, start=1):
            tokens = block(tokens)
            if index in selected:
                layers.append(tokens)
            if index == PAPER_DINO_LAYERS[-1]:
                break
    else:
        output = model(images)
        if not (
            isinstance(output, (tuple, list))
            and len(output) >= 2
            and isinstance(output[1], (tuple, list))
        ):
            raise TypeError(
                "DINO model must be the GLAD author model or expose prepare_tokens() and blocks."
            )
        layers = list(output[1])
    if len(layers) != len(PAPER_DINO_LAYERS):
        raise ValueError(
            f"GLAD requires DINO layers {PAPER_DINO_LAYERS}, got {len(layers)} outputs."
        )
    if any(layer.ndim != 3 or layer.shape[1] < 2 for layer in layers):
        raise ValueError("Each DINO output must have shape (batch, cls+patches, channels).")
    return layers


def _feature_anomaly_map(
    input_tokens: Sequence[torch.Tensor],
    reconstruction_tokens: Sequence[torch.Tensor],
    *,
    output_size: int,
    reverse: bool,
) -> torch.Tensor:
    if len(input_tokens) != len(reconstruction_tokens) or not input_tokens:
        raise ValueError("Input and reconstruction DINO layers must be non-empty and aligned.")
    forward_map = None
    backward_map = None
    for input_layer, reconstruction_layer in zip(input_tokens, reconstruction_tokens):
        input_patches = F.normalize(input_layer[:, 1:], dim=-1)
        reconstruction_patches = F.normalize(reconstruction_layer[:, 1:], dim=-1)
        similarity = torch.bmm(input_patches, reconstruction_patches.transpose(1, 2))
        patch_count = int(input_patches.shape[1])
        grid = math.isqrt(patch_count)
        if grid * grid != patch_count:
            raise ValueError(f"DINO patch count must form a square grid, got {patch_count}.")
        forward = (1 - similarity).amin(dim=-1).reshape(-1, 1, grid, grid)
        layer_forward = F.interpolate(
            forward, size=(output_size, output_size), mode="bilinear", align_corners=True
        )
        forward_map = layer_forward if forward_map is None else forward_map + layer_forward
        if reverse:
            backward = (1 - similarity).amin(dim=-2).reshape(-1, 1, grid, grid)
            layer_backward = F.interpolate(
                backward, size=(output_size, output_size), mode="bilinear", align_corners=True
            )
            backward_map = layer_backward if backward_map is None else backward_map + layer_backward
    result = cast(torch.Tensor, forward_map)
    return result if backward_map is None else result + backward_map


def _gaussian_blur(images: torch.Tensor, sigma: float = PAPER_GAUSSIAN_SIGMA) -> torch.Tensor:
    radius = int(4 * float(sigma) + 0.5)
    coordinates = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    horizontal = kernel.reshape(1, 1, 1, -1).expand(images.shape[1], 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(images.shape[1], 1, -1, 1)
    blurred = F.conv2d(
        F.pad(images, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=images.shape[1],
    )
    return F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=images.shape[1],
    )


def _add_noise(
    alphas_cumprod: torch.Tensor,
    samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alphas = alphas_cumprod.to(device=samples.device, dtype=samples.dtype)[timesteps]
    while alphas.ndim < samples.ndim:
        alphas = alphas.unsqueeze(-1)
    return alphas.sqrt() * samples + (1 - alphas).sqrt() * noise


def _decode(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents / float(vae.config.scaling_factor), return_dict=False)
    return decoded[0] if isinstance(decoded, (tuple, list)) else decoded.sample


def _adaptive_ddim_step(
    *,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    image_latents: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    final_alpha_cumprod: torch.Tensor,
    step_ratio: int,
    vae: Any,
    dino_model: Any,
    active: torch.Tensor,
    chosen_steps: torch.Tensor,
    thresholds: torch.Tensor,
    input_threshold: float,
    min_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_t = alphas_cumprod[timestep].to(device=sample.device, dtype=sample.dtype)
    previous_timestep = int(timestep) - int(step_ratio)
    alpha_previous = (
        alphas_cumprod[previous_timestep] if previous_timestep >= 0 else final_alpha_cumprod
    ).to(device=sample.device, dtype=sample.dtype)
    beta_t = 1 - alpha_t
    predicted = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
    fused = predicted

    if input_threshold > 0 and bool(active.any()):
        timestep_batch = torch.full(
            (len(sample),), int(timestep), device=sample.device, dtype=torch.long
        )
        directly_noised = _add_noise(alphas_cumprod, image_latents, noise, timestep_batch)
        direct_prediction = (directly_noised - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        input_tokens = _dino_patch_tokens(dino_model, _decode(vae, direct_prediction))
        reconstruction_tokens = _dino_patch_tokens(dino_model, _decode(vae, predicted))
        ads_map = _feature_anomaly_map(
            input_tokens[-1:],
            reconstruction_tokens[-1:],
            output_size=int(predicted.shape[-1]),
            reverse=False,
        )
        scores = torch.topk(ads_map.flatten(1), k=min(PAPER_ADS_TOPK, ads_map[0].numel())).values
        scores = scores.mean(dim=1)
        fused = predicted.clone()
        for index in range(len(sample)):
            if bool(active[index]) and (
                bool(scores[index] > input_threshold) or timestep < min_step
            ):
                if bool(scores[index] > input_threshold):
                    mask = torch.sigmoid(ads_map[index]).to(dtype=predicted.dtype)
                    fused[index] = mask * predicted[index] + (1 - mask) * direct_prediction[index]
                else:
                    fused[index] = direct_prediction[index]
                active[index] = False
                chosen_steps[index] = timestep
                thresholds[index] = scores[index]

    direction = (1 - alpha_previous).sqrt() * model_output
    return alpha_previous.sqrt() * fused + direction, active


def _load_state_dict(module: Any, path: str | Path) -> None:
    payload = safe_torch_load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"Checkpoint {path!s} must contain a state_dict mapping.")
    module.load_state_dict(
        {key: value for key, value in payload.items() if "loss" not in key}, strict=True
    )


class TorchGLADBackend:
    """GLAD inference using a fine-tuned SD-v1.4 UNet and DINO ViT-B/8."""

    def __init__(
        self,
        *,
        preset: GLADPreset,
        class_name: str,
        dataset: str,
        pretrained_model_path: str = PAPER_BASE_MODEL,
        unet_checkpoint_path: str | Path | None = None,
        vae_checkpoint_path: str | Path | None = None,
        dino_checkpoint_path: str | Path | None = None,
        pipeline: Any = None,
        dino_model: Any = None,
        ads_dino_model: Any = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        batch_size: int = 2,
        allow_download: bool = False,
        dino_hub_repo: str = "facebookresearch/dino:main",
    ) -> None:
        self.preset = preset
        self.class_name = str(class_name).lower()
        self.dataset = str(dataset).lower()
        self.pretrained_model_path = str(pretrained_model_path)
        self.unet_checkpoint_path = unet_checkpoint_path
        self.vae_checkpoint_path = vae_checkpoint_path
        self.dino_checkpoint_path = dino_checkpoint_path
        self.pipeline = pipeline
        self.dino_model = dino_model
        self.ads_dino_model = ads_dino_model
        resolved_device = (
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(resolved_device)
        self.dtype = torch.float32 if self.device.type == "cpu" else (dtype or torch.float16)
        self.batch_size = int(batch_size)
        self.allow_download = bool(allow_download)
        self.dino_hub_repo = str(dino_hub_repo)
        self._pipeline_injected = pipeline is not None
        self._loaded = False
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.pipeline is None:
            if self.unet_checkpoint_path is None:
                raise ValueError(
                    "GLAD requires unet_checkpoint_path; the base Stable Diffusion UNet is not a GLAD model."
                )
            try:
                from diffusers import DDIMScheduler, StableDiffusionPipeline
            except ImportError as exc:
                raise ImportError(
                    "GLAD requires the 'diffusion' extra: pip install 'pyimgano[diffusion]'."
                ) from exc
            scheduler = DDIMScheduler.from_pretrained(
                self.pretrained_model_path,
                subfolder="scheduler",
                local_files_only=not self.allow_download,
            )
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.pretrained_model_path,
                scheduler=scheduler,
                torch_dtype=self.dtype,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                local_files_only=not self.allow_download,
            )
        required = ("vae", "unet", "text_encoder", "tokenizer", "scheduler")
        missing = [name for name in required if not hasattr(self.pipeline, name)]
        if missing:
            raise TypeError(f"GLAD pipeline is missing components: {', '.join(missing)}.")
        if self.unet_checkpoint_path is not None:
            _load_state_dict(self.pipeline.unet, self.unet_checkpoint_path)
            self.unet_checkpoint_path = None
        elif not self._pipeline_injected:
            raise ValueError("A fine-tuned GLAD UNet checkpoint is required.")

        if self.dino_model is None:
            if not self.allow_download:
                raise ValueError(
                    "GLAD requires dino_model when downloads are disabled; pass the official DINO ViT-B/8."
                )
            self.dino_model = torch.hub.load(self.dino_hub_repo, PAPER_DINO_MODEL, pretrained=True)
        uses_fine_tuned_dino = _uses_fine_tuned_dino(self.dataset)
        if self.ads_dino_model is None:
            if uses_fine_tuned_dino:
                if self.dino_checkpoint_path is None:
                    raise ValueError(
                        "VisA/PCB-Bank GLAD requires both the fine-tuned final DINO and "
                        "the frozen pretrained DINO used by ADS."
                    )
                self.ads_dino_model = deepcopy(self.dino_model)
            else:
                self.ads_dino_model = self.dino_model
        if self.vae_checkpoint_path is not None:
            _load_state_dict(self.pipeline.vae, self.vae_checkpoint_path)
            self.vae_checkpoint_path = None
        if self.dino_checkpoint_path is not None:
            _load_state_dict(self.dino_model, self.dino_checkpoint_path)
            self.dino_checkpoint_path = None

        modules = (
            self.pipeline.vae,
            self.pipeline.unet,
            self.pipeline.text_encoder,
            self.dino_model,
            self.ads_dino_model,
        )
        for index, module in enumerate(modules):
            if any(module is previous for previous in modules[:index]):
                continue
            module.requires_grad_(False).eval().to(device=self.device, dtype=self.dtype)
        self._loaded = True

    def _prompt_embeddings(self, batch_size: int) -> torch.Tensor:
        tokenizer = self.pipeline.tokenizer
        max_length = int(tokenizer.model_max_length)

        def encode(text: str) -> torch.Tensor:
            inputs = tokenizer(
                [text],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            attention_mask = None
            config = getattr(self.pipeline.text_encoder, "config", None)
            if bool(getattr(config, "use_attention_mask", False)):
                attention_mask = inputs.attention_mask.to(self.device)
            return self.pipeline.text_encoder(
                inputs.input_ids.to(self.device), attention_mask=attention_mask
            )[0].to(dtype=self.dtype)

        positive = encode(PAPER_PROMPT).repeat(batch_size, 1, 1)
        negative = encode("").repeat(batch_size, 1, 1)
        return torch.cat((negative, positive))

    def _encode(self, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        distribution = self.pipeline.vae.encode(images).latent_dist
        try:
            latents = distribution.sample(generator=generator)
        except TypeError:
            latents = distribution.sample()
        return latents * float(self.pipeline.vae.config.scaling_factor)

    def _reconstruct(
        self, images: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_latents = self._encode(images, generator)
        noise = torch.randn(
            image_latents.shape,
            generator=generator,
            device=self.device,
            dtype=image_latents.dtype,
        )
        alphas = self.pipeline.scheduler.alphas_cumprod.to(self.device)
        initial_step = self.preset.denoise_step - int(_uses_fine_tuned_dino(self.dataset))
        initial_steps = torch.full(
            (len(images),), initial_step, device=self.device, dtype=torch.long
        )
        latents = _add_noise(alphas, image_latents, noise, initial_steps)
        step_ratio = self.preset.denoise_step // self.preset.inference_steps
        if step_ratio <= 0 or self.preset.denoise_step >= len(alphas):
            raise ValueError("Invalid GLAD denoising preset for the scheduler.")
        offset = int(getattr(self.pipeline.scheduler.config, "steps_offset", 0))
        timesteps = torch.arange(self.preset.inference_steps, device=self.device) * step_ratio
        timesteps = timesteps.flip(0).to(torch.long) + offset
        embeddings = self._prompt_embeddings(len(images))
        active = torch.ones(len(images), device=self.device, dtype=torch.bool)
        chosen_steps = torch.zeros(len(images), device=self.device, dtype=torch.long)
        thresholds = torch.zeros(len(images), device=self.device, dtype=torch.float32)
        final_alpha = self.pipeline.scheduler.final_alpha_cumprod

        for timestep_tensor in timesteps:
            timestep = int(timestep_tensor.item())
            model_input = torch.cat((latents, latents))
            model_input = self.pipeline.scheduler.scale_model_input(model_input, timestep_tensor)
            prediction = self.pipeline.unet(
                model_input,
                timestep_tensor,
                encoder_hidden_states=embeddings,
                return_dict=False,
            )
            prediction = (
                prediction[0] if isinstance(prediction, (tuple, list)) else prediction.sample
            )
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + PAPER_GUIDANCE_SCALE * (conditional - unconditional)
            latents, active = _adaptive_ddim_step(
                model_output=prediction,
                timestep=timestep,
                sample=latents,
                image_latents=image_latents,
                noise=noise,
                alphas_cumprod=alphas,
                final_alpha_cumprod=final_alpha,
                step_ratio=step_ratio,
                vae=self.pipeline.vae,
                dino_model=self.ads_dino_model,
                active=active,
                chosen_steps=chosen_steps,
                thresholds=thresholds,
                input_threshold=self.preset.input_threshold,
                min_step=self.preset.min_step,
            )
        return _decode(self.pipeline.vae, latents), chosen_steps

    def _score_batch(
        self, images: torch.Tensor, object_masks: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction, steps = self._reconstruct(images, generator)
        size = self.preset.dino_resolution
        inputs = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=True)
        reconstruction = F.interpolate(
            reconstruction, size=(size, size), mode="bilinear", align_corners=True
        )
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device, dtype=self.dtype)[
            None, :, None, None
        ]
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device, dtype=self.dtype)[
            None, :, None, None
        ]
        normalized_inputs = ((inputs + 1) / 2 - mean) / std
        normalized_reconstruction = ((reconstruction + 1) / 2 - mean) / std
        feature_map = _feature_anomaly_map(
            _dino_patch_tokens(self.dino_model, normalized_inputs),
            _dino_patch_tokens(self.dino_model, normalized_reconstruction),
            output_size=size,
            reverse=self.class_name in _REVERSE_DISTANCE_CLASSES,
        )
        if self.preset.pixel_weight:
            pixel_map = (
                (normalized_inputs - normalized_reconstruction).abs().mean(dim=1, keepdim=True)
            )
            maximum = pixel_map.max()
            if bool(maximum > 0):
                feature_map = feature_map + (
                    self.preset.pixel_weight * feature_map.max() / maximum * pixel_map
                )
        maps = _gaussian_blur(feature_map)[:, 0]
        masks = F.interpolate(
            object_masks[:, None], size=(size, size), mode="bilinear", align_corners=True
        )[:, 0]
        maps = maps * masks
        topk = min(PAPER_IMAGE_TOPK, maps[0].numel())
        scores = torch.topk(maps.flatten(1), k=topk).values.mean(dim=1)
        return scores, maps, steps

    @torch.inference_mode()
    def score_items(
        self, items: Sequence[Any], *, seed: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
        self._ensure_loaded()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        scores: list[NDArray[np.float32]] = []
        maps: list[NDArray[np.float32]] = []
        steps: list[NDArray[np.int64]] = []
        for start in range(0, len(items), self.batch_size):
            images, object_masks = _prepare_images(
                items[start : start + self.batch_size], self.class_name, self.device, self.dtype
            )
            batch_scores, batch_maps, batch_steps = self._score_batch(
                images, object_masks, generator
            )
            scores.append(batch_scores.float().cpu().numpy())
            maps.append(batch_maps.float().cpu().numpy())
            steps.append(batch_steps.cpu().numpy())
        return (
            np.concatenate(scores).astype(np.float32, copy=False),
            np.concatenate(maps).astype(np.float32, copy=False),
            np.concatenate(steps).astype(np.int64, copy=False),
        )


@register_model(
    "vision_glad",
    tags=("vision", "deep", "glad", "diffusion", "reconstruction", "eccv2024"),
    metadata={
        "description": "Checkpoint-backed GLAD SD-v1.4/DINO inference with ADS and SAFF",
        "paper": "GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection",
        "paper_url": "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08940.pdf",
        "official_repository": "https://github.com/hyao1/GLAD",
        "year": 2024,
        "implementation_status": "native-paper-inference-diffusers-adaptation",
        "paper_fidelity": "paper-adaptation",
        "conference": "ECCV",
        "type": "diffusion",
        "requires_checkpoint": True,
        "supports_pixel_map": True,
        "supervision": "one-class",
        "weights_source": "Official GLAD category UNet and optional VAE/DINO checkpoints",
    },
)
class VisionGLAD(BaseDetector):
    """GLAD released inference path; ``fit`` only calibrates the score threshold."""

    def __init__(
        self,
        *,
        dataset: str = "mvtec",
        class_name: str = "bottle",
        pretrained_model_path: str = PAPER_BASE_MODEL,
        checkpoint_path: str | Path | None = None,
        vae_checkpoint_path: str | Path | None = None,
        dino_checkpoint_path: str | Path | None = None,
        pipeline: Any = None,
        dino_model: Any = None,
        ads_dino_model: Any = None,
        backend: Any = None,
        batch_size: int = 2,
        device: str = "cuda",
        random_state: int = 0,
        allow_download: bool = False,
        contamination: float = 0.1,
    ) -> None:
        preset = get_glad_preset(dataset, class_name)
        batch_size_int = validate_batch_size(batch_size)
        assert batch_size_int is not None
        resolved_device = (
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        super().__init__(contamination=contamination)
        self._set_n_classes(None)
        self.dataset = str(dataset)
        self.class_name = str(class_name).lower()
        self.checkpoint_path = checkpoint_path
        self.preset = preset
        self.batch_size = batch_size_int
        self.device = resolved_device
        self.random_state = int(random_state)
        self.backend = (
            backend
            if backend is not None
            else TorchGLADBackend(
                preset=preset,
                class_name=self.class_name,
                dataset=self.dataset,
                pretrained_model_path=pretrained_model_path,
                unet_checkpoint_path=checkpoint_path,
                vae_checkpoint_path=vae_checkpoint_path,
                dino_checkpoint_path=dino_checkpoint_path,
                pipeline=pipeline,
                dino_model=dino_model,
                ads_dino_model=ads_dino_model,
                device=resolved_device,
                batch_size=batch_size_int,
                allow_download=allow_download,
            )
        )

    def _score_items(
        self, items: Sequence[Any]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
        if hasattr(self.backend, "batch_size"):
            result = call_with_temporary_attr(
                self.backend,
                "batch_size",
                self.batch_size,
                lambda: self.backend.score_items(items, seed=self.random_state),
            )
        else:
            result = self.backend.score_items(items, seed=self.random_state)
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("GLAD backend score_items() must return (scores, maps, steps).")
        scores = np.asarray(result[0], dtype=np.float32).reshape(-1)
        maps = np.asarray(result[1], dtype=np.float32)
        steps = np.asarray(result[2], dtype=np.int64).reshape(-1)
        if scores.shape != (len(items),) or maps.ndim != 3 or maps.shape[0] != len(items):
            raise ValueError("GLAD backend returned shapes inconsistent with the input batch.")
        if (
            steps.shape != (len(items),)
            or not np.isfinite(scores).all()
            or not np.isfinite(maps).all()
        ):
            raise ValueError("GLAD backend returned invalid scores, maps, or denoising steps.")
        return scores, maps, steps

    @isolated_random_state_method
    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray[Any]] = None,
        **kwargs: object,
    ) -> "VisionGLAD":
        del y
        values = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        scores, _, _ = self._score_items(_as_items(values))
        self.decision_scores_ = scores
        self._process_decision_scores()
        self.is_fitted_ = True
        return self

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        return self._score_items(_as_items(values))[0]

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        return self._score_items(_as_items(values))[1]

    def get_anomaly_map(self, image: Any) -> NDArray[np.float32]:
        return self.predict_anomaly_map([image])[0]

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        values = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        batch_size_int = validate_batch_size(batch_size)
        if batch_size_int is None:
            return self.predict(values)
        return call_with_temporary_attr(
            self,
            "batch_size",
            batch_size_int,
            lambda: self.predict(values),
        )
