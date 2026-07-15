"""RealNet feature reconstruction for image anomaly detection.

Paper: "RealNet: A Feature Selection Network with Realistic Synthetic Anomaly
for Anomaly Detection" (CVPR 2024).

The detector path implements AFS, independent multi-scale reconstruction, and
RRS.  SDAS is the paper's separate offline diffusion stage, so fitting requires
paired synthetic anomaly images and masks instead of silently substituting a
different image corruption method.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import functional as tv_functional

from pyimgano.models._imagenet_preprocess import preprocess_imagenet_batch
from pyimgano.utils.random_state import isolated_random_state_method
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._image_batch import coerce_rgb_image_batch
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, int(channels))
    while int(channels) % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


class RealNetFeatureExtractor(nn.Module):
    """Frozen ImageNet ResNet block outputs used by RealNet."""

    _CHANNELS = {
        "resnet18": (64, 128, 256, 512),
        "resnet34": (64, 128, 256, 512),
        "resnet50": (256, 512, 1024, 2048),
        "wide_resnet50": (256, 512, 1024, 2048),
        "wide_resnet50_2": (256, 512, 1024, 2048),
    }

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        *,
        pretrained: bool = True,
        weights_name: str = "IMAGENET1K_V1",
    ) -> None:
        super().__init__()
        if backbone not in self._CHANNELS:
            raise ValueError(f"Unsupported RealNet backbone: {backbone!r}.")
        model, _ = load_torchvision_model(
            backbone,
            pretrained=pretrained,
            weights_name=weights_name if pretrained else None,
        )
        self.backbone = model
        self.out_channels = self._CHANNELS[backbone]
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        model = self.backbone
        with torch.no_grad():
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            f1 = model.layer1(x)
            f2 = model.layer2(f1)
            f3 = model.layer3(f2)
            f4 = model.layer4(f3)
        return f1, f2, f3, f4


class AnomalyAwareFeatureSelection(nn.Module):
    """Cache the per-layer channel indexes selected by the paper's AFS loss."""

    def __init__(self, in_channels: Sequence[int], selected_channels: Sequence[int]) -> None:
        super().__init__()
        if len(in_channels) != 4 or len(selected_channels) != 4:
            raise ValueError("RealNet AFS expects four feature levels.")
        self.in_channels = tuple(int(value) for value in in_channels)
        self.selected_channels = tuple(int(value) for value in selected_channels)
        for index, (available, selected) in enumerate(
            zip(self.in_channels, self.selected_channels)
        ):
            if selected <= 0 or selected > available:
                raise ValueError(
                    f"selected_channels[{index}] must be in [1, {available}], got {selected}."
                )
            self.register_buffer(f"indexes_{index}", torch.arange(selected, dtype=torch.long))

    def set_indexes(self, level: int, indexes: torch.Tensor) -> None:
        target = getattr(self, f"indexes_{int(level)}")
        if len(indexes) != len(target):
            raise ValueError("AFS index count does not match selected_channels.")
        target.copy_(indexes.to(device=target.device, dtype=torch.long))

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.index_select(feature, 1, getattr(self, f"indexes_{index}"))
            for index, feature in enumerate(features)
        )


class _RealNetResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        *,
        up: bool = False,
        down: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels or channels)
        self.up = bool(up)
        self.down = bool(down)
        self.norm = _group_norm(self.channels)
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        if self.up:
            self.h_scale = nn.ConvTranspose2d(self.channels, self.channels, 4, 2, 1)
            self.x_scale = nn.ConvTranspose2d(self.channels, self.channels, 4, 2, 1)
        elif self.down:
            self.h_scale = nn.AvgPool2d(2, 2)
            self.x_scale = nn.AvgPool2d(2, 2)
        else:
            self.h_scale = self.x_scale = nn.Identity()
        self.skip = (
            nn.Identity()
            if self.channels == self.out_channels
            else nn.Conv2d(self.channels, self.out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.norm(x))
        if self.up or self.down:
            h = self.h_scale(h)
            x = self.x_scale(x)
        return self.skip(x) + self.conv(h)


class _RealNetAttention(nn.Module):
    def __init__(self, channels: int, head_channels: int = 64) -> None:
        super().__init__()
        if channels % head_channels:
            raise ValueError(
                f"Attention channels ({channels}) must be divisible by head_channels ({head_channels})."
            )
        self.channels = int(channels)
        self.heads = int(channels // head_channels)
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.projection = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, *spatial = x.shape
        flat = x.reshape(batch, channels, -1)
        qkv = self.qkv(self.norm(flat))
        head_channels = channels // self.heads
        q, k, v = qkv.reshape(batch * self.heads, head_channels * 3, -1).split(head_channels, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(head_channels))
        weights = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weights = torch.softmax(weights.float(), dim=-1).to(dtype=weights.dtype)
        attended = torch.einsum("bts,bcs->bct", weights, v)
        attended = attended.reshape(batch, channels, -1)
        return (flat + self.projection(attended)).reshape(batch, channels, *spatial)


class RealNetReconstructionUNet(nn.Module):
    """Paper reconstruction U-Net used independently at one feature scale."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_ratio: float = 0.5,
        channel_mult: Sequence[int] = (1, 2, 4),
        attention_mult: Sequence[int] = (2, 4),
        num_res_blocks: int = 2,
        attention_head_channels: int = 64,
    ) -> None:
        super().__init__()
        model_channels = int(float(hidden_ratio) * int(channels))
        if model_channels <= 0:
            raise ValueError("hidden_ratio produces zero reconstruction channels.")
        multipliers = tuple(int(value) for value in channel_mult)
        attention_scales = {int(value) for value in attention_mult}
        if not multipliers or int(num_res_blocks) <= 0:
            raise ValueError("channel_mult and num_res_blocks must be non-empty and positive.")

        current = input_channels = multipliers[0] * model_channels
        self.input_blocks = nn.ModuleList([nn.Conv2d(channels, current, 3, padding=1)])
        skip_channels = [current]
        downsample = 1
        for level, multiplier in enumerate(multipliers):
            for _ in range(int(num_res_blocks)):
                output = multiplier * model_channels
                layers: list[nn.Module] = [_RealNetResBlock(current, output)]
                current = output
                if downsample in attention_scales:
                    layers.append(_RealNetAttention(current, attention_head_channels))
                self.input_blocks.append(nn.Sequential(*layers))
                skip_channels.append(current)
            if level != len(multipliers) - 1:
                self.input_blocks.append(_RealNetResBlock(current, current, down=True))
                skip_channels.append(current)
                downsample *= 2

        self.middle = nn.Sequential(
            _RealNetResBlock(current),
            _RealNetAttention(current, attention_head_channels),
            _RealNetResBlock(current),
        )
        self.output_blocks = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(multipliers))):
            for block_index in range(int(num_res_blocks) + 1):
                skip = skip_channels.pop()
                output = multiplier * model_channels
                layers = [_RealNetResBlock(current + skip, output)]
                current = output
                if downsample in attention_scales:
                    layers.append(_RealNetAttention(current, attention_head_channels))
                if level and block_index == int(num_res_blocks):
                    layers.append(_RealNetResBlock(current, current, up=True))
                    downsample //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        output_conv = nn.Conv2d(input_channels, channels, 3, padding=1)
        nn.init.zeros_(output_conv.weight)
        nn.init.zeros_(output_conv.bias)
        self.output = nn.Sequential(_group_norm(current), nn.SiLU(), output_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for block in self.input_blocks:
            h = block(h)
            skips.append(h)
        h = self.middle(h)
        for block in self.output_blocks:
            h = block(torch.cat((h, skips.pop()), dim=1))
        return self.output(h)


class RealNetReconstruction(nn.Module):
    def __init__(self, channels: Sequence[int], **kwargs: object) -> None:
        super().__init__()
        self.networks = nn.ModuleList(
            [RealNetReconstructionUNet(int(value), **kwargs) for value in channels]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(network(feature) for network, feature in zip(self.networks, features))


class _ResidualStack(nn.Module):
    def __init__(self, channels: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 1, bias=False),
                )
                for _ in range(int(layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return F.relu(x)


class ReconstructionResidualSelection(nn.Module):
    """Paper RRS: per-sample max/mean Top-K residuals plus pixel decoder."""

    def __init__(
        self,
        channels: Sequence[int],
        *,
        modes: Sequence[str] = ("max", "mean"),
        mode_numbers: Sequence[int] = (256, 256),
        num_residual_layers: int = 2,
    ) -> None:
        super().__init__()
        self.modes = tuple(str(mode) for mode in modes)
        self.mode_numbers = tuple(int(value) for value in mode_numbers)
        if len(self.modes) != len(self.mode_numbers) or any(
            mode not in {"max", "mean"} for mode in self.modes
        ):
            raise ValueError("RRS modes must pair 'max'/'mean' with mode_numbers.")
        total_channels = sum(int(value) for value in channels)
        if any(value <= 0 for value in self.mode_numbers) or any(
            value > total_channels for value in self.mode_numbers
        ):
            raise ValueError("Each RRS mode number must fit the concatenated residual channels.")
        selected = sum(self.mode_numbers)
        self.selector_norm = nn.BatchNorm2d(total_channels, momentum=0.9, affine=False)
        self.decoder1 = nn.Sequential(
            _ResidualStack(selected, int(num_residual_layers)),
            nn.Conv2d(selected, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, 3, padding=1),
        )

    def forward(
        self, residuals: Sequence[torch.Tensor], output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        align_size = residuals[0].shape[-2:]
        aligned = [
            residual
            if residual.shape[-2:] == align_size
            else F.interpolate(residual, align_size, mode="bilinear", align_corners=True)
            for residual in residuals
        ]
        residual = torch.cat(aligned, dim=1)
        normalized = self.selector_norm(residual)
        batch, _, height, width = residual.shape
        choices: list[torch.Tensor] = []
        for mode, count in zip(self.modes, self.mode_numbers):
            pooled = normalized.flatten(2)
            pooled = pooled.amax(dim=2) if mode == "max" else pooled.mean(dim=2)
            indexes = pooled.topk(count, dim=1, largest=True, sorted=True).indices
            choices.append(
                torch.gather(
                    residual,
                    1,
                    indexes[:, :, None, None].expand(batch, count, height, width),
                )
            )
        decoded = self.decoder2(self.decoder1(torch.cat(choices, dim=1)))
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=True)
        logits = self.decoder3(decoded)
        logits = F.interpolate(logits, output_size, mode="bilinear", align_corners=True)
        return logits, torch.softmax(logits, dim=1)[:, 1:2]


class RealNetModel(nn.Module):
    """Frozen backbone + AFS + reconstruction + RRS."""

    def __init__(
        self,
        *,
        backbone: str,
        pretrained: bool,
        weights_name: str,
        selected_channels: Sequence[int],
        hidden_ratio: float,
        channel_mult: Sequence[int],
        attention_mult: Sequence[int],
        num_res_blocks: int,
        attention_head_channels: int,
        rrs_modes: Sequence[str],
        rrs_mode_numbers: Sequence[int],
        rrs_num_residual_layers: int,
    ) -> None:
        super().__init__()
        self.feature_extractor = RealNetFeatureExtractor(
            backbone, pretrained=pretrained, weights_name=weights_name
        )
        self.afs = AnomalyAwareFeatureSelection(
            self.feature_extractor.out_channels, selected_channels
        )
        self.reconstruction = RealNetReconstruction(
            selected_channels,
            hidden_ratio=hidden_ratio,
            channel_mult=channel_mult,
            attention_mult=attention_mult,
            num_res_blocks=num_res_blocks,
            attention_head_channels=attention_head_channels,
        )
        self.rrs = ReconstructionResidualSelection(
            selected_channels,
            modes=rrs_modes,
            mode_numbers=rrs_mode_numbers,
            num_residual_layers=rrs_num_residual_layers,
        )

    def forward(self, image: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, object]:
        selected = self.afs(self.feature_extractor(image))
        reconstructed = self.reconstruction(selected)
        residuals = tuple(
            (feature - reconstruction).square()
            for feature, reconstruction in zip(selected, reconstructed)
        )
        logits, anomaly_map = self.rrs(residuals, tuple(int(v) for v in image.shape[-2:]))
        output: dict[str, object] = {
            "selected": selected,
            "reconstructed": reconstructed,
            "residuals": residuals,
            "logits": logits,
            "anomaly_map": anomaly_map,
        }
        if target is not None:
            output["target_selected"] = self.afs(self.feature_extractor(target))
        return output


@register_model(
    "vision_realnet",
    tags=("vision", "deep", "realnet", "feature-reconstruction", "pixel_map", "paper"),
    metadata={
        "description": "RealNet AFS, multi-scale reconstruction, and RRS detector path with external SDAS pairs",
        "paper": "RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.html",
        "year": 2024,
        "implementation_status": "paper-afs-reconstruction-rrs-path-aligned-external-sdas",
        "paper_fidelity": "paper-adaptation",
        "conference": "CVPR",
        "type": "feature-reconstruction",
        "supports_pixel_map": True,
    },
)
class VisionRealNet(BaseVisionDeepDetector):
    """Paper-aligned RealNet detector path.

    The published defaults use WideResNet50-2 features, selected dimensions
    ``(256, 512, 512, 256)``, four independent reconstruction U-Nets, RRS
    ``max/mean`` Top-K dimensions ``(256, 256)``, 256px inputs, batch size 16,
    Adam at ``1e-4``, and 1,000 epochs.

    ``fit`` requires already blended synthetic anomaly images and masks paired
    with the supplied normal images. Use the authors' SIA/SDAS data to reproduce
    the paper; generic local corruptions are deliberately not substituted.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        *,
        pretrained: bool = True,
        weights_name: str = "IMAGENET1K_V1",
        selected_channels: Sequence[int] = (256, 512, 512, 256),
        hidden_ratio: float = 0.5,
        channel_mult: Sequence[int] = (1, 2, 4),
        attention_mult: Sequence[int] = (2, 4),
        num_res_blocks: int = 2,
        attention_head_channels: int = 64,
        rrs_modes: Sequence[str] = ("max", "mean"),
        rrs_mode_numbers: Sequence[int] = (256, 256),
        rrs_num_residual_layers: int = 2,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 1000,
        afs_batches: int = 64,
        image_size: int = 256,
        image_score_pool_size: int = 16,
        contamination: float = 0.1,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        if int(batch_size) <= 0 or int(epochs) <= 0 or int(afs_batches) <= 0:
            raise ValueError("batch_size, epochs, and afs_batches must be positive.")
        if float(learning_rate) <= 0:
            raise ValueError("learning_rate must be positive.")
        if (
            int(num_res_blocks) <= 0
            or int(attention_head_channels) <= 0
            or int(rrs_num_residual_layers) <= 0
        ):
            raise ValueError("Residual-layer and attention dimensions must be positive.")
        if int(image_size) <= 0 or int(image_score_pool_size) <= 0:
            raise ValueError("image_size and image_score_pool_size must be positive.")
        resolved_device = device
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        super().__init__(
            contamination=contamination,
            batch_size=int(batch_size),
            device=resolved_device,
            random_state=random_state,
            **kwargs,
        )
        self.backbone = str(backbone)
        self.pretrained = bool(pretrained)
        self.weights_name = str(weights_name)
        self.selected_channels = tuple(int(value) for value in selected_channels)
        self.hidden_ratio = float(hidden_ratio)
        self.channel_mult = tuple(int(value) for value in channel_mult)
        self.attention_mult = tuple(int(value) for value in attention_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_head_channels = int(attention_head_channels)
        self.rrs_modes = tuple(str(value) for value in rrs_modes)
        self.rrs_mode_numbers = tuple(int(value) for value in rrs_mode_numbers)
        self.rrs_num_residual_layers = int(rrs_num_residual_layers)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.afs_batches = int(afs_batches)
        self.image_size = int(image_size)
        self.image_score_pool_size = int(image_score_pool_size)
        self.random_state = random_state
        self.model: RealNetModel | None = None
        self.is_fitted_ = False

    def _build_model(self) -> RealNetModel:
        return RealNetModel(
            backbone=self.backbone,
            pretrained=self.pretrained,
            weights_name=self.weights_name,
            selected_channels=self.selected_channels,
            hidden_ratio=self.hidden_ratio,
            channel_mult=self.channel_mult,
            attention_mult=self.attention_mult,
            num_res_blocks=self.num_res_blocks,
            attention_head_channels=self.attention_head_channels,
            rrs_modes=self.rrs_modes,
            rrs_mode_numbers=self.rrs_mode_numbers,
            rrs_num_residual_layers=self.rrs_num_residual_layers,
        ).to(self.device)

    def _preprocess(self, x: object) -> torch.Tensor:
        images = coerce_rgb_image_batch(x)
        resized = [
            np.asarray(
                tv_functional.resize(
                    tv_functional.to_pil_image(image),
                    [self.image_size, self.image_size],
                )
            )
            for image in images
        ]
        return preprocess_imagenet_batch(np.stack(resized))

    def _preprocess_masks(self, masks: object, count: int) -> torch.Tensor:
        array = np.asarray(masks)
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 3 or len(array) != int(count):
            raise ValueError(f"synthetic_masks must have shape ({count}, H, W).")
        tensor = torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(1)
        if not bool(torch.isfinite(tensor).all()) or float(tensor.min()) < 0.0:
            raise ValueError("synthetic_masks must contain finite non-negative values.")
        if float(tensor.max()) > 1.0:
            tensor /= 255.0
        if float(tensor.max()) > 1.0:
            raise ValueError("synthetic_masks must use binary, [0,1], or uint8 scale.")
        tensor = F.interpolate(tensor, (self.image_size, self.image_size), mode="nearest")
        return (tensor > 0.5).to(dtype=torch.float32)

    def _initialize_afs(
        self,
        anomaly_images: torch.Tensor,
        target_images: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        if self.model is None:
            raise RuntimeError("RealNet model is not initialized.")
        loader = DataLoader(
            TensorDataset(anomaly_images, target_images, masks),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        scores = [
            torch.zeros(channels, device=self.device)
            for channels in self.model.feature_extractor.out_channels
        ]
        self.model.feature_extractor.eval()
        iterator = iter(loader)
        with torch.no_grad():
            for _ in range(self.afs_batches):
                try:
                    anomaly, target, mask = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    anomaly, target, mask = next(iterator)
                anomaly_features = self.model.feature_extractor(anomaly.to(self.device))
                target_features = self.model.feature_extractor(target.to(self.device))
                mask = mask.to(self.device)
                for level, (anomaly_feature, target_feature) in enumerate(
                    zip(anomaly_features, target_features)
                ):
                    residual = (anomaly_feature - target_feature).square()
                    batch, channels, height, width = residual.shape
                    flat = residual.permute(1, 0, 2, 3).reshape(channels, -1)
                    minimum = flat.amin(dim=1, keepdim=True)
                    maximum = flat.amax(dim=1, keepdim=True)
                    flat = (flat - minimum) / (maximum - minimum + 1e-4)
                    label = F.interpolate(mask, (height, width), mode="nearest")
                    label = label.permute(1, 0, 2, 3).reshape(1, batch * height * width)
                    scores[level] += (flat - label).square().mean(dim=1)

        for level, (level_scores, count) in enumerate(zip(scores, self.selected_channels)):
            level_scores = torch.nan_to_num(level_scores, nan=torch.inf)
            indexes = level_scores.topk(int(count), largest=False).indices.sort().values
            self.model.afs.set_indexes(level, indexes)

    @staticmethod
    def _loss(output: dict[str, object], masks: torch.Tensor) -> torch.Tensor:
        reconstructed = output["reconstructed"]
        target = output["target_selected"]
        logits = output["logits"]
        if not isinstance(reconstructed, tuple) or not isinstance(target, tuple):
            raise RuntimeError("RealNet training output is incomplete.")
        reconstruction_loss = torch.stack(
            [F.mse_loss(actual, expected) for actual, expected in zip(reconstructed, target)]
        ).sum()
        return reconstruction_loss + F.cross_entropy(logits, masks[:, 0].long())

    def _predict_maps_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("RealNet model is not initialized.")
        self.model.eval()
        maps: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                output = self.model(tensor[start : start + self.batch_size].to(self.device))
                anomaly_map = output["anomaly_map"]
                if not isinstance(anomaly_map, torch.Tensor):
                    raise RuntimeError("RealNet did not return an anomaly map.")
                maps.append(anomaly_map.cpu())
        return torch.cat(maps, dim=0)

    def _image_scores(self, maps: torch.Tensor) -> NDArray[np.float32]:
        kernel = min(self.image_score_pool_size, int(maps.shape[-2]), int(maps.shape[-1]))
        pooled = F.avg_pool2d(maps, kernel_size=kernel, stride=1)
        return pooled.flatten(1).amax(dim=1).numpy().astype(np.float32, copy=False)

    @isolated_random_state_method
    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        *,
        synthetic_images: object | None = None,
        synthetic_masks: object | None = None,
        **kwargs: object,
    ) -> "VisionRealNet":
        values = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        if synthetic_images is None or synthetic_masks is None:
            raise ValueError(
                "Paper-aligned RealNet requires paired synthetic_images and "
                "synthetic_masks from SDAS/SIA (or an explicitly chosen paper ablation)."
            )
        normal = self._preprocess(values)
        anomaly = self._preprocess(synthetic_images)
        if len(anomaly) != len(normal):
            raise ValueError("synthetic_images must be paired one-to-one with normal images.")
        masks = self._preprocess_masks(synthetic_masks, len(normal))

        self.model = self._build_model()
        self._initialize_afs(anomaly, normal, masks)
        train_images = torch.cat((normal, anomaly), dim=0)
        train_targets = torch.cat((normal, normal), dim=0)
        train_masks = torch.cat((torch.zeros_like(masks), masks), dim=0)
        loader = DataLoader(
            TensorDataset(train_images, train_targets, train_masks),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(
            list(self.model.reconstruction.parameters()) + list(self.model.rrs.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
        )
        last_loss = 0.0
        for _ in range(self.epochs):
            self.model.train()
            self.model.feature_extractor.eval()
            for image, target, mask in loader:
                image = image.to(self.device)
                target = target.to(self.device)
                mask = mask.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self.model(image, target), mask)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

        logger.info("RealNet finished %d epochs; loss=%.6f", self.epochs, last_loss)
        self.training_loss_ = last_loss
        maps = self._predict_maps_tensor(normal)
        self.decision_scores_ = self._image_scores(maps).astype(np.float64)
        self._process_decision_scores()
        self._set_n_classes(y)
        self.is_fitted_ = True
        return self

    def predict_anomaly_map(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        self._check_is_fitted()
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        old_batch_size = self.batch_size
        if batch_size is not None:
            if int(batch_size) <= 0:
                raise ValueError("batch_size must be positive.")
            self.batch_size = int(batch_size)
        try:
            maps = self._predict_maps_tensor(self._preprocess(values))
            return maps[:, 0].numpy().astype(np.float32, copy=False)
        finally:
            self.batch_size = old_batch_size

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
        return self._image_scores(torch.from_numpy(self.predict_anomaly_map(values)).unsqueeze(1))

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        values = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        if batch_size is None:
            return self.predict(values)
        old_batch_size = self.batch_size
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        try:
            self.batch_size = int(batch_size)
            return self.predict(values)
        finally:
            self.batch_size = old_batch_size


__all__ = [
    "AnomalyAwareFeatureSelection",
    "RealNetFeatureExtractor",
    "RealNetModel",
    "RealNetReconstructionUNet",
    "ReconstructionResidualSelection",
    "VisionRealNet",
]
