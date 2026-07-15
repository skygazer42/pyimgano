# -*- coding: utf-8 -*-
"""FastFlow 2-D normalizing-flow detector.

The native implementation follows the ResNet variants described in the
FastFlow paper: frozen ImageNet features from residual stages 1--3, one flow
per scale, ActNorm, fixed channel permutations, convolutional affine coupling,
and spatial probability maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .deep_io import safe_torch_load
from .registry import register_model

_PAPER_LAYERS = ("layer1", "layer2", "layer3")
_RESNET_CHANNELS = {
    "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256},
    "wide_resnet50_2": {"layer1": 256, "layer2": 512, "layer3": 1024},
}


def _paper_transform(image_size: int):
    from pyimgano.utils.optional_deps import require

    transforms = require(
        "torchvision.transforms",
        extra="torch",
        purpose="FastFlow image preprocessing",
    )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _spatial_logdet(x: torch.Tensor, logdet: torch.Tensor | None) -> torch.Tensor:
    expected = (int(x.shape[0]), int(x.shape[2]), int(x.shape[3]))
    if logdet is None:
        return x.new_zeros(expected)
    if tuple(logdet.shape) != expected:
        raise ValueError(
            "FastFlow log-determinants must have shape [batch, height, width]; "
            f"got {tuple(logdet.shape)} for {expected}."
        )
    return logdet


class ActNorm2d(nn.Module):
    """Per-channel ActNorm with data-dependent initialization."""

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.eps = float(eps)

    def _initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(self.eps)
            self.bias.copy_(-mean)
            self.log_scale.copy_(-torch.log(std))
            self.initialized.fill_(True)

    def forward(
        self,
        x: torch.Tensor,
        logdet: torch.Tensor | None = None,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.initialized.item()):
            if reverse:
                raise RuntimeError("ActNorm must run forward once before it can be inverted.")
            self._initialize(x)

        logdet = _spatial_logdet(x, logdet)
        channel_logdet = self.log_scale.flatten().sum()
        if reverse:
            return (
                x * torch.exp(-self.log_scale) - self.bias,
                logdet - channel_logdet,
            )
        return (
            (x + self.bias) * torch.exp(self.log_scale),
            logdet + channel_logdet,
        )


class ChannelPermutation(nn.Module):
    """Fixed, checkpointed channel permutation with zero log-determinant."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        permutation = torch.randperm(channels)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(channels)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", inverse)

    def forward(
        self,
        x: torch.Tensor,
        logdet: torch.Tensor | None = None,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = _spatial_logdet(x, logdet)
        indices = self.inverse_permutation if reverse else self.permutation
        return x[:, indices], logdet


class AffineCoupling(nn.Module):
    """Paper-style convolutional affine coupling transformation."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_ratio: float = 1.0,
        kernel_size: int = 3,
        affine_clamp: float | None = 2.0,
    ) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("AffineCoupling channels must be even.")
        if hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be positive.")
        if kernel_size not in (1, 3):
            raise ValueError("FastFlow subnet kernels must be 1 or 3.")
        if affine_clamp is not None and affine_clamp <= 0:
            raise ValueError("affine_clamp must be positive or None.")

        split_channels = channels // 2
        hidden_channels = max(1, int(round(split_channels * hidden_ratio)))
        padding = kernel_size // 2
        self.subnet = nn.Sequential(
            nn.Conv2d(
                split_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )
        self.affine_clamp = affine_clamp

    def _bounded_scale(self, raw_scale: torch.Tensor) -> torch.Tensor:
        if self.affine_clamp is None:
            return raw_scale
        clamp = float(self.affine_clamp)
        return clamp * torch.tanh(raw_scale / clamp)

    def forward(
        self,
        x: torch.Tensor,
        logdet: torch.Tensor | None = None,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = _spatial_logdet(x, logdet)
        unchanged, transformed = torch.chunk(x, 2, dim=1)
        shift, raw_scale = torch.chunk(self.subnet(unchanged), 2, dim=1)
        scale = self._bounded_scale(raw_scale)

        if reverse:
            transformed = (transformed - shift) * torch.exp(-scale)
            logdet = logdet - scale.sum(dim=1)
        else:
            transformed = transformed * torch.exp(scale) + shift
            logdet = logdet + scale.sum(dim=1)
        return torch.cat((unchanged, transformed), dim=1), logdet


class FlowStep(nn.Module):
    """ActNorm, channel permutation, and one affine coupling step."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 1.0,
        *,
        kernel_size: int = 3,
        affine_clamp: float | None = 2.0,
    ) -> None:
        super().__init__()
        self.actnorm = ActNorm2d(channels)
        self.permutation = ChannelPermutation(channels)
        self.coupling = AffineCoupling(
            channels,
            hidden_ratio=hidden_ratio,
            kernel_size=kernel_size,
            affine_clamp=affine_clamp,
        )

    def forward(
        self,
        x: torch.Tensor,
        logdet: torch.Tensor | None = None,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            x, logdet = self.coupling(x, logdet, reverse=True)
            x, logdet = self.permutation(x, logdet, reverse=True)
            return self.actnorm(x, logdet, reverse=True)
        x, logdet = self.actnorm(x, logdet)
        x, logdet = self.permutation(x, logdet)
        return self.coupling(x, logdet)


class FlowStage(nn.Module):
    """A FastFlow stack for one backbone feature scale."""

    def __init__(
        self,
        channels: int,
        n_steps: int,
        hidden_ratio: float,
        *,
        conv3x3_only: bool,
        affine_clamp: float | None,
    ) -> None:
        super().__init__()
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        self.steps = nn.ModuleList(
            [
                FlowStep(
                    channels,
                    hidden_ratio,
                    kernel_size=3 if conv3x3_only or index % 2 == 0 else 1,
                    affine_clamp=affine_clamp,
                )
                for index in range(n_steps)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = x.new_zeros((x.shape[0], x.shape[2], x.shape[3]))
        steps = reversed(self.steps) if reverse else self.steps
        for step in steps:
            x, logdet = step(x, logdet, reverse=reverse)
        return x, logdet


class ResNetFeatureExtractor(nn.Module):
    """Frozen ResNet stages used by the paper's CNN variants."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        layers: Sequence[str] = _PAPER_LAYERS,
    ) -> None:
        super().__init__()
        if backbone == "wide_resnet50":
            backbone = "wide_resnet50_2"
        if backbone not in _RESNET_CHANNELS:
            raise ValueError("FastFlow supports 'resnet18' and 'wide_resnet50_2'.")
        selected_layers = tuple(layers)
        if not selected_layers or len(set(selected_layers)) != len(selected_layers):
            raise ValueError("selected_layers must be a non-empty sequence without duplicates.")
        unsupported = set(selected_layers) - set(_PAPER_LAYERS)
        if unsupported:
            raise ValueError(
                "FastFlow's ResNet path supports paper stages layer1--layer3; "
                f"got {sorted(unsupported)}."
            )

        net, _ = load_torchvision_model(backbone, pretrained=bool(pretrained))
        self.backbone = backbone
        self.selected_layers = selected_layers
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def output_channels(self) -> tuple[int, ...]:
        channels = _RESNET_CHANNELS[self.backbone]
        return tuple(channels[layer] for layer in self.selected_layers)

    def train(self, mode: bool = True):
        del mode
        return super().train(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs: dict[str, torch.Tensor] = {}
        for name in _PAPER_LAYERS:
            x = getattr(self, name)(x)
            if name in self.selected_layers:
                outputs[name] = x
            if len(outputs) == len(self.selected_layers):
                break
        return [outputs[name] for name in self.selected_layers]


@register_model(
    "vision_fastflow",
    tags=("vision", "deep", "flow", "pixel_map"),
    metadata={
        "description": "FastFlow ResNet feature pyramid with 2-D normalizing flows",
        "paper": "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows",
        "paper_url": "https://arxiv.org/abs/2111.07677",
        "year": 2021,
        "supervision": "one-class",
        "implementation_status": "paper-resnet-network-and-objective-aligned",
        "paper_fidelity": "paper-adaptation",
    },
)
class FastFlow(BaseVisionDeepDetector):
    """FastFlow's paper-defined ResNet architecture and spatial objective.

    The paper leaves affine scale stabilization, probability-map normalization,
    image-score reduction, and the exact training rotation range unspecified.
    This implementation uses an explicit clamp, channel-normalized probability
    maps, max-pixel image scores, and no category-dependent augmentation. The
    offline-safe default also leaves ImageNet weights disabled; set
    ``pretrained_backbone=True`` for the paper training protocol.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = False,
        selected_layers: Sequence[str] = _PAPER_LAYERS,
        image_size: int = 256,
        n_flow_steps: int = 8,
        flow_hidden_ratio: float = 1.0,
        conv3x3_only: bool | None = None,
        affine_clamp: float | None = 2.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epoch_num: int = 500,
        batch_size: int = 32,
        device: str | None = None,
        verbose: int = 1,
        random_state: int = 42,
        train_transform: object | None = None,
        eval_transform: object | None = None,
    ) -> None:
        if backbone == "wide_resnet50":
            backbone = "wide_resnet50_2"
        if backbone not in _RESNET_CHANNELS:
            raise ValueError("FastFlow supports 'resnet18' and 'wide_resnet50_2'.")
        if image_size <= 0 or n_flow_steps <= 0:
            raise ValueError("image_size and n_flow_steps must be positive.")
        if flow_hidden_ratio <= 0 or lr <= 0 or weight_decay < 0:
            raise ValueError("Flow ratio/lr must be positive and weight_decay non-negative.")

        self.backbone = backbone
        self.pretrained_backbone = bool(pretrained_backbone)
        self.selected_layers = tuple(selected_layers)
        self.image_size = int(image_size)
        self.n_flow_steps = int(n_flow_steps)
        self.flow_hidden_ratio = float(flow_hidden_ratio)
        self.conv3x3_only = backbone == "resnet18" if conv3x3_only is None else bool(conv3x3_only)
        self.affine_clamp = affine_clamp

        train_transform = train_transform or _paper_transform(self.image_size)
        eval_transform = eval_transform or _paper_transform(self.image_size)
        super().__init__(
            contamination=contamination,
            preprocessing=True,
            lr=lr,
            epoch_num=epoch_num,
            batch_size=batch_size,
            optimizer_name="adam",
            criterion_name="mse",
            device=device,
            random_state=random_state,
            verbose=verbose,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
        self.weight_decay = float(weight_decay)

    def _checkpoint_config(self) -> dict[str, object]:
        return {
            "backbone": self.backbone,
            "selected_layers": list(self.selected_layers),
            "image_size": self.image_size,
            "n_flow_steps": self.n_flow_steps,
            "flow_hidden_ratio": self.flow_hidden_ratio,
            "conv3x3_only": self.conv3x3_only,
            "affine_clamp": self.affine_clamp,
            "architecture": "fastflow-resnet-2d-flow-v2",
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        self._check_is_fitted()
        if getattr(self, "model", None) is None or not hasattr(self, "feature_extractor"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "pyimgano.fastflow",
                "schema_version": 2,
                "config": self._checkpoint_config(),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in self.model.state_dict().items()
                },
                "feature_extractor_state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.feature_extractor.state_dict().items()
                },
                "decision_scores": torch.as_tensor(
                    np.asarray(self.decision_scores_, dtype=np.float64),
                    dtype=torch.float64,
                ),
                "threshold": float(self.threshold_),
                "labels": (
                    None
                    if getattr(self, "labels_", None) is None
                    else torch.as_tensor(np.asarray(self.labels_, dtype=np.int64))
                ),
            },
            output,
        )
        return output

    def load_checkpoint(self, path: str | Path) -> None:
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format") != "pyimgano.fastflow":
            raise ValueError(
                "Unsupported legacy FastFlow checkpoint; refit with the paper-aligned "
                "ResNet 2-D flow architecture."
            )
        if int(payload.get("schema_version", -1)) != 2:
            raise ValueError("Unsupported FastFlow checkpoint schema version.")
        if payload.get("config") != self._checkpoint_config():
            raise ValueError("FastFlow checkpoint configuration does not match the detector.")

        if getattr(self, "model", None) is None or not hasattr(self, "feature_extractor"):
            self.model = self.build_model()
        model_state = payload.get("model_state_dict")
        extractor_state = payload.get("feature_extractor_state_dict")
        if not isinstance(model_state, dict) or not isinstance(extractor_state, dict):
            raise ValueError("FastFlow checkpoint is missing required state dictionaries.")
        self.model.load_state_dict(model_state, strict=True)
        self.feature_extractor.load_state_dict(extractor_state, strict=True)
        self.model.to(self.device).eval()
        self.feature_extractor.to(self.device).eval()

        self.decision_scores_ = np.asarray(payload["decision_scores"], dtype=np.float64)
        self.threshold_ = float(payload["threshold"])
        labels = payload.get("labels")
        if labels is not None:
            self.labels_ = np.asarray(labels, dtype=np.int64)
        self.is_fitted_ = True

    def build_model(self) -> nn.ModuleList:
        self.feature_extractor = ResNetFeatureExtractor(
            backbone=self.backbone,
            pretrained=self.pretrained_backbone,
            layers=self.selected_layers,
        ).to(self.device)
        self.feature_extractor.eval()

        self.flow_stages = nn.ModuleList(
            [
                FlowStage(
                    channels,
                    self.n_flow_steps,
                    self.flow_hidden_ratio,
                    conv3x3_only=self.conv3x3_only,
                    affine_clamp=self.affine_clamp,
                )
                for channels in self.feature_extractor.output_channels
            ]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.flow_stages.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return self.flow_stages

    def _extract_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        self.feature_extractor.eval()
        with torch.no_grad():
            return [feature.detach() for feature in self.feature_extractor(images)]

    @staticmethod
    def _flow_nll(z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        dimensions = int(z.shape[1] * z.shape[2] * z.shape[3])
        latent_energy = 0.5 * z.square().sum(dim=(1, 2, 3))
        return (latent_energy - logdet.sum(dim=(1, 2))) / dimensions

    @staticmethod
    def _latent_anomaly_map(z: torch.Tensor) -> torch.Tensor:
        normal_probability = torch.exp(-0.5 * z.square()).mean(dim=1)
        return 1.0 - normal_probability

    def _anomaly_maps(
        self,
        images: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        features = self._extract_features(images)
        self.flow_stages.eval()
        maps = []
        for feature, flow in zip(features, self.flow_stages):
            latent, _ = flow(feature)
            stage_map = self._latent_anomaly_map(latent).unsqueeze(1)
            maps.append(
                F.interpolate(
                    stage_map,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            )
        return torch.stack(maps, dim=0).mean(dim=0)

    def training_forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        images, _ = batch
        images = images.to(self.device)
        features = self._extract_features(images)

        self.flow_stages.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        for feature, flow in zip(features, self.flow_stages):
            latent, logdet = flow(feature)
            losses.append(self._flow_nll(latent, logdet).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluating_forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> NDArray:
        images, _ = batch
        images = images.to(self.device)
        maps = self._anomaly_maps(
            images,
            output_size=(int(images.shape[-2]), int(images.shape[-1])),
        )
        return maps.flatten(start_dim=1).amax(dim=1).cpu().numpy()

    @torch.no_grad()
    def get_anomaly_map(self, image: str | Path | NDArray) -> NDArray:
        self._check_is_fitted()
        from PIL import Image

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected a uint8 RGB image with shape (H, W, 3).")
            original_size = (int(image.shape[0]), int(image.shape[1]))
            pil_image = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        else:
            with Image.open(str(image)) as opened:
                pil_image = opened.convert("RGB")
            original_size = (int(pil_image.height), int(pil_image.width))

        tensor = self.eval_transform(pil_image).unsqueeze(0).to(self.device)
        anomaly_map = self._anomaly_maps(tensor, output_size=original_size)
        return anomaly_map[0].cpu().numpy().astype(np.float32, copy=False)

    def predict_anomaly_map(
        self,
        x: object = MISSING,
        **kwargs: object,
    ) -> NDArray:
        items = list(resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"))
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(item) for item in items]
        if len({anomaly_map.shape for anomaly_map in maps}) != 1:
            raise ValueError("All images must share a size to stack anomaly maps.")
        return np.stack(maps, axis=0)

    def fit(self, x: object = MISSING, y: Iterable[int] | None = None, **kwargs: object):
        return super().fit(resolve_legacy_x_keyword(x, kwargs, method_name="fit"), y)
