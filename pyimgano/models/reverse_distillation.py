# -*- coding: utf-8 -*-
"""Reverse Distillation from a one-class bottleneck embedding.

The default network follows the authors' CVPR 2022 implementation: a frozen
ImageNet WideResNet50-2 teacher, its three first residual stages, the OCBE
bottleneck, and a reverse WideResNet50-2 decoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from pyimgano.utils.random_state import isolated_random_state
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

_PAPER_BACKBONE = "wide_resnet50_2"
_PAPER_LAYERS = ("layer1", "layer2", "layer3")


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def _deconv2x2(in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)


def _init_author_residual_modules(module: nn.Module) -> None:
    """Match the initialization loops in the authors' encoder-side modules."""

    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)


class WideResNet50Encoder(nn.Module):
    """Frozen full WideResNet50-2 teacher returning stages 1--3."""

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        net, _ = load_torchvision_model(_PAPER_BACKBONE, pretrained=pretrained)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        # Retained for full author-checkpoint/parameter parity although forward stops at layer3.
        self.layer4 = net.layer4
        self.avgpool = net.avgpool
        self.fc = net.fc
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        feature1 = self.layer1(features)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        return [feature1, feature2, feature3]


class WideResidualBottleneck(nn.Module):
    """WideResNet bottleneck used by the author's OCBE module."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        width_per_group: int = 128,
    ) -> None:
        super().__init__()
        width = int(planes * (width_per_group / 64.0))
        self.conv1 = _conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity = features
        output = self.relu(self.bn1(self.conv1(features)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.downsample is not None:
            identity = self.downsample(features)
        return self.relu(output + identity)


class OneClassBottleneck(nn.Module):
    """Authors' one-class bottleneck embedding (OCBE)."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 1024
        self.bn_layer = self._make_layer(planes=512, blocks=3, stride=2)
        self.conv1 = _conv3x3(256, 512, 2)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(512, 1024, 2)
        self.bn2 = nn.BatchNorm2d(1024)
        self.conv3 = _conv3x3(512, 1024, 2)
        self.bn3 = nn.BatchNorm2d(1024)
        # Present in the released OCBE checkpoint even though its forward path is unused.
        self.conv4 = _conv1x1(4096, 2048)
        self.bn4 = nn.BatchNorm2d(2048)
        _init_author_residual_modules(self)

    def _make_layer(self, *, planes: int, blocks: int, stride: int) -> nn.Sequential:
        output_channels = planes * WideResidualBottleneck.expansion
        fused_channels = self.in_channels * 3
        downsample = nn.Sequential(
            _conv1x1(fused_channels, output_channels, stride),
            nn.BatchNorm2d(output_channels),
        )
        layers = [
            WideResidualBottleneck(
                fused_channels,
                planes,
                stride=stride,
                downsample=downsample,
            )
        ]
        layers.extend(WideResidualBottleneck(output_channels, planes) for _ in range(1, blocks))
        self.in_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError(f"Expected three encoder features, got {len(features)}.")
        low = self.relu(self.bn1(self.conv1(features[0])))
        low = self.relu(self.bn2(self.conv2(low)))
        mid = self.relu(self.bn3(self.conv3(features[1])))
        high = features[2]
        if low.shape[-2:] != high.shape[-2:] or mid.shape[-2:] != high.shape[-2:]:
            raise RuntimeError("Reverse Distillation feature scales are inconsistent.")
        return self.bn_layer(torch.cat([low, mid, high], dim=1)).contiguous()


class ReverseWideBottleneck(nn.Module):
    """Wide residual bottleneck with the author's stride-2 transposed convolution."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        *,
        stride: int = 1,
        upsample: nn.Module | None = None,
        width_per_group: int = 128,
    ) -> None:
        super().__init__()
        width = int(planes * (width_per_group / 64.0))
        self.conv1 = _conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2: nn.Module = _deconv2x2(width, width) if stride == 2 else _conv3x3(width, width)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity = features
        output = self.relu(self.bn1(self.conv1(features)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.upsample is not None:
            identity = self.upsample(features)
        return self.relu(output + identity)


class ReverseWideResNet50Decoder(nn.Module):
    """Authors' reverse WideResNet50-2 decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 2048
        self.layer1 = self._make_layer(planes=256, blocks=3)
        self.layer2 = self._make_layer(planes=128, blocks=4)
        self.layer3 = self._make_layer(planes=64, blocks=6)
        _init_author_residual_modules(self)

    def _make_layer(self, *, planes: int, blocks: int) -> nn.Sequential:
        output_channels = planes * ReverseWideBottleneck.expansion
        upsample = nn.Sequential(
            _deconv2x2(self.in_channels, output_channels),
            nn.BatchNorm2d(output_channels),
        )
        layers = [
            ReverseWideBottleneck(
                self.in_channels,
                planes,
                stride=2,
                upsample=upsample,
            )
        ]
        layers.extend(ReverseWideBottleneck(output_channels, planes) for _ in range(1, blocks))
        self.in_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        feature3 = self.layer1(embedding)
        feature2 = self.layer2(feature3)
        feature1 = self.layer3(feature2)
        return [feature1, feature2, feature3]


class ReverseDistillationNetwork(nn.Module):
    """Checkpointable teacher, one-class bottleneck, and reverse decoder."""

    def __init__(self, *, pretrained_backbone: bool) -> None:
        super().__init__()
        self.teacher = WideResNet50Encoder(pretrained=pretrained_backbone)
        self.bottleneck = OneClassBottleneck()
        self.decoder = ReverseWideResNet50Decoder()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.teacher.eval()
        with torch.no_grad():
            teacher_features = self.teacher(images)
        embedding = self.bottleneck(teacher_features)
        return teacher_features, self.decoder(embedding)


@register_model(
    "vision_reverse_dist",
    tags=("vision", "deep", "distillation", "pixel_map"),
    metadata={
        "description": "Alias for paper-aligned WideResNet50-2 Reverse Distillation",
        "paper": "Anomaly Detection via Reverse Distillation from One-Class Embedding",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html",
        "year": 2022,
        "supervision": "one-class",
        "implementation_status": "paper-network-and-defaults-aligned",
        "paper_fidelity": "core-aligned",
    },
)
@register_model(
    "vision_reverse_distillation",
    tags=("vision", "deep", "distillation", "pixel_map"),
    metadata={
        "description": "WideResNet50-2 reverse distillation through the authors' OCBE bottleneck",
        "paper": "Anomaly Detection via Reverse Distillation from One-Class Embedding",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html",
        "year": 2022,
        "supervision": "one-class",
        "implementation_status": "paper-network-and-defaults-aligned",
        "paper_fidelity": "core-aligned",
    },
)
class ReverseDistillation(BaseVisionDeepDetector):
    """Paper-aligned WideResNet50-2 Reverse Distillation detector."""

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = _PAPER_BACKBONE,
        pretrained_backbone: bool = True,
        selected_layers: Sequence[str] = _PAPER_LAYERS,
        anomaly_map_mode: str = "add",
        anomaly_smoothing_sigma: float = 4.0,
        image_size: int = 256,
        lr: float = 5e-3,
        epoch_num: int = 200,
        batch_size: int = 16,
        device: str | None = None,
        verbose: int = 1,
        random_state: int = 42,
        train_transform: object | None = None,
        eval_transform: object | None = None,
    ) -> None:
        if backbone == "wide_resnet50":
            backbone = _PAPER_BACKBONE
        if backbone != _PAPER_BACKBONE:
            raise ValueError("Reverse Distillation requires the paper backbone 'wide_resnet50_2'.")
        if tuple(selected_layers) != _PAPER_LAYERS:
            raise ValueError(
                "Reverse Distillation requires selected_layers=('layer1', 'layer2', 'layer3') "
                "for its one-class multi-scale bottleneck."
            )
        if anomaly_map_mode not in {"add", "multiply"}:
            raise ValueError("anomaly_map_mode must be 'add' or 'multiply'.")
        if float(anomaly_smoothing_sigma) < 0:
            raise ValueError("anomaly_smoothing_sigma must be non-negative.")
        if int(image_size) < 32 or int(image_size) % 32:
            raise ValueError("image_size must be at least 32 and divisible by 32.")

        if train_transform is None or eval_transform is None:
            from pyimgano.datasets import default_eval_transforms

            paper_transform = default_eval_transforms(
                resize=(int(image_size), int(image_size)), crop_size=int(image_size)
            )
            train_transform = paper_transform if train_transform is None else train_transform
            eval_transform = paper_transform if eval_transform is None else eval_transform

        self.backbone = backbone
        self.pretrained_backbone = bool(pretrained_backbone)
        self.selected_layers = tuple(selected_layers)
        self.anomaly_map_mode = anomaly_map_mode
        self.anomaly_smoothing_sigma = float(anomaly_smoothing_sigma)
        self.image_size = int(image_size)
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

    def build_model(self):
        network = ReverseDistillationNetwork(pretrained_backbone=self.pretrained_backbone).to(
            self.device
        )
        self.teacher = network.teacher
        self.bottleneck = network.bottleneck
        self.decoder = network.decoder
        self.optimizer = torch.optim.Adam(
            list(self.bottleneck.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=0.0,
        )
        return network

    def _forward_features(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.model(images)

    @staticmethod
    def _distillation_loss(
        teacher_features: Sequence[torch.Tensor],
        student_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for teacher, student in zip(teacher_features, student_features):
            similarity = F.cosine_similarity(teacher.flatten(1), student.flatten(1), dim=1)
            losses.append((1.0 - similarity).mean())
        return torch.stack(losses).sum()

    def _anomaly_maps(
        self,
        teacher_features: Sequence[torch.Tensor],
        student_features: Sequence[torch.Tensor],
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        maps = []
        for teacher, student in zip(teacher_features, student_features):
            distance = 1.0 - F.cosine_similarity(teacher, student, dim=1)
            maps.append(
                F.interpolate(
                    distance.unsqueeze(1),
                    size=output_size,
                    mode="bilinear",
                    align_corners=True,
                )
            )
        stacked = torch.stack(maps, dim=0)
        if self.anomaly_map_mode == "multiply":
            return torch.prod(stacked, dim=0)
        return torch.sum(stacked, dim=0)

    def _smooth_anomaly_maps(self, maps: torch.Tensor) -> NDArray:
        values = maps[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        if self.anomaly_smoothing_sigma == 0:
            return values
        from scipy.ndimage import gaussian_filter

        return np.stack(
            [
                gaussian_filter(item, sigma=self.anomaly_smoothing_sigma).astype(
                    np.float32, copy=False
                )
                for item in values
            ],
            axis=0,
        )

    def training_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        images, _ = batch
        images = images.to(self.device)
        teacher_features, student_features = self._forward_features(images)
        loss = self._distillation_loss(teacher_features, student_features)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluating_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> NDArray:
        images, _ = batch
        images = images.to(self.device)
        teacher_features, student_features = self._forward_features(images)
        maps = self._anomaly_maps(
            teacher_features,
            student_features,
            output_size=(int(images.shape[-2]), int(images.shape[-1])),
        )
        smoothed = self._smooth_anomaly_maps(maps)
        return smoothed.reshape(smoothed.shape[0], -1).max(axis=1)

    @torch.no_grad()
    def get_anomaly_map(self, image: str | Path | NDArray) -> NDArray:
        """Return a paper-style multi-scale cosine-distance map for one image."""

        self._check_is_fitted()
        from PIL import Image

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected a uint8 RGB image with shape (H, W, 3).")
            original_size = (int(image.shape[0]), int(image.shape[1]))
            pil_image = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        else:
            pil_image = Image.open(str(image)).convert("RGB")
            original_size = (int(pil_image.height), int(pil_image.width))

        tensor = self.eval_transform(pil_image).unsqueeze(0).to(self.device)
        teacher_features, student_features = self._forward_features(tensor)
        anomaly_map = self._anomaly_maps(
            teacher_features,
            student_features,
            output_size=original_size,
        )
        return self._smooth_anomaly_maps(anomaly_map)[0]

    def predict_anomaly_map(
        self,
        x: object = MISSING,
        **kwargs: object,
    ) -> NDArray:
        items = list(resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"))
        if not items:
            return np.zeros((0, 1, 1), dtype=np.float32)
        maps = [self.get_anomaly_map(item) for item in items]
        shapes = {item.shape for item in maps}
        if len(shapes) != 1:
            raise ValueError("All images must share a size to stack anomaly maps.")
        return np.stack(maps, axis=0)

    def fit(self, x: object = MISSING, y: Iterable[int] | None = None, **kwargs: object):
        return super().fit(resolve_legacy_x_keyword(x, kwargs, method_name="fit"), y)

    def save_checkpoint(self, path: str | Path) -> Path:
        self._check_is_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "pyimgano.reverse_distillation",
            "schema_version": 2,
            "config": {
                "backbone": self.backbone,
                "selected_layers": list(self.selected_layers),
                "anomaly_map_mode": self.anomaly_map_mode,
                "anomaly_smoothing_sigma": self.anomaly_smoothing_sigma,
                "image_size": self.image_size,
                "architecture": "wide-resnet50-2-ocbe-reverse-wide-resnet50-2",
            },
            "model_state_dict": {
                key: value.detach().cpu() for key, value in self.model.state_dict().items()
            },
            "decision_scores": getattr(self, "decision_scores_", None),
            "threshold": getattr(self, "threshold_", None),
            "labels": getattr(self, "labels_", None),
        }
        torch.save(payload, output)
        return output

    def load_checkpoint(self, path: str | Path) -> None:
        from .deep_io import safe_torch_load

        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format") != (
            "pyimgano.reverse_distillation"
        ):
            raise ValueError("Unsupported Reverse Distillation checkpoint format.")
        if int(payload.get("schema_version", -1)) != 2:
            raise ValueError(
                "Unsupported legacy Reverse Distillation checkpoint; refit with the "
                "paper-aligned WideResNet50-2/OCBE architecture."
            )

        config = dict(payload.get("config", {}))
        expected = {
            "backbone": self.backbone,
            "selected_layers": list(self.selected_layers),
            "anomaly_map_mode": self.anomaly_map_mode,
            "anomaly_smoothing_sigma": self.anomaly_smoothing_sigma,
            "image_size": self.image_size,
            "architecture": "wide-resnet50-2-ocbe-reverse-wide-resnet50-2",
        }
        if config != expected:
            raise ValueError(
                "Reverse Distillation checkpoint configuration does not match the detector. "
                f"Expected {expected!r}, got {config!r}."
            )

        with isolated_random_state(self.random_state):
            self.model = self.build_model()
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Reverse Distillation checkpoint is missing model_state_dict.")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        for attr, key in (
            ("decision_scores_", "decision_scores"),
            ("threshold_", "threshold"),
            ("labels_", "labels"),
        ):
            value = payload.get(key)
            if value is not None:
                setattr(self, attr, value)
        self._set_n_classes(None)


__all__ = [
    "OneClassBottleneck",
    "ReverseDistillation",
    "ReverseDistillationNetwork",
    "ReverseWideResNet50Decoder",
    "WideResNet50Encoder",
]
