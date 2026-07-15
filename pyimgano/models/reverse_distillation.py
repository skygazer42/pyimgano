# -*- coding: utf-8 -*-
"""Reverse Distillation from a one-class bottleneck embedding.

This module implements the defining data flow from Deng and Li (CVPR 2022):
a frozen ResNet encoder produces a multi-scale feature pyramid, a trainable
one-class bottleneck fuses and compresses it, and a reverse ResNet decoder
restores the encoder features from deep to shallow levels.

It is a ResNet-18 architecture adaptation, not the authors' WideResNet50-2,
OCBE bottleneck, and reverse-WRN implementation.
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

_PAPER_LAYERS = ("layer1", "layer2", "layer3")


class ResNet18Encoder(nn.Module):
    """Frozen ResNet-18 encoder exposing the paper's first three blocks."""

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        net, _ = load_torchvision_model("resnet18", pretrained=pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        feature1 = self.layer1(self.stem(images))
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        return [feature1, feature2, feature3]


class ResidualDownBlock(nn.Module):
    """Residual block that compresses the fused pyramid by one scale."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(features) + self.skip(features), inplace=True)


class OneClassBottleneck(nn.Module):
    """Multi-scale feature fusion followed by one-class compression."""

    def __init__(self) -> None:
        super().__init__()
        self.downsample_low = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.downsample_mid = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.embedding = ResidualDownBlock(256 * 3, 512)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError(f"Expected three encoder features, got {len(features)}.")
        low, mid, high = features
        low = self.downsample_low(low)
        mid = self.downsample_mid(mid)
        if low.shape[-2:] != high.shape[-2:] or mid.shape[-2:] != high.shape[-2:]:
            raise RuntimeError("Reverse Distillation feature scales are inconsistent.")
        return self.embedding(torch.cat([low, mid, high], dim=1))


class ReverseResidualBlock(nn.Module):
    """Residual decoder block with optional 2x spatial upsampling."""

    def __init__(self, in_channels: int, out_channels: int, *, upsample: bool) -> None:
        super().__init__()
        if upsample:
            first: nn.Module = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=False
            )
            skip: nn.Module = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            first = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            skip = nn.Identity()

        self.main = nn.Sequential(
            first,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = skip

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(features) + self.skip(features), inplace=True)


class ReverseResNet18Decoder(nn.Module):
    """ResNet-18 decoder restoring layer3, layer2, then layer1 features."""

    def __init__(self) -> None:
        super().__init__()
        self.restore3 = nn.Sequential(
            ReverseResidualBlock(512, 256, upsample=True),
            ReverseResidualBlock(256, 256, upsample=False),
        )
        self.restore2 = nn.Sequential(
            ReverseResidualBlock(256, 128, upsample=True),
            ReverseResidualBlock(128, 128, upsample=False),
        )
        self.restore1 = nn.Sequential(
            ReverseResidualBlock(128, 64, upsample=True),
            ReverseResidualBlock(64, 64, upsample=False),
        )

    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        feature3 = self.restore3(embedding)
        feature2 = self.restore2(feature3)
        feature1 = self.restore1(feature2)
        return [feature1, feature2, feature3]


class ReverseDistillationNetwork(nn.Module):
    """Checkpointable teacher, one-class bottleneck, and reverse decoder."""

    def __init__(self, *, pretrained_backbone: bool) -> None:
        super().__init__()
        self.teacher = ResNet18Encoder(pretrained=pretrained_backbone)
        self.bottleneck = OneClassBottleneck()
        self.decoder = ReverseResNet18Decoder()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.teacher.eval()
        with torch.no_grad():
            teacher_features = self.teacher(images)
        embedding = self.bottleneck(teacher_features)
        return teacher_features, self.decoder(embedding)


@register_model(
    "vision_reverse_dist",
    tags=("vision", "deep", "distillation", "pixel_map"),
    metadata={
        "description": "Alias for the ResNet-18 Reverse Distillation adaptation",
        "paper": "Anomaly Detection via Reverse Distillation from One-Class Embedding",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html",
        "year": 2022,
        "supervision": "one-class",
        "implementation_status": "resnet18-architecture-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
@register_model(
    "vision_reverse_distillation",
    tags=("vision", "deep", "distillation", "pixel_map"),
    metadata={
        "description": "ResNet-18 adaptation of reverse distillation through a one-class bottleneck",
        "paper": "Anomaly Detection via Reverse Distillation from One-Class Embedding",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html",
        "year": 2022,
        "supervision": "one-class",
        "implementation_status": "resnet18-architecture-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
class ReverseDistillation(BaseVisionDeepDetector):
    """Reverse Distillation adaptation using a ResNet-18 encoder/decoder."""

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = False,
        selected_layers: Sequence[str] = _PAPER_LAYERS,
        anomaly_map_mode: str = "add",
        lr: float = 5e-3,
        epoch_num: int = 20,
        batch_size: int = 16,
        device: str | None = None,
        verbose: int = 1,
        random_state: int = 42,
    ) -> None:
        if backbone != "resnet18":
            raise ValueError("The native Reverse Distillation implementation supports resnet18.")
        if tuple(selected_layers) != _PAPER_LAYERS:
            raise ValueError(
                "Reverse Distillation requires selected_layers=('layer1', 'layer2', 'layer3') "
                "for its one-class multi-scale bottleneck."
            )
        if anomaly_map_mode not in {"add", "multiply"}:
            raise ValueError("anomaly_map_mode must be 'add' or 'multiply'.")

        self.backbone = backbone
        self.pretrained_backbone = bool(pretrained_backbone)
        self.selected_layers = tuple(selected_layers)
        self.anomaly_map_mode = anomaly_map_mode
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
        )

    def build_model(self):
        network = ReverseDistillationNetwork(
            pretrained_backbone=self.pretrained_backbone
        ).to(self.device)
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
        return maps.flatten(1).amax(dim=1).cpu().numpy()

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
        return anomaly_map[0, 0].cpu().numpy().astype(np.float32, copy=False)

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
            "schema_version": 1,
            "config": {
                "backbone": self.backbone,
                "selected_layers": list(self.selected_layers),
                "anomaly_map_mode": self.anomaly_map_mode,
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
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported Reverse Distillation checkpoint schema version.")

        config = dict(payload.get("config", {}))
        expected = {
            "backbone": self.backbone,
            "selected_layers": list(self.selected_layers),
            "anomaly_map_mode": self.anomaly_map_mode,
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
    "ResNet18Encoder",
    "ReverseDistillation",
    "ReverseDistillationNetwork",
    "ReverseResNet18Decoder",
]
