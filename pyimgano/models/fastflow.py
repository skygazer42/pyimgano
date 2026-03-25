# -*- coding: utf-8 -*-
"""FastFlow-based visual anomaly detector implementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

# ---------------------------------------------------------------------------
# Flow building blocks
# ---------------------------------------------------------------------------


class ActNorm2d(nn.Module):
    """Activation normalization with data-dependent initialization."""

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.initialized = False
        self.eps = eps

    def _initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True) + self.eps
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(torch.log(1.0 / std))
        self.initialized = True

    def forward(
        self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._initialize(x)
        if logdet is None:
            logdet = x.new_zeros(x.size(0))

        h, w = x.shape[2], x.shape[3]
        if reverse:
            x = (x - self.bias) * torch.exp(-self.log_scale)
            logdet = logdet - torch.sum(self.log_scale) * h * w
        else:
            x = (x + self.bias) * torch.exp(self.log_scale)
            logdet = logdet + torch.sum(self.log_scale) * h * w
        return x, logdet


class InvConv2d(nn.Module):
    """Invertible 1x1 convolution following Glow."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        weight = torch.linalg.qr(torch.randn(num_features, num_features), mode="reduced").Q
        weight = weight.view(num_features, num_features, 1, 1)
        self.weight = nn.Parameter(weight)

    def _log_det(self) -> torch.Tensor:
        w = self.weight.squeeze(-1).squeeze(-1)
        _sign, log_abs_det = torch.linalg.slogdet(w)
        return log_abs_det

    def forward(
        self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logdet is None:
            logdet = x.new_zeros(x.size(0))
        h, w = x.shape[2], x.shape[3]
        log_abs_det = self._log_det() * h * w
        if reverse:
            weight = torch.inverse(self.weight.squeeze(-1).squeeze(-1))
            weight = weight.view_as(self.weight)
            x = F.conv2d(x, weight)
            logdet = logdet - log_abs_det
        else:
            x = F.conv2d(x, self.weight)
            logdet = logdet + log_abs_det
        return x, logdet


class AffineCoupling(nn.Module):
    """Affine coupling layer."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logdet is None:
            logdet = x.new_zeros(x.size(0))
        x1, x2 = torch.chunk(x, 2, dim=1)
        h = self.hidden(x1)
        shift, scale = torch.chunk(h, 2, dim=1)
        scale = torch.tanh(scale)
        if reverse:
            x2 = (x2 * torch.exp(-scale)) - shift
            logdet = logdet - scale.view(scale.size(0), -1).sum(dim=1)
        else:
            x2 = (x2 + shift) * torch.exp(scale)
            logdet = logdet + scale.view(scale.size(0), -1).sum(dim=1)
        return torch.cat([x1, x2], dim=1), logdet


class FlowStep(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 1.5) -> None:
        super().__init__()
        hidden_channels = int(math.ceil(channels * hidden_ratio))
        if channels % 2 != 0:
            raise ValueError("FlowStep channels must be even.")
        self.actnorm = ActNorm2d(channels)
        self.invconv = InvConv2d(channels)
        self.coupling = AffineCoupling(channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, logdet = self.actnorm(x, logdet, reverse=reverse)
        x, logdet = self.invconv(x, logdet, reverse=reverse)
        x, logdet = self.coupling(x, logdet, reverse=reverse)
        return x, logdet


class FlowStage(nn.Module):
    """A sequence of FlowSteps applied to a feature map."""

    def __init__(self, channels: int, n_steps: int, hidden_ratio: float) -> None:
        super().__init__()
        self.steps = nn.ModuleList(
            [FlowStep(channels, hidden_ratio=hidden_ratio) for _ in range(n_steps)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = x.new_zeros(x.size(0))
        for step in self.steps:
            x, logdet = step(x, logdet, reverse=False)
        return x, logdet

    @torch.no_grad()
    def forward_no_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        layers: Sequence[str] = ("layer2", "layer3", "layer4"),
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Currently only resnet18 backbone is supported.")
        weights = None
        if pretrained:
            try:  # torchvision>=0.13
                weights = models.ResNet18_Weights.DEFAULT
            except AttributeError:  # fallback older versions
                weights = (
                    models.ResNet18_Weights.IMAGENET1K_V1
                    if hasattr(models, "ResNet18_Weights")
                    else "DEFAULT"
                )
        net = models.resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.selected_layers = tuple(layers)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feature_map = {
            "layer2": feat2,
            "layer3": feat3,
            "layer4": feat4,
        }
        return [feature_map[name] for name in self.selected_layers]


# ---------------------------------------------------------------------------
# FastFlow detector
# ---------------------------------------------------------------------------


@register_model(
    "vision_fastflow",
    tags=("vision", "deep", "flow"),
    metadata={
        "description": "FastFlow-based visual anomaly detector",
        "paper": "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows",
        "year": 2021,
    },
)
class FastFlow(BaseVisionDeepDetector):
    """Implementation of FastFlow (ICCV'21) anomaly detector."""

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = False,
        selected_layers: Sequence[str] = ("layer2", "layer3", "layer4"),
        embedding_dim: int = 256,
        n_flow_steps: int = 8,
        flow_hidden_ratio: float = 1.5,
        lr: float = 1e-4,
        epoch_num: int = 20,
        batch_size: int = 16,
        device: str | None = None,
        verbose: int = 1,
        random_state: int = 42,
    ) -> None:
        self.backbone = backbone
        self.pretrained_backbone = pretrained_backbone
        self.selected_layers = tuple(selected_layers)
        self.embedding_dim = embedding_dim
        if self.embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for affine coupling.")
        self.n_flow_steps = n_flow_steps
        self.flow_hidden_ratio = flow_hidden_ratio
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

    def save_checkpoint(self, path: str | Path) -> Path:
        if getattr(self, "model", None) is None or not hasattr(self, "feature_extractor"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model_state_dict: dict[str, object] = {}
        for key, value in dict(self.model.state_dict()).items():
            detach = getattr(value, "detach", None)
            cpu = getattr(value, "cpu", None)
            if callable(detach) and callable(cpu):
                model_state_dict[str(key)] = detach().cpu()
            else:
                model_state_dict[str(key)] = value

        feature_extractor_state_dict: dict[str, object] = {}
        for key, value in dict(self.feature_extractor.state_dict()).items():
            detach = getattr(value, "detach", None)
            cpu = getattr(value, "cpu", None)
            if callable(detach) and callable(cpu):
                feature_extractor_state_dict[str(key)] = detach().cpu()
            else:
                feature_extractor_state_dict[str(key)] = value

        torch.save(
            {
                "model_state_dict": model_state_dict,
                "feature_extractor_state_dict": feature_extractor_state_dict,
                "decision_scores_": (
                    None
                    if getattr(self, "decision_scores_", None) is None
                    else np.asarray(self.decision_scores_, dtype=np.float64)
                ),
                "threshold_": (
                    None
                    if getattr(self, "threshold_", None) is None
                    else float(self.threshold_)
                ),
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError("Invalid FastFlow checkpoint payload.")

        if getattr(self, "model", None) is None or not hasattr(self, "feature_extractor"):
            self.model = self.build_model()

        model_state_dict = state.get("model_state_dict", None)
        feature_extractor_state_dict = state.get("feature_extractor_state_dict", None)
        if not isinstance(model_state_dict, dict) or not isinstance(feature_extractor_state_dict, dict):
            raise ValueError("FastFlow checkpoint is missing required state_dict payloads.")

        self.model.load_state_dict(dict(model_state_dict), strict=False)
        self.feature_extractor.load_state_dict(dict(feature_extractor_state_dict), strict=False)
        self.model.to(self.device)
        self.feature_extractor.to(self.device)
        self.model.eval()
        self.feature_extractor.eval()
        for module in self.model.modules():
            if isinstance(module, ActNorm2d):
                module.initialized = True

        decision_scores = state.get("decision_scores_", None)
        if decision_scores is not None:
            self.decision_scores_ = np.asarray(decision_scores, dtype=np.float64)
        threshold = state.get("threshold_", None)
        if threshold is not None:
            self.threshold_ = float(threshold)

    # ------------------------------------------------------------------
    def build_model(self):
        self.feature_extractor = ResNetFeatureExtractor(
            backbone=self.backbone,
            pretrained=self.pretrained_backbone,
            layers=self.selected_layers,
        ).to(self.device)
        self.feature_extractor.eval()

        adaptor_list = []
        stage_list = []
        channel_map = {"layer2": 128, "layer3": 256, "layer4": 512}
        for layer in self.selected_layers:
            in_channels = channel_map.get(layer)
            if in_channels is None:
                raise ValueError(f"Unsupported layer {layer}")
            adaptor = nn.Conv2d(in_channels, self.embedding_dim, kernel_size=1, stride=1, bias=True)
            adaptor_list.append(adaptor)
            stage_list.append(
                FlowStage(self.embedding_dim, self.n_flow_steps, self.flow_hidden_ratio)
            )

        self.adapters = nn.ModuleList(adaptor_list).to(self.device)
        self.flow_stages = nn.ModuleList(stage_list).to(self.device)

        params = list(self.adapters.parameters()) + list(self.flow_stages.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=0.0)

        return nn.ModuleList([self.adapters, self.flow_stages])

    # ------------------------------------------------------------------
    def _extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            feats = self.feature_extractor(images)
        return [feat.detach() for feat in feats]

    # ------------------------------------------------------------------
    def _flow_nll(self, z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        flat = z.view(z.size(0), -1)
        n_dims = flat.size(1)
        log_prob = (-0.5 * flat.pow(2).sum(dim=1) + logdet) / n_dims
        return -log_prob  # negative log likelihood per sample

    # ------------------------------------------------------------------
    def training_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        images, _ = batch
        images = images.to(self.device)

        features = self._extract_features(images)

        self.adapters.train()
        self.flow_stages.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss = 0.0
        for feat, adaptor, flow in zip(features, self.adapters, self.flow_stages):
            feat = adaptor(feat.to(self.device))
            z, logdet = flow(feat)
            loss_stage = self._flow_nll(z, logdet).mean()
            loss = loss + loss_stage

        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluating_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        images, _ = batch
        images = images.to(self.device)
        features = self._extract_features(images)

        self.adapters.eval()
        self.flow_stages.eval()
        scores = []
        for feat, adaptor, flow in zip(features, self.adapters, self.flow_stages):
            feat = adaptor(feat.to(self.device))
            z, logdet = flow.forward_no_grad(feat)
            stage_score = self._flow_nll(z, logdet)
            scores.append(stage_score)
        total = torch.stack(scores, dim=1).mean(dim=1)
        return total.cpu().numpy()

    # ------------------------------------------------------------------
    def fit(self, x: object = MISSING, y: Iterable[int] | None = None, **kwargs: object):
        # Override to ensure feature extractor is on correct device before DataLoader loop
        return super().fit(resolve_legacy_x_keyword(x, kwargs, method_name="fit"), y)

    def build_model_loader(self, x: Sequence[str]) -> DataLoader:
        # Not overriding base behaviour; placeholder for compatibility.
        return super().build_model_loader(x)
