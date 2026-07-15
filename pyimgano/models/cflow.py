"""CFLOW-AD conditional normalizing-flow anomaly detector.

This module implements the ResNet path released by the CFLOW-AD authors:
multi-scale frozen features, 2-D positional conditions, independent conditional
flow decoders, exact feature likelihoods, and multi-scale anomaly maps.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from pyimgano.datasets import ImagePathDataset
from pyimgano.utils.random_state import isolated_random_state_method
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."
_BACKBONE_CHANNELS = {
    "resnet18": (128, 256, 512),
    "wide_resnet50_2": (512, 1024, 2048),
}
_GAUSSIAN_LOG_CONSTANT = -0.5 * math.log(2.0 * math.pi)

logger = logging.getLogger(__name__)


def positional_encoding_2d(
    dimensions: int,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the author's sinusoidal ``[C, H, W]`` spatial condition."""

    if dimensions <= 0 or dimensions % 4:
        raise ValueError(
            "CFLOW positional condition dimensions must be positive and divisible by 4."
        )
    half = dimensions // 2
    frequencies = torch.exp(
        torch.arange(0, half, 2, device=device, dtype=dtype) * -(math.log(10_000.0) / half)
    )
    horizontal = torch.arange(width, device=device, dtype=dtype).unsqueeze(1)
    vertical = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)
    encoding = torch.zeros(dimensions, height, width, device=device, dtype=dtype)
    encoding[0:half:2] = torch.sin(horizontal * frequencies).T.unsqueeze(1).expand(-1, height, -1)
    encoding[1:half:2] = torch.cos(horizontal * frequencies).T.unsqueeze(1).expand(-1, height, -1)
    encoding[half::2] = torch.sin(vertical * frequencies).T.unsqueeze(2).expand(-1, -1, width)
    encoding[half + 1 :: 2] = torch.cos(vertical * frequencies).T.unsqueeze(2).expand(-1, -1, width)
    return encoding


def _soft_permutation(dimensions: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(dimensions, dimensions, generator=generator)
    orthogonal, triangular = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(triangular)).clamp_min(0).mul(2).sub(1)
    return orthogonal * signs.unsqueeze(0)


class ConditionalAllInOneBlock(nn.Module):
    """Dependency-free equivalent of the authors' FrEIA AllInOneBlock setup."""

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        *,
        clamp_alpha: float = 1.9,
        soft_permutation: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if feature_dim < 2:
            raise ValueError("feature_dim must be at least 2.")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")
        if clamp_alpha <= 0:
            raise ValueError("clamp_alpha must be positive.")

        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.split_dim_1 = self.feature_dim - self.feature_dim // 2
        self.split_dim_2 = self.feature_dim // 2
        self.clamp_alpha = float(clamp_alpha)
        self.soft_permutation = bool(soft_permutation)

        subnet_input = self.split_dim_1 + self.condition_dim
        subnet_hidden = 2 * subnet_input
        self.subnet = nn.Sequential(
            nn.Linear(subnet_input, subnet_hidden),
            nn.ReLU(),
            nn.Linear(subnet_hidden, 2 * self.split_dim_2),
        )

        raw_global_scale = 2.0 * math.log(math.expm1(5.0))
        self.global_scale = nn.Parameter(torch.full((self.feature_dim,), raw_global_scale))
        self.global_offset = nn.Parameter(torch.zeros(self.feature_dim))

        if self.soft_permutation:
            self.register_buffer("permutation_matrix", _soft_permutation(self.feature_dim, seed))
            self.register_buffer("permutation", torch.empty(0, dtype=torch.long))
            self.register_buffer("inverse_permutation", torch.empty(0, dtype=torch.long))
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            permutation = torch.randperm(self.feature_dim, generator=generator)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(self.feature_dim)
            self.register_buffer("permutation_matrix", torch.empty(0))
            self.register_buffer("permutation", permutation)
            self.register_buffer("inverse_permutation", inverse)

    def _global_scale(self) -> torch.Tensor:
        return 0.1 * F.softplus(self.global_scale, beta=0.5)

    def _permute(self, value: torch.Tensor, *, reverse: bool) -> torch.Tensor:
        if self.soft_permutation:
            weight = self.permutation_matrix.T if reverse else self.permutation_matrix
            return F.linear(value, weight)
        indices = self.inverse_permutation if reverse else self.permutation
        return value.index_select(1, indices)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim != 2 or condition.ndim != 2 or value.shape[0] != condition.shape[0]:
            raise ValueError("CFLOW values and conditions must be aligned 2-D tensors.")
        if value.shape[1] != self.feature_dim or condition.shape[1] != self.condition_dim:
            raise ValueError("CFLOW value or condition dimensions do not match this block.")

        global_scale = self._global_scale()
        global_logdet = torch.log(global_scale).sum().expand(value.shape[0])
        if reverse:
            value = self._permute(value, reverse=True)
            value = (value - self.global_offset) / global_scale
            global_logdet = -global_logdet

        identity, transformed = torch.split(
            value,
            (self.split_dim_1, self.split_dim_2),
            dim=1,
        )
        coefficients = 0.1 * self.subnet(torch.cat((identity, condition), dim=1))
        raw_scale, shift = coefficients.split(self.split_dim_2, dim=1)
        log_scale = self.clamp_alpha * torch.tanh(raw_scale)
        if reverse:
            transformed = (transformed - shift) * torch.exp(-log_scale)
            coupling_logdet = -log_scale.sum(dim=1)
        else:
            transformed = transformed * torch.exp(log_scale) + shift
            coupling_logdet = log_scale.sum(dim=1)

        value = torch.cat((identity, transformed), dim=1)
        if not reverse:
            value = self._permute(value * global_scale + self.global_offset, reverse=False)
        return value, coupling_logdet + global_logdet


class ConditionalFlow(nn.Module):
    """Sequence of the conditional flow blocks used by CFLOW-AD."""

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        n_flows: int = 8,
        *,
        clamp_alpha: float = 1.9,
        soft_permutation: bool = True,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        if n_flows <= 0:
            raise ValueError("n_flows must be positive.")
        self.flows = nn.ModuleList(
            [
                ConditionalAllInOneBlock(
                    feature_dim,
                    condition_dim,
                    clamp_alpha=clamp_alpha,
                    soft_permutation=soft_permutation,
                    seed=int(random_state) + index,
                )
                for index in range(int(n_flows))
            ]
        )

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = value.new_zeros(value.shape[0])
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            value, flow_logdet = flow(value, condition, reverse=reverse)
            logdet = logdet + flow_logdet
        return value, logdet


class CFlowEncoder(nn.Module):
    """Frozen ResNet encoder returning the author's layer2--layer4 pyramid."""

    def __init__(
        self,
        backbone: str,
        *,
        pretrained: bool,
        pool_layers: int,
    ) -> None:
        super().__init__()
        if backbone == "wide_resnet50":
            backbone = "wide_resnet50_2"
        if backbone not in _BACKBONE_CHANNELS:
            raise ValueError("CFLOW supports 'resnet18' and 'wide_resnet50_2'.")
        if pool_layers not in (1, 2, 3):
            raise ValueError("pool_layers must be 1, 2, or 3.")

        network, _ = load_torchvision_model(backbone, pretrained=bool(pretrained))
        self.backbone = backbone
        self.pool_layers = int(pool_layers)
        self.stem = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def output_channels(self) -> tuple[int, ...]:
        return _BACKBONE_CHANNELS[self.backbone][-self.pool_layers :]

    def train(self, mode: bool = True):
        del mode
        return super().train(False)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = self.stem(images)
        features = self.layer1(features)
        feature2 = self.layer2(features)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        return [feature2, feature3, feature4][-self.pool_layers :]


def _cflow_transform(image_size: int, *, training: bool):
    operations: list[object] = [
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
    ]
    if training:
        operations.append(transforms.RandomRotation(5))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


@register_model(
    "vision_cflow",
    tags=("vision", "deep", "cflow", "normalizing-flow", "real-time", "pixel_map"),
    metadata={
        "description": "CFLOW-AD multi-scale conditional likelihood decoder",
        "paper": "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows",
        "paper_url": "https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html",
        "author_code": "https://github.com/gudovskiy/cflow-ad",
        "year": 2022,
        "supervision": "one-class",
        "implementation_status": "paper-resnet-cflow-core-aligned",
        "paper_fidelity": "core-aligned",
        "speed": "real-time",
    },
)
class VisionCFlow(BaseVisionDeepDetector):
    """Paper-aligned CFLOW-AD ResNet detector.

    The default network and training hyperparameters follow the author's
    released configuration. ImageNet downloads remain opt-in for offline use;
    set ``pretrained_backbone=True`` for the paper protocol. The repository's
    contamination threshold is an API adaptation of the paper's metric-time
    threshold selection.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        pretrained_backbone: bool = False,
        pool_layers: int = 3,
        n_flows: int = 8,
        condition_dim: int = 128,
        clamp_alpha: float = 1.9,
        soft_permutation: bool = True,
        image_size: int = 256,
        epochs: int = 25,
        sub_epochs: int = 8,
        batch_size: int = 32,
        fiber_batch_size: int = 256,
        lr: float = 2e-4,
        num_workers: int = 0,
        contamination: float = 0.1,
        device: str | None = None,
        verbose: int = 1,
        random_state: int = 42,
        train_transform: object | None = None,
        eval_transform: object | None = None,
        **kwargs: object,
    ) -> None:
        if backbone == "wide_resnet50":
            backbone = "wide_resnet50_2"
        if backbone not in _BACKBONE_CHANNELS:
            raise ValueError("CFLOW supports 'resnet18' and 'wide_resnet50_2'.")
        if pool_layers not in (1, 2, 3):
            raise ValueError("pool_layers must be 1, 2, or 3.")
        if n_flows <= 0 or condition_dim <= 0 or condition_dim % 4:
            raise ValueError("n_flows must be positive and condition_dim divisible by 4.")
        if clamp_alpha <= 0 or image_size <= 0:
            raise ValueError("clamp_alpha and image_size must be positive.")
        if epochs < 0 or sub_epochs <= 0 or batch_size <= 0 or fiber_batch_size <= 0:
            raise ValueError("Training epoch and batch parameters must be positive.")
        if lr <= 0 or num_workers < 0:
            raise ValueError("lr must be positive and num_workers non-negative.")

        train_transform = train_transform or _cflow_transform(image_size, training=True)
        eval_transform = eval_transform or _cflow_transform(image_size, training=False)
        super().__init__(
            contamination=contamination,
            preprocessing=True,
            lr=lr,
            epoch_num=epochs,
            batch_size=batch_size,
            optimizer_name="adam",
            device=device,
            random_state=random_state,
            verbose=verbose,
            train_transform=train_transform,
            eval_transform=eval_transform,
            **kwargs,
        )

        self.backbone_name = backbone
        self.pretrained_backbone = bool(pretrained_backbone)
        self.pool_layers = int(pool_layers)
        self.n_flows = int(n_flows)
        self.condition_dim = int(condition_dim)
        self.clamp_alpha = float(clamp_alpha)
        self.soft_permutation = bool(soft_permutation)
        self.image_size = int(image_size)
        self.epochs = int(epochs)
        self.sub_epochs = int(sub_epochs)
        self.fiber_batch_size = int(fiber_batch_size)
        self.num_workers = int(num_workers)
        self._is_fitted = False
        self.feature_extractor: CFlowEncoder | None = None
        self.decoders: nn.ModuleList | None = None

    def _build_model(self) -> None:
        self.feature_extractor = CFlowEncoder(
            self.backbone_name,
            pretrained=self.pretrained_backbone,
            pool_layers=self.pool_layers,
        ).to(self.device)
        self.feature_extractor.eval()
        self.decoders = nn.ModuleList(
            [
                ConditionalFlow(
                    channels,
                    self.condition_dim,
                    self.n_flows,
                    clamp_alpha=self.clamp_alpha,
                    soft_permutation=self.soft_permutation,
                    random_state=int(self.random_state or 0) + scale * self.n_flows,
                )
                for scale, channels in enumerate(self.feature_extractor.output_channels)
            ]
        ).to(self.device)

    def _require_model(self) -> tuple[CFlowEncoder, nn.ModuleList]:
        if self.feature_extractor is None or self.decoders is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        return self.feature_extractor, self.decoders

    def _extract_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        encoder, _ = self._require_model()
        encoder.eval()
        with torch.no_grad():
            return [feature.detach() for feature in encoder(images)]

    def _fibers(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = feature.shape
        values = feature.permute(0, 2, 3, 1).reshape(-1, channels)
        condition = positional_encoding_2d(
            self.condition_dim,
            height,
            width,
            device=feature.device,
            dtype=feature.dtype,
        )
        condition = (
            condition.unsqueeze(0)
            .expand(batch, -1, -1, -1)
            .permute(0, 2, 3, 1)
            .reshape(-1, self.condition_dim)
        )
        return values, condition

    @staticmethod
    def _log_probability(
        feature_dim: int,
        latent: torch.Tensor,
        logdet: torch.Tensor,
    ) -> torch.Tensor:
        return feature_dim * _GAUSSIAN_LOG_CONSTANT - 0.5 * latent.square().sum(dim=1) + logdet

    def _scheduled_lr(self, epoch: int, progress: float) -> float:
        if self.epochs <= 0:
            return self.lr
        eta_min = self.lr * 1e-3
        cosine_lr = (
            eta_min + (self.lr - eta_min) * (1.0 + math.cos(math.pi * epoch / self.epochs)) / 2.0
        )
        if epoch >= 2:
            return cosine_lr
        warmup_to = (
            eta_min
            + (self.lr - eta_min)
            * (1.0 + math.cos(math.pi * min(2, self.epochs) / self.epochs))
            / 2.0
        )
        fraction = min(1.0, (epoch + progress) / 2.0)
        return self.lr / 10.0 + fraction * (warmup_to - self.lr / 10.0)

    @isolated_random_state_method
    def fit(
        self,
        x: object = MISSING,
        y: NDArray | None = None,
        **kwargs: object,
    ) -> "VisionCFlow":
        del y
        x_list = list(cast(Iterable[str], resolve_legacy_x_keyword(x, kwargs, method_name="fit")))
        if not x_list:
            raise ValueError("Training set cannot be empty.")
        self._build_model()
        _, decoders = self._require_model()

        dataset = ImagePathDataset(
            x_list,
            transform=self.train_transform,
            return_full_path=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.optimizer = Adam(decoders.parameters(), lr=self.lr, weight_decay=0.0)
        steps_per_meta_epoch = max(1, len(loader) * self.sub_epochs)

        for epoch in range(self.epochs):
            decoders.train()
            epoch_loss = 0.0
            updates = 0
            for sub_epoch in range(self.sub_epochs):
                for batch_index, (images, _) in enumerate(loader):
                    progress = (sub_epoch * len(loader) + batch_index) / steps_per_meta_epoch
                    learning_rate = self._scheduled_lr(epoch, progress)
                    for group in self.optimizer.param_groups:
                        group["lr"] = learning_rate

                    features = self._extract_features(images.to(self.device))
                    for feature, decoder in zip(features, decoders):
                        values, condition = self._fibers(feature)
                        order = torch.randperm(values.shape[0], device=values.device)
                        for start in range(0, values.shape[0], self.fiber_batch_size):
                            indices = order[start : start + self.fiber_batch_size]
                            latent, logdet = decoder(values[indices], condition[indices])
                            log_probability = (
                                self._log_probability(feature.shape[1], latent, logdet)
                                / feature.shape[1]
                            )
                            loss = F.softplus(-log_probability).mean()
                            self.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            self.optimizer.step()
                            epoch_loss += float(loss.detach().item())
                            updates += 1
            if self.verbose:
                logger.info(
                    "CFLOW epoch %d/%d loss %.6f",
                    epoch + 1,
                    self.epochs,
                    epoch_loss / max(1, updates),
                )

        self._is_fitted = True
        self.is_fitted_ = True
        self.decision_scores_ = self.decision_function(x_list)
        self._process_decision_scores()
        return self

    @torch.no_grad()
    def _feature_log_probability_maps(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = self._extract_features(images)
        _, decoders = self._require_model()
        decoders.eval()
        maps: list[torch.Tensor] = []
        for feature, decoder in zip(features, decoders):
            values, condition = self._fibers(feature)
            log_probabilities = []
            for start in range(0, values.shape[0], self.fiber_batch_size):
                latent, logdet = decoder(
                    values[start : start + self.fiber_batch_size],
                    condition[start : start + self.fiber_batch_size],
                )
                log_probabilities.append(
                    self._log_probability(feature.shape[1], latent, logdet) / feature.shape[1]
                )
            batch, _, height, width = feature.shape
            maps.append(torch.cat(log_probabilities).reshape(batch, height, width))
        return maps

    @torch.no_grad()
    def _anomaly_maps_for_paths(
        self,
        paths: list[str],
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if not paths:
            return torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
        dataset = ImagePathDataset(
            paths,
            transform=self.eval_transform,
            return_full_path=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        collected = [[] for _ in range(self.pool_layers)]
        try:
            for images, _ in loader:
                for scale, log_probability in enumerate(
                    self._feature_log_probability_maps(images.to(self.device))
                ):
                    collected[scale].append(log_probability.cpu())
        except FileNotFoundError as exc:
            raise ValueError(f"Failed to load image: {exc}") from exc

        probability_sum = torch.zeros(
            (len(paths), self.image_size, self.image_size),
            dtype=torch.float32,
        )
        for scale_maps in collected:
            log_probability = torch.cat(scale_maps)
            probability = torch.exp(log_probability - log_probability.amax())
            probability_sum += F.interpolate(
                probability.unsqueeze(1),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)
        return probability_sum.amax() - probability_sum

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        self._check_fitted()
        scores = self.decision_function(
            cast(Iterable[str], resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        )
        return (scores >= self.threshold_).astype(int)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: int | None = None,
        **kwargs: object,
    ) -> NDArray:
        current_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if current_batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._check_fitted()
        paths = list(
            cast(
                Iterable[str],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        maps = self._anomaly_maps_for_paths(paths, batch_size=current_batch_size)
        return maps.flatten(start_dim=1).amax(dim=1).numpy().astype(np.float64)

    @staticmethod
    def _resize_anomaly_map(anomaly_map: torch.Tensor, path: str) -> NDArray:
        from PIL import Image

        with Image.open(path) as image:
            output_size = (int(image.height), int(image.width))
        resized = F.interpolate(
            anomaly_map.reshape(1, 1, *anomaly_map.shape),
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        return resized[0, 0].numpy().astype(np.float32, copy=False)

    def get_anomaly_map(self, image_path: str | Path) -> NDArray:
        self._check_fitted()
        path = str(image_path)
        anomaly_map = self._anomaly_maps_for_paths([path], batch_size=1)[0]
        return self._resize_anomaly_map(anomaly_map, path)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        self._check_fitted()
        paths = list(
            cast(
                Iterable[str],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
            )
        )
        if not paths:
            return np.zeros((0, self.image_size, self.image_size), dtype=np.float32)
        maps = self._anomaly_maps_for_paths(paths, batch_size=self.batch_size)
        resized = [
            self._resize_anomaly_map(anomaly_map, path) for anomaly_map, path in zip(maps, paths)
        ]
        if len({anomaly_map.shape for anomaly_map in resized}) != 1:
            raise ValueError("All images must share a size to stack anomaly maps.")
        return np.stack(resized)
