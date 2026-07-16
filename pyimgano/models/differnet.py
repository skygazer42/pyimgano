from __future__ import annotations

"""DifferNet normalizing-flow anomaly detector.

Implements the core architecture from "Same Same But DifferNet" (WACV 2021):
multi-scale AlexNet features, an invertible density model, and likelihood
aggregation over fixed image transformations.
"""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torchvision import transforms

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .deep_io import export_module_state_dict, safe_torch_load
from .registry import register_model


class DifferNetSubnet(nn.Sequential):
    """Paper s/t subnet: three 2048-unit hidden fully connected layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        layers: list[nn.Module] = []
        width = int(input_dim)
        for _ in range(3):
            layers.extend(
                [
                    nn.Linear(width, int(hidden_dim)),
                    nn.Dropout(float(dropout)),
                    nn.ReLU(),
                ]
            )
            width = int(hidden_dim)
        layers.append(nn.Linear(width, int(output_dim)))
        super().__init__(*layers)


class DifferNetCouplingBlock(nn.Module):
    """Fixed permutation followed by the paper's two-sided affine coupling."""

    def __init__(
        self,
        feature_dim: int,
        *,
        seed: int,
        hidden_dim: int = 2048,
        clamp: float = 3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim < 2:
            raise ValueError("feature_dim must be at least 2.")
        if clamp <= 0:
            raise ValueError("clamp must be positive.")

        self.split_dim_1 = int(feature_dim) // 2
        self.split_dim_2 = int(feature_dim) - self.split_dim_1
        self.clamp = float(clamp)
        self.s1 = DifferNetSubnet(
            self.split_dim_1,
            self.split_dim_2 * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.s2 = DifferNetSubnet(
            self.split_dim_2,
            self.split_dim_1 * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        permutation = np.random.RandomState(int(seed)).permutation(int(feature_dim))
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(int(feature_dim))
        self.register_buffer("permutation", torch.as_tensor(permutation, dtype=torch.long))
        self.register_buffer("inverse_permutation", torch.as_tensor(inverse, dtype=torch.long))

    def _log_scale(self, value: torch.Tensor) -> torch.Tensor:
        # Equation 2 and the exact coefficient used by the authors' FrEIA snapshot.
        return self.clamp * 0.636 * torch.atan(value / self.clamp)

    def forward(
        self, value: torch.Tensor, *, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            value = value.index_select(1, self.permutation)

        value_1, value_2 = torch.split(value, (self.split_dim_1, self.split_dim_2), dim=1)
        if reverse:
            scale_1, shift_1 = self.s1(value_1).split(self.split_dim_2, dim=1)
            log_scale_1 = self._log_scale(scale_1)
            output_2 = (value_2 - shift_1) * torch.exp(-log_scale_1)

            scale_2, shift_2 = self.s2(output_2).split(self.split_dim_1, dim=1)
            log_scale_2 = self._log_scale(scale_2)
            output_1 = (value_1 - shift_2) * torch.exp(-log_scale_2)
            output = torch.cat((output_1, output_2), dim=1)
            output = output.index_select(1, self.inverse_permutation)
            logdet = -(log_scale_1.sum(dim=1) + log_scale_2.sum(dim=1))
            return output, logdet

        scale_2, shift_2 = self.s2(value_2).split(self.split_dim_1, dim=1)
        log_scale_2 = self._log_scale(scale_2)
        output_1 = torch.exp(log_scale_2) * value_1 + shift_2

        scale_1, shift_1 = self.s1(output_1).split(self.split_dim_2, dim=1)
        log_scale_1 = self._log_scale(scale_1)
        output_2 = torch.exp(log_scale_1) * value_2 + shift_1
        output = torch.cat((output_1, output_2), dim=1).clamp(-1e6, 1e6)
        logdet = log_scale_1.sum(dim=1) + log_scale_2.sum(dim=1)
        return output, logdet


class DifferNetFlow(nn.Module):
    """Chain of paper-aligned DifferNet coupling blocks."""

    def __init__(
        self,
        feature_dim: int,
        *,
        n_blocks: int = 8,
        hidden_dim: int = 2048,
        clamp: float = 3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive.")
        self.blocks = nn.ModuleList(
            [
                DifferNetCouplingBlock(
                    feature_dim,
                    seed=index,
                    hidden_dim=hidden_dim,
                    clamp=clamp,
                    dropout=dropout,
                )
                for index in range(int(n_blocks))
            ]
        )

    def forward(
        self, value: torch.Tensor, *, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = value.new_zeros(value.shape[0])
        blocks = reversed(self.blocks) if reverse else self.blocks
        for block in blocks:
            value, block_logdet = block(value, reverse=reverse)
            logdet = logdet + block_logdet
        return value, logdet


class DifferNetNetwork(nn.Module):
    """Frozen multi-scale AlexNet encoder followed by a vector normalizing flow."""

    def __init__(
        self,
        *,
        pretrained: bool,
        image_size: int,
        n_scales: int,
        n_flow_steps: int,
        flow_hidden_dim: int = 2048,
        flow_clamp: float = 3.0,
        flow_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_scales <= 0:
            raise ValueError("n_scales must be positive.")

        alexnet, _ = load_torchvision_model("alexnet", pretrained=bool(pretrained))
        self.feature_extractor = alexnet.features
        self.image_size = int(image_size)
        self.n_scales = int(n_scales)
        self.feature_dim = 256 * self.n_scales
        self.flow = DifferNetFlow(
            self.feature_dim,
            n_blocks=n_flow_steps,
            hidden_dim=flow_hidden_dim,
            clamp=flow_clamp,
            dropout=flow_dropout,
        )

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        self.feature_extractor.eval()

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode="nearest")
        features = []
        for scale in range(self.n_scales):
            scaled = images
            if scale:
                scaled = F.interpolate(
                    images,
                    size=(self.image_size // (2**scale), self.image_size // (2**scale)),
                    mode="nearest",
                )
            encoded = self.feature_extractor(scaled)
            features.append(encoded.mean(dim=(-2, -1)))
        return torch.cat(features, dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.extract_features(images)
        return self.flow(features)


@register_model(
    "vision_differnet",
    tags=("vision", "deep", "flow"),
    metadata={
        "description": "DifferNet paper detection path with AlexNet multi-scale features and exact affine flow",
        "paper": "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows",
        "paper_url": "https://arxiv.org/abs/2008.12577",
        "year": 2021,
        "supervision": "one-class",
        "implementation_status": "paper-detection-path-aligned-no-localization",
        "paper_fidelity": "paper-adaptation",
    },
)
@register_model(
    "differnet",
    tags=("vision", "deep", "flow"),
    metadata={
        "description": "Legacy alias for the paper-aligned DifferNet detection path",
        "paper": "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows",
        "paper_url": "https://arxiv.org/abs/2008.12577",
        "year": 2021,
        "supervision": "one-class",
        "implementation_status": "paper-detection-path-aligned-no-localization",
        "paper_fidelity": "paper-adaptation",
    },
)
class DifferNetDetector(BaseVisionDeepDetector):
    def __init__(
        self,
        *,
        pretrained: bool = False,
        image_size: int = 448,
        n_scales: int = 3,
        n_flow_steps: int = 8,
        n_transforms: int = 4,
        n_transforms_test: int = 64,
        flow_hidden_dim: int = 2048,
        flow_clamp: float = 3.0,
        flow_dropout: float = 0.0,
        epochs: int = 192,
        batch_size: int = 24,
        learning_rate: float = 2e-4,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        if "train_difference" in kwargs:
            raise TypeError(
                "train_difference belonged to the removed kNN implementation and is not "
                "a DifferNet paper parameter."
            )
        if n_transforms <= 0 or n_transforms_test <= 0:
            raise ValueError("n_transforms and n_transforms_test must be positive.")
        if flow_hidden_dim <= 0:
            raise ValueError("flow_hidden_dim must be positive.")
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)
        self.n_scales = int(n_scales)
        self.n_flow_steps = int(n_flow_steps)
        self.n_transforms = int(n_transforms)
        self.n_transforms_test = int(n_transforms_test)
        self.flow_hidden_dim = int(flow_hidden_dim)
        self.flow_clamp = float(flow_clamp)
        self.flow_dropout = float(flow_dropout)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        requested_random_state = None if random_state is None else int(random_state)

        base_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        train_transform = kwargs.pop("train_transform", base_transform)
        eval_transform = kwargs.pop("eval_transform", base_transform)
        super().__init__(
            lr=self.learning_rate,
            epoch_num=self.epochs,
            batch_size=int(batch_size),
            device=device,
            random_state=None,
            verbose=0,
            train_transform=train_transform,
            eval_transform=eval_transform,
            **kwargs,
        )
        self.random_state = requested_random_state

    def build_model(self) -> DifferNetNetwork:
        with torch.random.fork_rng(devices=[]):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            model = DifferNetNetwork(
                pretrained=self.pretrained,
                image_size=self.image_size,
                n_scales=self.n_scales,
                n_flow_steps=self.n_flow_steps,
                flow_hidden_dim=self.flow_hidden_dim,
                flow_clamp=self.flow_clamp,
                flow_dropout=self.flow_dropout,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            model.flow.parameters(),
            lr=self.learning_rate,
            betas=(0.8, 0.8),
            eps=1e-4,
            weight_decay=1e-5,
        )
        return model

    @staticmethod
    def _rotate_batch(images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        radians = angles.to(dtype=images.dtype) * (torch.pi / 180.0)
        cosines, sines = torch.cos(radians), torch.sin(radians)
        theta = images.new_zeros((images.shape[0], 2, 3))
        theta[:, 0, 0] = cosines
        theta[:, 0, 1] = sines
        theta[:, 1, 0] = -sines
        theta[:, 1, 1] = cosines
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        rotated = F.grid_sample(
            images,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        # torchvision rotates PIL inputs with a black fill before ImageNet
        # normalization. Restore that fill after rotating normalized tensors.
        mask = F.grid_sample(
            images.new_ones((images.shape[0], 1, images.shape[2], images.shape[3])),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return rotated * mask + (-mean / std) * (1.0 - mask)

    def _transformed_batch(self, images: torch.Tensor, *, training: bool) -> torch.Tensor:
        count = self.n_transforms if training else self.n_transforms_test
        expanded = images.repeat((count, 1, 1, 1))
        if training:
            angles = torch.empty(expanded.shape[0], device=images.device).uniform_(-180.0, 180.0)
        else:
            angles = (
                torch.arange(count, device=images.device, dtype=images.dtype) * (360.0 / count)
            ).repeat_interleave(images.shape[0])
        return self._rotate_batch(expanded, angles)

    @staticmethod
    def _nll(z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        return (0.5 * z.square().sum(dim=1) - logdet) / z.shape[1]

    def training_forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        images, _ = batch
        images = self._transformed_batch(images.to(self.device), training=True)
        self.model.train()
        self.model.feature_extractor.eval()
        self.optimizer.zero_grad(set_to_none=True)
        z, logdet = self.model(images)
        loss = self._nll(z, logdet).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluating_forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> NDArray:
        images, _ = batch
        batch_size = images.shape[0]
        images = self._transformed_batch(images.to(self.device), training=False)
        self.model.eval()
        z, _logdet = self.model(images)
        # The authors train with flow likelihood, but rank anomalies by the
        # mean latent energy across image transformations.
        scores = z.square().mean(dim=1).view(self.n_transforms_test, batch_size).mean(dim=0)
        return scores.cpu().numpy()

    def fit(
        self,
        x: object = MISSING,
        y: Optional[Iterable[int]] = None,
        **kwargs: object,
    ) -> "DifferNetDetector":
        value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        fitted = super().fit(value, y)
        self.is_fitted_ = True
        return fitted

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        items = list(resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
        if not items:
            return np.zeros((0,), dtype=np.float64)
        effective_batch_size = batch_size
        if effective_batch_size is None:
            effective_batch_size = max(
                1, self.batch_size * self.n_transforms // self.n_transforms_test
            )
        return np.asarray(
            super().decision_function(items, batch_size=effective_batch_size), dtype=np.float64
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        if getattr(self, "model", None) is None or not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 3,
                "detector": "vision_differnet",
                "config": {
                    "pretrained": self.pretrained,
                    "image_size": self.image_size,
                    "n_scales": self.n_scales,
                    "n_flow_steps": self.n_flow_steps,
                    "n_transforms": self.n_transforms,
                    "n_transforms_test": self.n_transforms_test,
                    "flow_hidden_dim": self.flow_hidden_dim,
                    "flow_clamp": self.flow_clamp,
                    "flow_dropout": self.flow_dropout,
                    "learning_rate": self.learning_rate,
                },
                "model_state_dict": export_module_state_dict(self.model),
                "decision_scores_": torch.as_tensor(self.decision_scores_, dtype=torch.float64),
                "threshold_": float(self.threshold_),
                "labels_": torch.as_tensor(self.labels_, dtype=torch.int64),
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema_version") != 3:
            raise ValueError(
                "Unsupported DifferNet checkpoint. Retrain checkpoints created before the paper flow alignment."
            )
        if str(payload.get("detector", "")) not in {"vision_differnet", "differnet"}:
            raise ValueError("Invalid DifferNet checkpoint: detector marker mismatch.")

        config = payload.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("Invalid DifferNet checkpoint: missing config.")
        self.pretrained = bool(config.get("pretrained", self.pretrained))
        self.image_size = int(config.get("image_size", self.image_size))
        self.n_scales = int(config.get("n_scales", self.n_scales))
        self.n_flow_steps = int(config.get("n_flow_steps", self.n_flow_steps))
        self.n_transforms = int(config.get("n_transforms", self.n_transforms))
        self.n_transforms_test = int(config.get("n_transforms_test", self.n_transforms_test))
        self.flow_hidden_dim = int(config.get("flow_hidden_dim", self.flow_hidden_dim))
        self.flow_clamp = float(config.get("flow_clamp", self.flow_clamp))
        self.flow_dropout = float(config.get("flow_dropout", self.flow_dropout))
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))

        self.model = self.build_model()
        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Invalid DifferNet checkpoint: missing model state.")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.decision_scores_ = np.asarray(payload["decision_scores_"], dtype=np.float64)
        self.threshold_ = float(payload["threshold_"])
        self.labels_ = np.asarray(payload["labels_"], dtype=np.int64)
        self.is_fitted_ = True


__all__ = [
    "DifferNetCouplingBlock",
    "DifferNetDetector",
    "DifferNetFlow",
    "DifferNetNetwork",
    "DifferNetSubnet",
]
