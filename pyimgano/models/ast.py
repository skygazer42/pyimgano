"""Asymmetric Student-Teacher Networks (AST) for RGB anomaly detection.

This module implements the MVTec AD path from the WACV 2023 paper: a frozen
ImageNet EfficientNet-B5 feature extractor, a conditional normalizing-flow
teacher, and an asymmetric residual convolutional student.  The teacher is
trained first by maximum likelihood; the student then regresses its outputs on
normal training images only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from pyimgano.datasets import ImagePathDataset
from pyimgano.models._imagenet_preprocess import preprocess_imagenet_batch
from pyimgano.utils.random_state import isolated_random_state_method
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .cflow import positional_encoding_2d
from .deep_io import export_module_state_dict, safe_torch_load
from .registry import register_model

logger = logging.getLogger(__name__)

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."
_PAPER_BACKBONE = "efficientnet_b5"


def _fixed_channel_permutation(dimensions: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact fixed NumPy permutation convention used by the authors."""

    permutation = np.random.RandomState(int(seed)).permutation(int(dimensions))
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(int(dimensions))
    return (
        torch.as_tensor(permutation, dtype=torch.long),
        torch.as_tensor(inverse, dtype=torch.long),
    )


class ASTFeatureExtractor(nn.Module):
    """Frozen EfficientNet-B5 through the paper's layer-36 feature map."""

    out_channels = 304

    def __init__(
        self,
        backbone: str = _PAPER_BACKBONE,
        *,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if backbone != _PAPER_BACKBONE:
            raise ValueError("AST's paper RGB path requires backbone='efficientnet_b5'.")

        network, _ = load_torchvision_model(backbone, pretrained=bool(pretrained))
        stages = list(network.features.children())
        if len(stages) < 7:
            raise RuntimeError("Unexpected EfficientNet-B5 structure: expected at least 7 stages.")

        # efficientnet_pytorch block index 35 is the final 304-channel block in
        # stage 6.  torchvision's equivalent is features[:7].
        self.features = nn.Sequential(*stages[:7])
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        del mode
        return super().train(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class ASTSubnet(nn.Module):
    """The shallow convolutional s/t subnet used inside an AST coupling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int,
        kernel_size: int,
        gamma_trick: bool = True,
    ) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv1 = nn.Conv2d(
            int(in_channels),
            int(hidden_channels),
            kernel_size=int(kernel_size),
            padding=padding,
            padding_mode="replicate",
        )
        self.conv2 = nn.Conv2d(
            int(hidden_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            padding=padding,
            padding_mode="replicate",
        )
        self.activation = nn.ReLU()
        if gamma_trick:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gamma", None)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv2(self.activation(self.conv1(value)))
        return value if self.gamma is None else value * self.gamma


class ASTCouplingBlock(nn.Module):
    """Fixed permutation followed by the paper's two-sided affine coupling."""

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        *,
        hidden_channels: int,
        kernel_size: int,
        clamp: float,
        seed: int,
        gamma_trick: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim < 2:
            raise ValueError("feature_dim must be at least 2.")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if clamp <= 0:
            raise ValueError("clamp must be positive.")

        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.split_dim_1 = self.feature_dim // 2
        self.split_dim_2 = self.feature_dim - self.split_dim_1
        self.clamp = float(clamp)

        self.subnet_1 = ASTSubnet(
            self.split_dim_1 + self.condition_dim,
            self.split_dim_2 * 2,
            hidden_channels=int(hidden_channels),
            kernel_size=int(kernel_size),
            gamma_trick=bool(gamma_trick),
        )
        self.subnet_2 = ASTSubnet(
            self.split_dim_2 + self.condition_dim,
            self.split_dim_1 * 2,
            hidden_channels=int(hidden_channels),
            kernel_size=int(kernel_size),
            gamma_trick=bool(gamma_trick),
        )

        permutation, inverse = _fixed_channel_permutation(self.feature_dim, int(seed))
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", inverse)

    def _log_scale(self, value: torch.Tensor) -> torch.Tensor:
        # Alpha-clamping used by the FrEIA snapshot in the authors' release.
        return self.clamp * 0.636 * torch.atan(value / self.clamp)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim != 4 or condition.ndim != 4:
            raise ValueError("AST flow values and conditions must be 4-D tensors.")
        if value.shape[0] != condition.shape[0] or value.shape[2:] != condition.shape[2:]:
            raise ValueError("AST flow values and conditions must share batch/spatial dimensions.")
        if value.shape[1] != self.feature_dim or condition.shape[1] != self.condition_dim:
            raise ValueError("AST flow value or condition channels do not match this block.")

        if not reverse:
            value = value.index_select(1, self.permutation)

        value_1, value_2 = torch.split(
            value,
            (self.split_dim_1, self.split_dim_2),
            dim=1,
        )

        if reverse:
            scale_1, shift_1 = self.subnet_1(torch.cat((value_1, condition), dim=1)).split(
                self.split_dim_2, dim=1
            )
            log_scale_1 = self._log_scale(scale_1)
            output_2 = (value_2 - shift_1) * torch.exp(-log_scale_1)

            scale_2, shift_2 = self.subnet_2(torch.cat((output_2, condition), dim=1)).split(
                self.split_dim_1, dim=1
            )
            log_scale_2 = self._log_scale(scale_2)
            output_1 = (value_1 - shift_2) * torch.exp(-log_scale_2)

            output = torch.cat((output_1, output_2), dim=1)
            output = output.index_select(1, self.inverse_permutation)
            logdet = -(log_scale_1.sum(dim=1) + log_scale_2.sum(dim=1))
            return output, logdet

        scale_2, shift_2 = self.subnet_2(torch.cat((value_2, condition), dim=1)).split(
            self.split_dim_1, dim=1
        )
        log_scale_2 = self._log_scale(scale_2)
        output_1 = torch.exp(log_scale_2) * value_1 + shift_2

        scale_1, shift_1 = self.subnet_1(torch.cat((output_1, condition), dim=1)).split(
            self.split_dim_2, dim=1
        )
        log_scale_1 = self._log_scale(scale_1)
        output_2 = torch.exp(log_scale_1) * value_2 + shift_1

        output = torch.cat((output_1, output_2), dim=1).clamp(-1e6, 1e6)
        logdet = log_scale_1.sum(dim=1) + log_scale_2.sum(dim=1)
        return output, logdet


class ASTTeacherFlow(nn.Module):
    """RealNVP teacher used to create AST regression targets."""

    def __init__(
        self,
        feature_dim: int = 304,
        condition_dim: int = 32,
        *,
        hidden_channels: int = 1024,
        kernel_sizes: Sequence[int] = (3, 3, 3, 5),
        clamp: float = 3.0,
        gamma_trick: bool = True,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes cannot be empty.")
        self.blocks = nn.ModuleList(
            [
                ASTCouplingBlock(
                    int(feature_dim),
                    int(condition_dim),
                    hidden_channels=int(hidden_channels),
                    kernel_size=int(kernel_size),
                    clamp=float(clamp),
                    seed=index,
                    gamma_trick=bool(gamma_trick),
                )
                for index, kernel_size in enumerate(kernel_sizes)
            ]
        )

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = value.new_zeros((value.shape[0], value.shape[2], value.shape[3]))
        blocks = reversed(self.blocks) if reverse else self.blocks
        for block in blocks:
            value, block_logdet = block(value, condition, reverse=reverse)
            logdet = logdet + block_logdet
        return value, logdet


class ASTResidualBlock(nn.Module):
    """Two 3x3 Conv-BN-LeakyReLU layers with a residual connection."""

    def __init__(self, channels: int, *, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(channels))
        self.conv2 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(int(channels))
        self.activation = nn.LeakyReLU(float(negative_slope))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.activation(self.bn1(self.conv1(value)))
        value = self.activation(self.bn2(self.conv2(value)))
        return value + residual


class ASTStudent(nn.Module):
    """Paper feed-forward student with four residual convolutional blocks."""

    def __init__(
        self,
        feature_dim: int = 304,
        condition_dim: int = 32,
        *,
        hidden_channels: int = 1024,
        n_blocks: int = 4,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or condition_dim <= 0 or hidden_channels <= 0 or n_blocks <= 0:
            raise ValueError("AST student dimensions and n_blocks must be positive.")
        self.input_conv = nn.Conv2d(
            int(feature_dim) + int(condition_dim),
            int(hidden_channels),
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.LeakyReLU(float(negative_slope))
        self.residual_blocks = nn.ModuleList(
            [
                ASTResidualBlock(int(hidden_channels), negative_slope=float(negative_slope))
                for _ in range(int(n_blocks))
            ]
        )
        self.output_conv = nn.Conv2d(
            int(hidden_channels),
            int(feature_dim),
            kernel_size=3,
            padding=1,
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = self.activation(self.input_conv(torch.cat((condition, value), dim=1)))
        for block in self.residual_blocks:
            value = block(value)
        return self.output_conv(value)


class ASTNetwork(nn.Module):
    """Frozen feature extractor plus the paper's teacher/student pair."""

    def __init__(
        self,
        *,
        backbone: str,
        pretrained_backbone: bool,
        feature_dim: int,
        condition_dim: int,
        teacher_hidden_channels: int,
        student_hidden_channels: int,
        student_blocks: int,
        kernel_sizes: Sequence[int],
        clamp: float,
        gamma_trick: bool,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.feature_extractor = ASTFeatureExtractor(
            backbone,
            pretrained=bool(pretrained_backbone),
        )
        self.teacher = ASTTeacherFlow(
            self.feature_dim,
            self.condition_dim,
            hidden_channels=int(teacher_hidden_channels),
            kernel_sizes=tuple(int(value) for value in kernel_sizes),
            clamp=float(clamp),
            gamma_trick=bool(gamma_trick),
        )
        self.student = ASTStudent(
            self.feature_dim,
            self.condition_dim,
            hidden_channels=int(student_hidden_channels),
            n_blocks=int(student_blocks),
            negative_slope=float(negative_slope),
        )

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()
        features = self.feature_extractor(images)
        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"AST expected {self.feature_dim} feature channels, got {features.shape[1]}."
            )
        return features.detach()

    def positional_condition(self, features: torch.Tensor) -> torch.Tensor:
        condition = positional_encoding_2d(
            self.condition_dim,
            features.shape[2],
            features.shape[3],
            device=features.device,
            dtype=features.dtype,
        )
        return condition.unsqueeze(0).expand(features.shape[0], -1, -1, -1)


def _ast_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


@register_model(
    "vision_ast",
    tags=(
        "vision",
        "deep",
        "ast",
        "student-teacher",
        "normalizing-flow",
        "pixel_map",
    ),
    metadata={
        "description": "AST RGB path with a conditional-flow teacher and residual convolutional student",
        "paper": "Asymmetric Student-Teacher Networks for Industrial Anomaly Detection",
        "paper_url": "https://openaccess.thecvf.com/content/WACV2023/html/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.html",
        "author_code": "https://github.com/marco-rudolph/AST",
        "year": 2023,
        "supervision": "one-class",
        "implementation_status": "paper-mvtec-ad-rgb-path-aligned",
        "paper_fidelity": "paper-adaptation",
        "type": "knowledge-distillation",
    },
)
class VisionAST(BaseVisionDeepDetector):
    """Paper-aligned AST detector for the RGB-only MVTec AD protocol.

    Defaults match the paper's MVTec AD network and optimization settings.
    ImageNet weights remain opt-in to avoid an implicit download; set
    ``pretrained_backbone=True`` for the paper protocol.  The paper's optional
    MVTec 3D-AD depth/foreground path is outside this RGB detector.
    """

    def __init__(
        self,
        backbone: str = _PAPER_BACKBONE,
        pretrained_backbone: bool = False,
        image_size: int = 768,
        feature_dim: int = 304,
        condition_dim: int = 32,
        n_coupling_blocks: int = 4,
        teacher_hidden_channels: int = 1024,
        student_hidden_channels: int = 1024,
        student_blocks: int = 4,
        kernel_sizes: Optional[Sequence[int]] = None,
        clamp: float = 3.0,
        gamma_trick: bool = True,
        negative_slope: float = 0.2,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 8,
        epochs: int = 240,
        num_workers: int = 0,
        contamination: float = 0.1,
        device: Optional[str] = None,
        verbose: int = 1,
        random_state: Optional[int] = 42,
        **kwargs: object,
    ) -> None:
        if "anomaly_ratio" in kwargs or "hidden_channels" in kwargs:
            raise TypeError(
                "Synthetic anomalies and the legacy decoder were removed; use the AST paper "
                "teacher_hidden_channels/student_hidden_channels parameters."
            )
        if backbone != _PAPER_BACKBONE:
            raise ValueError("AST's paper RGB path requires backbone='efficientnet_b5'.")
        if image_size <= 0 or feature_dim <= 1:
            raise ValueError("image_size must be positive and feature_dim must be at least 2.")
        if condition_dim <= 0 or condition_dim % 4:
            raise ValueError("condition_dim must be positive and divisible by 4.")
        if n_coupling_blocks <= 0:
            raise ValueError("n_coupling_blocks must be positive.")
        if teacher_hidden_channels <= 0 or student_hidden_channels <= 0 or student_blocks <= 0:
            raise ValueError("AST hidden dimensions and student_blocks must be positive.")
        if clamp <= 0 or negative_slope < 0:
            raise ValueError("clamp must be positive and negative_slope non-negative.")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if batch_size <= 0 or epochs < 0 or num_workers < 0:
            raise ValueError(
                "batch_size must be positive; epochs/num_workers must be non-negative."
            )

        if kernel_sizes is None:
            resolved_kernel_sizes = (3,) * (int(n_coupling_blocks) - 1) + (5,)
        else:
            resolved_kernel_sizes = tuple(int(value) for value in kernel_sizes)
        if len(resolved_kernel_sizes) != int(n_coupling_blocks):
            raise ValueError("kernel_sizes length must match n_coupling_blocks.")
        if any(value <= 0 or value % 2 == 0 for value in resolved_kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers.")

        transform = _ast_transform(int(image_size))
        super().__init__(
            contamination=float(contamination),
            preprocessing=True,
            lr=float(learning_rate),
            epoch_num=int(epochs),
            batch_size=int(batch_size),
            optimizer_name="adam",
            device=device,
            random_state=random_state,
            verbose=int(verbose),
            train_transform=transform,
            eval_transform=transform,
            **kwargs,
        )
        self.backbone = str(backbone)
        self.pretrained_backbone = bool(pretrained_backbone)
        self.image_size = int(image_size)
        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.n_coupling_blocks = int(n_coupling_blocks)
        self.teacher_hidden_channels = int(teacher_hidden_channels)
        self.student_hidden_channels = int(student_hidden_channels)
        self.student_blocks = int(student_blocks)
        self.kernel_sizes = resolved_kernel_sizes
        self.clamp = float(clamp)
        self.gamma_trick = bool(gamma_trick)
        self.negative_slope = float(negative_slope)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.num_workers = int(num_workers)
        self.model: ASTNetwork | None = None
        self.feature_extractor_: ASTFeatureExtractor | None = None
        self.teacher_: ASTTeacherFlow | None = None
        self.student_: ASTStudent | None = None
        self.is_fitted_ = False

    def _preprocess(self, x: NDArray) -> torch.Tensor:
        """Apply the repository's shared ImageNet normalization."""

        return preprocess_imagenet_batch(x)

    @staticmethod
    def _materialize_inputs(value: object) -> list[object]:
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                return [value]
            if value.ndim == 4:
                return [value[index] for index in range(value.shape[0])]
            raise ValueError("AST arrays must have shape HWC/CHW or NHWC/NCHW.")
        if isinstance(value, (str, Path)):
            return [value]
        return list(cast(Iterable[object], value))

    def _array_tensor(self, items: Sequence[object]) -> torch.Tensor:
        arrays = [np.asarray(item) for item in items]
        try:
            batch = np.stack(arrays)
        except ValueError as exc:
            raise ValueError("AST in-memory images must share one shape.") from exc
        if not np.issubdtype(batch.dtype, np.number):
            raise TypeError("AST input arrays must contain numeric RGB values.")
        if not np.isfinite(batch).all():
            raise ValueError("AST input arrays must contain only finite values.")
        minimum = float(batch.min())
        maximum = float(batch.max())
        if minimum < 0 or maximum > 255:
            raise ValueError("AST image values must be in [0, 1] or [0, 255].")
        if np.issubdtype(batch.dtype, np.floating):
            if maximum <= 1:
                batch = batch * 255.0
        tensor = self._preprocess(batch)
        return F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def _make_loader(
        self,
        items: Sequence[object],
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        if not items:
            return DataLoader(
                TensorDataset(torch.empty((0, 3, 1, 1))),
                batch_size=batch_size,
                pin_memory=self.device.type == "cuda",
            )
        if isinstance(items[0], np.ndarray):
            if not all(isinstance(item, np.ndarray) for item in items):
                raise TypeError("AST inputs cannot mix arrays and paths.")
            dataset = TensorDataset(self._array_tensor(items))
        else:
            if not all(isinstance(item, (str, Path)) for item in items):
                raise TypeError("AST inputs must be all RGB arrays or all image paths.")
            dataset = ImagePathDataset(
                [str(item) for item in items],
                transform=self.eval_transform,
                return_full_path=True,
            )
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    @staticmethod
    def _images_from_batch(batch: object) -> torch.Tensor:
        if not isinstance(batch, (tuple, list)) or not batch:
            raise TypeError("AST data loader must return an image tuple.")
        images = batch[0]
        if not isinstance(images, torch.Tensor):
            raise TypeError("AST data loader must return image tensors.")
        return images

    def _build_model(self, *, load_pretrained: Optional[bool] = None) -> ASTNetwork:
        use_pretrained = self.pretrained_backbone if load_pretrained is None else load_pretrained
        model = ASTNetwork(
            backbone=self.backbone,
            pretrained_backbone=bool(use_pretrained),
            feature_dim=self.feature_dim,
            condition_dim=self.condition_dim,
            teacher_hidden_channels=self.teacher_hidden_channels,
            student_hidden_channels=self.student_hidden_channels,
            student_blocks=self.student_blocks,
            kernel_sizes=self.kernel_sizes,
            clamp=self.clamp,
            gamma_trick=self.gamma_trick,
            negative_slope=self.negative_slope,
        ).to(self.device)
        self.model = model
        self.feature_extractor_ = model.feature_extractor
        self.teacher_ = model.teacher
        self.student_ = model.student
        return model

    def _require_model(self) -> ASTNetwork:
        if self.model is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        return self.model

    @staticmethod
    def _teacher_loss(latent: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        """Equation 3: mean spatial negative log likelihood."""

        return (0.5 * latent.square().sum(dim=1) - logdet).mean()

    @staticmethod
    def _student_map(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Author implementation of Equation 4, averaged over feature channels."""

        return (target - output).square().mean(dim=1)

    @torch.no_grad()
    def _extract_training_features(self, loader: DataLoader) -> torch.Tensor:
        model = self._require_model()
        features = []
        for batch in loader:
            images = self._images_from_batch(batch).to(self.device, non_blocking=True)
            features.append(model.extract_features(images).cpu())
        return torch.cat(features)

    @isolated_random_state_method
    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionAST":
        del y
        value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        items = self._materialize_inputs(value)
        if not items:
            raise ValueError("Training set cannot be empty.")

        model = self._build_model()
        image_loader = self._make_loader(items, batch_size=self.batch_size, shuffle=False)
        training_features = self._extract_training_features(image_loader)
        feature_loader = DataLoader(
            TensorDataset(training_features),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )

        teacher_optimizer = torch.optim.Adam(
            model.teacher.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        model.teacher.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            updates = 0
            for (features,) in feature_loader:
                features = features.to(self.device, non_blocking=True)
                condition = model.positional_condition(features)
                latent, logdet = model.teacher(features, condition)
                loss = self._teacher_loss(latent, logdet)
                teacher_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                teacher_optimizer.step()
                total_loss += float(loss.detach().item())
                updates += 1
            if self.verbose:
                logger.info(
                    "AST teacher epoch %d/%d loss %.6f",
                    epoch + 1,
                    self.epochs,
                    total_loss / max(1, updates),
                )

        model.teacher.eval()
        del teacher_optimizer
        student_optimizer = torch.optim.Adam(
            model.student.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        model.student.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            updates = 0
            for (features,) in feature_loader:
                features = features.to(self.device, non_blocking=True)
                condition = model.positional_condition(features)
                with torch.no_grad():
                    target, _ = model.teacher(features, condition)
                output = model.student(features, condition)
                loss = self._student_map(target, output).mean()
                student_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                student_optimizer.step()
                total_loss += float(loss.detach().item())
                updates += 1
            if self.verbose:
                logger.info(
                    "AST student epoch %d/%d loss %.6f",
                    epoch + 1,
                    self.epochs,
                    total_loss / max(1, updates),
                )

        model.eval()
        self.is_fitted_ = True
        self.decision_scores_ = self.decision_function(items)
        self._process_decision_scores()
        self._set_n_classes(None)
        return self

    @torch.no_grad()
    def _score_maps(self, items: Sequence[object], *, batch_size: int) -> torch.Tensor:
        model = self._require_model()
        model.eval()
        maps = []
        loader = self._make_loader(items, batch_size=batch_size, shuffle=False)
        for batch in loader:
            images = self._images_from_batch(batch).to(self.device, non_blocking=True)
            features = model.extract_features(images)
            condition = model.positional_condition(features)
            target, _ = model.teacher(features, condition)
            output = model.student(features, condition)
            maps.append(self._student_map(target, output).cpu())
        if not maps:
            return torch.empty((0, 0, 0), dtype=torch.float32)
        return torch.cat(maps)

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.model is None:
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
        value = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        items = self._materialize_inputs(value)
        maps = self._score_maps(items, batch_size=self.batch_size)
        if maps.numel() == 0:
            return np.zeros((0,), dtype=np.float64)
        # The paper uses the spatial mean for RGB-only MVTec AD.
        return maps.mean(dim=(-2, -1)).numpy().astype(np.float64, copy=False)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        value = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        if batch_size is None:
            return self.predict(value)
        effective_batch_size = int(batch_size)
        if effective_batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        old_batch_size = self.batch_size
        try:
            self.batch_size = effective_batch_size
            return self.predict(value)
        finally:
            self.batch_size = old_batch_size

    def predict_anomaly_map(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        self._check_fitted()
        value = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        effective_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if effective_batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        items = self._materialize_inputs(value)
        maps = self._score_maps(items, batch_size=effective_batch_size)
        if maps.numel() == 0:
            return np.zeros((0, self.image_size, self.image_size), dtype=np.float32)
        maps = F.interpolate(
            maps.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        return maps.numpy().astype(np.float32, copy=False)

    def get_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        """Compatibility alias for :meth:`predict_anomaly_map`."""

        value = resolve_legacy_x_keyword(x, kwargs, method_name="get_anomaly_map")
        return self.predict_anomaly_map(value)

    def save_checkpoint(self, path: str | Path) -> Path:
        self._check_fitted()
        model = self._require_model()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "detector": "vision_ast",
                "config": {
                    "backbone": self.backbone,
                    "pretrained_backbone": self.pretrained_backbone,
                    "image_size": self.image_size,
                    "feature_dim": self.feature_dim,
                    "condition_dim": self.condition_dim,
                    "n_coupling_blocks": self.n_coupling_blocks,
                    "teacher_hidden_channels": self.teacher_hidden_channels,
                    "student_hidden_channels": self.student_hidden_channels,
                    "student_blocks": self.student_blocks,
                    "kernel_sizes": list(self.kernel_sizes),
                    "clamp": self.clamp,
                    "gamma_trick": self.gamma_trick,
                    "negative_slope": self.negative_slope,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                },
                "model_state_dict": export_module_state_dict(model),
                "decision_scores_": torch.as_tensor(self.decision_scores_, dtype=torch.float64),
                "threshold_": float(self.threshold_),
                "labels_": torch.as_tensor(self.labels_, dtype=torch.int64),
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(
                "Unsupported AST checkpoint. Legacy synthetic-anomaly proxy checkpoints "
                "cannot be loaded into the paper architecture."
            )
        if payload.get("detector") != "vision_ast":
            raise ValueError("Invalid AST checkpoint: detector marker mismatch.")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("Invalid AST checkpoint: missing config.")

        self.backbone = str(config["backbone"])
        self.pretrained_backbone = bool(config["pretrained_backbone"])
        self.image_size = int(config["image_size"])
        self.feature_dim = int(config["feature_dim"])
        self.condition_dim = int(config["condition_dim"])
        self.n_coupling_blocks = int(config["n_coupling_blocks"])
        self.teacher_hidden_channels = int(config["teacher_hidden_channels"])
        self.student_hidden_channels = int(config["student_hidden_channels"])
        self.student_blocks = int(config["student_blocks"])
        self.kernel_sizes = tuple(int(value) for value in config["kernel_sizes"])
        self.clamp = float(config["clamp"])
        self.gamma_trick = bool(config["gamma_trick"])
        self.negative_slope = float(config["negative_slope"])
        self.learning_rate = float(config["learning_rate"])
        self.lr = self.learning_rate
        self.weight_decay = float(config["weight_decay"])
        self.batch_size = int(config["batch_size"])
        self.epochs = int(config["epochs"])
        self.epoch_num = self.epochs
        transform = _ast_transform(self.image_size)
        self.train_transform = transform
        self.eval_transform = transform

        # The checkpoint contains the backbone weights, so loading never needs
        # network access even if it was trained with ImageNet initialization.
        model = self._build_model(load_pretrained=False)
        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Invalid AST checkpoint: missing model_state_dict.")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.decision_scores_ = np.asarray(payload["decision_scores_"], dtype=np.float64)
        self.threshold_ = float(payload["threshold_"])
        self.labels_ = np.asarray(payload["labels_"], dtype=np.int64)
        self.is_fitted_ = True
        self._set_n_classes(None)


__all__ = [
    "ASTCouplingBlock",
    "ASTFeatureExtractor",
    "ASTNetwork",
    "ASTResidualBlock",
    "ASTStudent",
    "ASTSubnet",
    "ASTTeacherFlow",
    "VisionAST",
]
