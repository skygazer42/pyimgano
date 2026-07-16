# -*- coding: utf-8 -*-
"""EfficientAD paper architecture and training objective.

The paper requires a PDN teacher distilled on ImageNet before per-dataset
training.  PyImgAno never downloads that external asset implicitly: strict
mode requires an explicit teacher checkpoint and ImageNet-style penalty data.

Reference:
    Batzner, Heckler, & König, EfficientAD: Accurate Visual Anomaly Detection
    at Millisecond-Level Latencies, WACV 2024.
"""

from __future__ import annotations

import math
from itertools import chain
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torchvision import transforms
from torchvision.datasets import ImageFolder

from pyimgano.utils.random_state import isolated_random_state_method

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .deep_io import export_module_state_dict, safe_torch_load
from .registry import register_model

_PAPER_IMAGE_SIZE = (256, 256)
_PAPER_CHANNELS = 384
_PAPER_STEPS = 70_000
_PAPER_HARD_QUANTILE = 0.999
_PAPER_MAP_QUANTILES = (0.9, 0.995)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

ImageInput = Union[str, Path, np.ndarray]


def _to_hw(image_size: int | Tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, tuple):
        hw = (int(image_size[0]), int(image_size[1]))
    else:
        hw = (int(image_size), int(image_size))
    if hw != _PAPER_IMAGE_SIZE:
        raise ValueError("EfficientAD's paper autoencoder requires image_size=256.")
    return hw


def _image_transform(image_size: tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])


def _imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


class SmallPatchDescriptionNetwork(nn.Module):
    """EfficientAD-S PDN from supplementary Table 6."""

    def __init__(self, out_channels: int, *, padding: bool = False) -> None:
        super().__init__()
        pad = int(bool(padding))
        self.conv1 = nn.Conv2d(3, 128, 4, padding=3 * pad)
        self.pool1 = nn.AvgPool2d(2, stride=2, padding=pad)
        self.conv2 = nn.Conv2d(128, 256, 4, padding=3 * pad)
        self.pool2 = nn.AvgPool2d(2, stride=2, padding=pad)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=pad)
        self.conv4 = nn.Conv2d(256, int(out_channels), 4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = _imagenet_normalize(images)
        features = self.pool1(F.relu(self.conv1(features)))
        features = self.pool2(F.relu(self.conv2(features)))
        features = F.relu(self.conv3(features))
        return self.conv4(features)


class MediumPatchDescriptionNetwork(nn.Module):
    """EfficientAD-M PDN from supplementary Table 7."""

    def __init__(self, out_channels: int, *, padding: bool = False) -> None:
        super().__init__()
        pad = int(bool(padding))
        self.conv1 = nn.Conv2d(3, 256, 4, padding=3 * pad)
        self.pool1 = nn.AvgPool2d(2, stride=2, padding=pad)
        self.conv2 = nn.Conv2d(256, 512, 4, padding=3 * pad)
        self.pool2 = nn.AvgPool2d(2, stride=2, padding=pad)
        self.conv3 = nn.Conv2d(512, 512, 1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=pad)
        self.conv5 = nn.Conv2d(512, int(out_channels), 4)
        self.conv6 = nn.Conv2d(int(out_channels), int(out_channels), 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = _imagenet_normalize(images)
        features = self.pool1(F.relu(self.conv1(features)))
        features = self.pool2(F.relu(self.conv2(features)))
        features = F.relu(self.conv3(features))
        features = F.relu(self.conv4(features))
        features = F.relu(self.conv5(features))
        return self.conv6(features)


class EfficientADAutoEncoder(nn.Module):
    """64-D bottleneck autoencoder from supplementary Table 8."""

    def __init__(self, out_channels: int, *, padding: bool = False) -> None:
        super().__init__()
        self.padding = bool(padding)
        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.Conv2d(32, 32, 4, stride=2, padding=1),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.Conv2d(64, 64, 4, stride=2, padding=1),
                nn.Conv2d(64, 64, 4, stride=2, padding=1),
                nn.Conv2d(64, 64, 8),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.Conv2d(64, int(out_channels), 3, padding=1),
            ]
        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(6)])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = _imagenet_normalize(images)
        for layer in self.encoder[:-1]:
            features = F.relu(layer(features))
        features = self.encoder[-1](features)

        sizes = ((3, 3), (8, 8), (15, 15), (32, 32), (63, 63), (127, 127))
        for size, layer, dropout in zip(sizes, self.decoder[:6], self.dropouts):
            features = F.interpolate(features, size=size, mode="bilinear", align_corners=False)
            features = dropout(F.relu(layer(features)))

        final_size = (64, 64) if self.padding else (56, 56)
        features = F.interpolate(features, size=final_size, mode="bilinear", align_corners=False)
        features = F.relu(self.decoder[6](features))
        return self.decoder[7](features)


def _build_pdn(model_size: str, out_channels: int, *, padding: bool) -> nn.Module:
    key = str(model_size).strip().lower()
    if key in {"s", "small"}:
        return SmallPatchDescriptionNetwork(out_channels, padding=padding)
    if key in {"m", "medium"}:
        return MediumPatchDescriptionNetwork(out_channels, padding=padding)
    raise ValueError("model_size must be 'small'/'s' or 'medium'/'m'.")


def _hard_feature_loss(distance: torch.Tensor) -> torch.Tensor:
    threshold = torch.quantile(distance, _PAPER_HARD_QUANTILE)
    return distance[distance >= threshold].mean()


def _normalize_map(anomaly_map: torch.Tensor, qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
    denominator = (qb - qa).clamp_min(torch.finfo(anomaly_map.dtype).eps)
    return 0.1 * (anomaly_map - qa) / denominator


class EfficientADModel(nn.Module):
    """Paper teacher, two-headed student, autoencoder, losses, and maps."""

    def __init__(
        self,
        *,
        model_size: str = "small",
        teacher_out_channels: int = _PAPER_CHANNELS,
        padding: bool = False,
    ) -> None:
        super().__init__()
        channels = int(teacher_out_channels)
        if channels <= 0:
            raise ValueError("teacher_out_channels must be positive.")
        self.teacher_out_channels = channels
        self.padding = bool(padding)
        self.teacher = _build_pdn(model_size, channels, padding=self.padding)
        self.student = _build_pdn(model_size, channels * 2, padding=self.padding)
        self.autoencoder = EfficientADAutoEncoder(channels, padding=self.padding)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

        self.register_buffer("teacher_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, channels, 1, 1))
        self.register_buffer("qa_st", torch.tensor(float("nan")))
        self.register_buffer("qb_st", torch.tensor(float("nan")))
        self.register_buffer("qa_ae", torch.tensor(float("nan")))
        self.register_buffer("qb_ae", torch.tensor(float("nan")))

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def normalized_teacher(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.teacher(images)
        return (output - self.teacher_mean) / self.teacher_std.clamp_min(1e-6)

    @staticmethod
    def augment_for_autoencoder(images: torch.Tensor) -> torch.Tensor:
        coefficient = float(torch.empty(1, device=images.device).uniform_(0.8, 1.2).item())
        index = int(torch.randint(0, 3, (1,), device=images.device).item())
        functions = (
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        )
        return functions[index](images, coefficient)

    def loss_terms(
        self, images: torch.Tensor, penalty_images: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher = self.normalized_teacher(images)
        student = self.student(images)
        distance_st = (teacher - student[:, : self.teacher_out_channels]) ** 2
        loss_st = _hard_feature_loss(distance_st)
        if penalty_images is not None:
            penalty = self.student(penalty_images)[:, : self.teacher_out_channels]
            loss_st = loss_st + penalty.square().mean()

        augmented = self.augment_for_autoencoder(images)
        teacher_augmented = self.normalized_teacher(augmented)
        autoencoder = self.autoencoder(augmented)
        student_autoencoder = self.student(augmented)[:, self.teacher_out_channels :]
        loss_ae = (teacher_augmented - autoencoder).square().mean()
        loss_stae = (autoencoder - student_autoencoder).square().mean()
        return loss_st, loss_ae, loss_stae

    def raw_maps(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        teacher = self.normalized_teacher(images)
        student = self.student(images)
        autoencoder = self.autoencoder(images)
        map_st = (
            (teacher - student[:, : self.teacher_out_channels]).square().mean(dim=1, keepdim=True)
        )
        map_ae = (
            (autoencoder - student[:, self.teacher_out_channels :])
            .square()
            .mean(dim=1, keepdim=True)
        )
        size = tuple(int(v) for v in images.shape[-2:])
        map_st = F.interpolate(map_st, size=size, mode="bilinear", align_corners=False)
        map_ae = F.interpolate(map_ae, size=size, mode="bilinear", align_corners=False)
        return map_st, map_ae

    def anomaly_map(self, images: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        map_st, map_ae = self.raw_maps(images)
        if normalize:
            if not all(
                bool(torch.isfinite(value).item())
                for value in (self.qa_st, self.qb_st, self.qa_ae, self.qb_ae)
            ):
                raise RuntimeError("EfficientAD map quantiles are not calibrated.")
            map_st = _normalize_map(map_st, self.qa_st, self.qb_st)
            map_ae = _normalize_map(map_ae, self.qa_ae, self.qb_ae)
        return 0.5 * map_st + 0.5 * map_ae


def _teacher_state_dict(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("EfficientAD teacher checkpoint must contain a state dictionary.")
    state: object = payload
    for key in ("state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break
    if not isinstance(state, dict):
        raise ValueError("EfficientAD teacher checkpoint is missing a state dictionary.")

    normalized = {str(key): value for key, value in state.items()}
    for prefix in ("model.teacher.", "teacher."):
        stripped = {
            key[len(prefix) :]: value for key, value in normalized.items() if key.startswith(prefix)
        }
        if stripped:
            return stripped
    return normalized


@register_model(
    "efficient_ad",
    tags=("vision", "deep", "distillation", "efficientad", "pixel_map"),
    metadata={
        "description": "EfficientAD PDN teacher/student and autoencoder paper adaptation",
        "paper": "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies",
        "year": 2024,
        "implementation_status": "paper-network-loss-score-and-defaults-aligned",
        "paper_fidelity": "paper-adaptation",
        "supports_save_load": True,
    },
    overwrite=True,
)
@register_model(
    "vision_efficientad",
    tags=("vision", "deep", "distillation", "efficientad", "pixel_map"),
    metadata={
        "description": "EfficientAD PDN teacher/student and autoencoder paper adaptation",
        "paper": "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies",
        "year": 2024,
        "implementation_status": "paper-network-loss-score-and-defaults-aligned",
        "paper_fidelity": "paper-adaptation",
        "supports_save_load": True,
    },
    overwrite=True,
)
class EfficientADDetector(BaseVisionDeepDetector):
    """EfficientAD-S/M with explicit paper-asset requirements.

    ``paper_strict=True`` requires both a distilled PDN teacher checkpoint and
    an ImageNet-style directory for the pretraining penalty.  Set it to false
    only for diagnostics; a random teacher or omitted penalty is not a paper
    reproduction.
    """

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        training_steps: int = _PAPER_STEPS,
        batch_size: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str | None = None,
        random_state: int = 0,
        verbose: int = 0,
        image_size: int | Tuple[int, int] = 256,
        model_size: str = "small",
        teacher_out_channels: int = _PAPER_CHANNELS,
        padding: bool = False,
        teacher_checkpoint: str | Path | None = None,
        imagenet_dir: str | Path | None = None,
        validation_fraction: float = 0.1,
        paper_strict: bool = True,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        hw = _to_hw(image_size)
        model_size_key = str(model_size).strip().lower()
        if model_size_key not in {"s", "small", "m", "medium"}:
            raise ValueError("model_size must be 'small'/'s' or 'medium'/'m'.")
        if int(training_steps) <= 0:
            raise ValueError("training_steps must be positive.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if bool(paper_strict) and int(batch_size) != 1:
            raise ValueError("EfficientAD paper training uses batch_size=1.")
        if not (0.0 < float(validation_fraction) < 1.0):
            raise ValueError("validation_fraction must be in (0, 1).")

        self.image_size = hw
        self.model_size = "small" if model_size_key in {"s", "small"} else "medium"
        self.teacher_out_channels = int(teacher_out_channels)
        self.padding = bool(padding)
        self.training_steps = int(training_steps)
        self.teacher_checkpoint = (
            None if teacher_checkpoint is None else str(Path(teacher_checkpoint))
        )
        self.imagenet_dir = None if imagenet_dir is None else str(Path(imagenet_dir))
        self.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path))
        self.validation_fraction = float(validation_fraction)
        self.paper_strict = bool(paper_strict)
        self.model: EfficientADModel | None = None
        self.teacher = None
        self.student = None
        self.autoencoder = None
        self._penalty_loader = None
        self._penalty_iterator = None
        self.teacher_checkpoint_loaded_ = False
        self.pretraining_penalty_applied_ = False

        transform = _image_transform(hw)
        super().__init__(
            contamination=float(contamination),
            preprocessing=True,
            lr=float(lr),
            epoch_num=1,
            batch_size=int(batch_size),
            optimizer_name="adam",
            criterion_name="mse",
            device=device,
            random_state=int(random_state),
            verbose=int(verbose),
            train_transform=transform,
            eval_transform=transform,
        )
        self.weight_decay = float(weight_decay)
        if self.checkpoint_path is not None:
            self.load_checkpoint(self.checkpoint_path)

    def _bind_model(self, model: EfficientADModel) -> EfficientADModel:
        self.model = model.to(self.device)
        self.teacher = self.model.teacher
        self.student = self.model.student
        self.autoencoder = self.model.autoencoder
        return self.model

    def build_model(self) -> EfficientADModel:
        model = EfficientADModel(
            model_size=self.model_size,
            teacher_out_channels=self.teacher_out_channels,
            padding=self.padding,
        )
        if self.teacher_checkpoint is not None:
            payload = safe_torch_load(self.teacher_checkpoint, map_location="cpu")
            model.teacher.load_state_dict(_teacher_state_dict(payload), strict=True)
            self.teacher_checkpoint_loaded_ = True
        else:
            self.teacher_checkpoint_loaded_ = False
        return self._bind_model(model)

    def _dataset(self, values: Sequence[ImageInput], *, transform):  # noqa: ANN001
        from pyimgano.datasets import VisionArrayDataset, VisionImageDataset

        if values and isinstance(values[0], np.ndarray):
            return VisionArrayDataset(
                images=values,
                transform=transform,
                fallback_shape=(3, *self.image_size),
            )
        return VisionImageDataset(
            image_paths=[str(value) for value in values],
            transform=transform,
            fallback_shape=(3, *self.image_size),
        )

    def _loader(
        self,
        values: Sequence[ImageInput],
        *,
        transform,
        shuffle: bool,
        batch_size: int | None = None,
    ):
        return torch.utils.data.DataLoader(
            self._dataset(values, transform=transform),
            batch_size=int(self.batch_size if batch_size is None else batch_size),
            shuffle=bool(shuffle),
            num_workers=int(self.num_workers),
        )

    def _split_training_values(
        self,
        values: Sequence[ImageInput],
        validation_images: Iterable[ImageInput] | None,
    ) -> tuple[list[ImageInput], list[ImageInput]]:
        all_values = list(values)
        if not all_values:
            raise ValueError("EfficientAD requires at least one normal training image.")
        if validation_images is not None:
            validation = list(validation_images)
            if not validation:
                raise ValueError("validation_images must not be empty.")
            return all_values, validation
        if len(all_values) < 2:
            if self.paper_strict:
                raise ValueError("paper_strict EfficientAD needs unseen validation images.")
            return all_values, all_values

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(len(all_values))
        validation_count = min(
            len(all_values) - 1,
            max(1, int(math.ceil(len(all_values) * self.validation_fraction))),
        )
        validation_indices = set(int(index) for index in indices[:validation_count])
        training = [
            value for index, value in enumerate(all_values) if index not in validation_indices
        ]
        validation = [
            value for index, value in enumerate(all_values) if index in validation_indices
        ]
        return training, validation

    def _prepare_penalty_loader(self) -> None:
        if self.imagenet_dir is None:
            self._penalty_loader = None
            self._penalty_iterator = None
            self.pretraining_penalty_applied_ = False
            return
        root = Path(self.imagenet_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"ImageNet penalty directory not found: {root}")
        penalty_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.RandomGrayscale(p=0.3),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )
        image_dataset = ImageFolder(root=str(root), transform=penalty_transform)
        self._penalty_loader = torch.utils.data.DataLoader(
            image_dataset, batch_size=1, shuffle=True
        )
        self._penalty_iterator = iter(self._penalty_loader)
        self.pretraining_penalty_applied_ = True

    def _next_penalty_images(self) -> torch.Tensor | None:
        if self._penalty_loader is None:
            return None
        try:
            batch = next(self._penalty_iterator)
        except StopIteration:
            self._penalty_iterator = iter(self._penalty_loader)
            batch = next(self._penalty_iterator)
        return batch[0].to(self.device)

    @torch.no_grad()
    def _calibrate_teacher(self, loader) -> None:  # noqa: ANN001
        if self.model is None:
            raise RuntimeError("EfficientAD model is not initialized.")
        self.model.teacher.eval()
        channel_sum = torch.zeros(self.teacher_out_channels, device=self.device)
        channel_square_sum = torch.zeros_like(channel_sum)
        count = 0
        for images, _targets in loader:
            output = self.model.teacher(images.to(self.device))
            channel_sum += output.sum(dim=(0, 2, 3))
            channel_square_sum += output.square().sum(dim=(0, 2, 3))
            count += int(output.shape[0] * output.shape[2] * output.shape[3])
        if count == 0:
            raise ValueError("Cannot calibrate EfficientAD teacher on an empty dataset.")
        mean = channel_sum / count
        variance = channel_square_sum / count - mean.square()
        self.model.teacher_mean.copy_(mean.view(1, -1, 1, 1))
        self.model.teacher_std.copy_(variance.clamp_min(0).sqrt().clamp_min(1e-6).view(1, -1, 1, 1))

    @torch.no_grad()
    def _calibrate_maps(self, loader) -> None:  # noqa: ANN001
        if self.model is None:
            raise RuntimeError("EfficientAD model is not initialized.")
        self.model.eval()
        maps_st: list[torch.Tensor] = []
        maps_ae: list[torch.Tensor] = []
        for images, _targets in loader:
            map_st, map_ae = self.model.raw_maps(images.to(self.device))
            maps_st.append(map_st.cpu().reshape(-1))
            maps_ae.append(map_ae.cpu().reshape(-1))
        if not maps_st:
            raise ValueError("Cannot calibrate EfficientAD maps on an empty dataset.")
        qa, qb = _PAPER_MAP_QUANTILES
        self.model.qa_st.copy_(torch.quantile(torch.cat(maps_st), qa).to(self.device))
        self.model.qb_st.copy_(torch.quantile(torch.cat(maps_st), qb).to(self.device))
        self.model.qa_ae.copy_(torch.quantile(torch.cat(maps_ae), qa).to(self.device))
        self.model.qb_ae.copy_(torch.quantile(torch.cat(maps_ae), qb).to(self.device))

    def training_forward(self, batch) -> float:  # noqa: ANN001
        if self.model is None or self.optimizer is None:
            raise RuntimeError("EfficientAD training is not initialized.")
        images, _targets = batch
        images = images.to(self.device)
        penalty_images = self._next_penalty_images()
        self.optimizer.zero_grad(set_to_none=True)
        loss_st, loss_ae, loss_stae = self.model.loss_terms(images, penalty_images)
        loss = loss_st + loss_ae + loss_stae
        loss.backward()
        self.optimizer.step()

        if self.training_steps_completed_ + 1 == int(0.95 * self.training_steps):
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr * 0.1
        self.last_loss_terms_ = {
            "student_teacher": float(loss_st.detach().item()),
            "autoencoder": float(loss_ae.detach().item()),
            "student_autoencoder": float(loss_stae.detach().item()),
        }
        return float(loss.detach().item())

    def evaluating_forward(self, batch) -> NDArray[np.float32]:  # noqa: ANN001
        if self.model is None:
            raise RuntimeError("EfficientAD model is not initialized.")
        images, _targets = batch
        anomaly_map = self.model.anomaly_map(images.to(self.device))
        return anomaly_map.amax(dim=(-2, -1)).reshape(-1).detach().cpu().numpy().astype(np.float32)

    @isolated_random_state_method
    def fit(
        self,
        x: object = MISSING,
        y=None,
        *,
        validation_images: Iterable[ImageInput] | None = None,
        **kwargs: object,
    ):
        values = list(resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        training_values, validation_values = self._split_training_values(values, validation_images)
        if self.paper_strict:
            missing = []
            if self.teacher_checkpoint is None:
                missing.append("teacher_checkpoint")
            if self.imagenet_dir is None:
                missing.append("imagenet_dir")
            if missing:
                raise ValueError("paper_strict EfficientAD requires " + " and ".join(missing) + ".")

        model = self.build_model()
        train_loader = self._loader(training_values, transform=self.train_transform, shuffle=True)
        statistics_loader = self._loader(
            training_values, transform=self.eval_transform, shuffle=False
        )
        validation_loader = self._loader(
            validation_values, transform=self.eval_transform, shuffle=False
        )
        self._calibrate_teacher(statistics_loader)
        self._prepare_penalty_loader()

        self.optimizer = torch.optim.Adam(
            chain(model.student.parameters(), model.autoencoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = None
        self.max_steps = self.training_steps
        self.epoch_num = int(math.ceil(self.training_steps / max(1, len(train_loader))))
        self.train(train_loader)
        self._calibrate_maps(validation_loader)
        self._penalty_iterator = None
        self._penalty_loader = None

        self.is_fitted_ = True
        self.decision_scores_ = self._decision_function(values)
        self._process_decision_scores()
        self._set_n_classes(y)
        return self

    @torch.no_grad()
    def _predict_maps(
        self, values: Sequence[ImageInput], *, batch_size: int | None = None
    ) -> NDArray[np.float32]:
        if self.model is None:
            raise RuntimeError("EfficientAD model is not initialized.")
        self.model.eval()
        loader = self._loader(
            values,
            transform=self.eval_transform,
            shuffle=False,
            batch_size=batch_size,
        )
        maps: list[np.ndarray] = []
        for images, _targets in loader:
            anomaly_map = self.model.anomaly_map(images.to(self.device))
            maps.append(anomaly_map[:, 0].cpu().numpy().astype(np.float32, copy=False))
        if not maps:
            return np.empty((0, *self.image_size), dtype=np.float32)
        return np.concatenate(maps, axis=0)

    def _decision_function(self, values: Sequence[ImageInput]) -> NDArray[np.float32]:
        maps = self._predict_maps(values)
        return maps.reshape(len(maps), -1).max(axis=1).astype(np.float32, copy=False)

    def decision_function(
        self, x: object = MISSING, batch_size: int | None = None, **kwargs: object
    ) -> NDArray[np.float32]:
        self._check_is_fitted()
        values = list(resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
        maps = self._predict_maps(values, batch_size=batch_size)
        return maps.reshape(len(maps), -1).max(axis=1).astype(np.float32, copy=False)

    def predict_anomaly_map(
        self, x: object = MISSING, batch_size: int | None = None, **kwargs: object
    ) -> NDArray[np.float32]:
        self._check_is_fitted()
        values = list(resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"))
        return self._predict_maps(values, batch_size=batch_size)

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.model is None or not hasattr(self, "threshold_"):
            raise RuntimeError("Fit EfficientAD before saving a checkpoint.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_state_dict": export_module_state_dict(self.model),
                "model_size": self.model_size,
                "teacher_out_channels": self.teacher_out_channels,
                "padding": self.padding,
                "threshold": float(self.threshold_),
                "decision_scores": np.asarray(self.decision_scores_, dtype=np.float64),
            },
            target,
        )
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported EfficientAD checkpoint schema.")
        if str(payload.get("model_size")) != self.model_size:
            raise ValueError("EfficientAD checkpoint model_size does not match detector.")
        if int(payload.get("teacher_out_channels", -1)) != self.teacher_out_channels:
            raise ValueError("EfficientAD checkpoint channel count does not match detector.")
        if bool(payload.get("padding")) != self.padding:
            raise ValueError("EfficientAD checkpoint padding does not match detector.")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("EfficientAD checkpoint is missing model_state_dict.")
        model = self._bind_model(
            EfficientADModel(
                model_size=self.model_size,
                teacher_out_channels=self.teacher_out_channels,
                padding=self.padding,
            )
        )
        model.load_state_dict(state, strict=True)
        self.teacher_checkpoint_loaded_ = True
        self.decision_scores_ = np.asarray(payload.get("decision_scores"), dtype=np.float64)
        self.threshold_ = float(payload["threshold"])
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        self.is_fitted_ = True
        self._set_n_classes(None)
        self.model.eval()

    def train_fast(self, train_folder: str, epochs: int = 10):
        paths = [
            str(path)
            for path in sorted(Path(train_folder).iterdir())
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]
        self.training_steps = max(1, int(epochs) * max(1, len(paths)))
        return self.fit(paths)

    def predict_fast(self, img_path: str):
        score = float(self.decision_function([str(img_path)])[0])
        return {"image": str(img_path), "anomaly_score": score}


__all__ = [
    "EfficientADAutoEncoder",
    "EfficientADDetector",
    "EfficientADModel",
    "MediumPatchDescriptionNetwork",
    "SmallPatchDescriptionNetwork",
]
