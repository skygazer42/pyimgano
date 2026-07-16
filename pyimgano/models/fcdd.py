"""FCDD adapted from the paper's 224 px MVTec-AD path."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision.transforms import functional as TVF

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ..base import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)

_PAPER_INPUT_SIZE = 224
_PAPER_RAW_SIZE = 240
_PAPER_RECEPTIVE_FIELD = 62
_PAPER_RECEPTIVE_STRIDE = 8


def _vgg11_bn_features() -> nn.Sequential:
    """Build the paper's truncated VGG11-BN path without its large classifier."""

    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 512, 3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
    )


class FCDDNetwork(nn.Module):
    """Paper MVTec network: truncated VGG11-BN followed by a 1x1 score head."""

    def __init__(self, *, pretrained: bool = False, freeze_features: bool = False) -> None:
        super().__init__()
        if pretrained:
            vgg, _transform = load_torchvision_model("vgg11_bn", pretrained=True)
            self.features = nn.Sequential(*list(vgg.features.children())[:21])
        else:
            self.features = _vgg11_bn_features()
        self.score_conv = nn.Conv2d(512, 1, 1, bias=True)

        if freeze_features:
            # This is the exact slice frozen by FCDD_CNN224_VGG_F in the author code.
            for parameter in self.features[:15].parameters():
                parameter.requires_grad_(False)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.score_conv(self.forward_features(images))


def _pseudo_huber(outputs: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(outputs.square() + 1.0) - 1.0


def _fcdd_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sample_scores = _pseudo_huber(outputs).flatten(1).mean(1)
    anomalous = -torch.log((-torch.expm1(-sample_scores)).clamp_min(1e-31))
    return torch.where(labels == 0, sample_scores, anomalous).mean()


def _gaussian_kernel(size: int, sigma: float, reference: torch.Tensor) -> torch.Tensor:
    # Match the author's even-kernel construction: duplicate the center sample,
    # then halve the 1-D kernel before taking its outer product.
    half = size // 2
    coords = torch.arange(size - 1, device=reference.device, dtype=reference.dtype) - (half - 1)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = torch.cat((kernel_1d[:half], kernel_1d[half - 1 :])) / 2.0
    return torch.outer(kernel_1d, kernel_1d)


@register_model(
    "vision_fcdd",
    tags=("vision", "deep", "fcdd", "one-class", "self-supervised", "pixel_map"),
    metadata={
        "description": "FCDD MVTec network, pseudo-Huber objective, and receptive-field heatmap",
        "paper": "Explainable Deep One-Class Classification",
        "paper_url": "https://openreview.net/forum?id=A5VV3UyIQz",
        "year": 2021,
        "supervision": "self-supervised",
        "implementation_status": "paper-mvtec-network-and-objective-industrial-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
class FCDD(BaseVisionDeepDetector):
    """Fully Convolutional Data Description for RGB industrial images.

    The network, loss, confetti parameters, optimizer, schedule, and Gaussian
    receptive-field upsampling follow the paper's MVTec-AD experiment. The
    offline default leaves ImageNet weights disabled; set ``pretrained=True``
    for the paper backbone. Dataset-specific normalization bounds are learned
    from the supplied normal images so the same API also works outside MVTec.
    """

    def __init__(
        self,
        *,
        pretrained: bool = False,
        freeze_features: Optional[bool] = None,
        synthetic_anomalies: bool = True,
        anomaly_probability: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_decay: float = 0.985,
        batch_size: int = 16,
        accumulate_batches: int = 8,
        epoch_size_multiplier: int = 10,
        epochs: int = 200,
        gaussian_sigma: float = 12.0,
        contamination: float = 0.1,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        if device is not None and str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        super().__init__(
            contamination=contamination,
            preprocessing=False,
            lr=learning_rate,
            epoch_num=epochs,
            batch_size=batch_size,
            optimizer_name="sgd",
            device=device,
            random_state=random_state,
            verbose=0,
        )
        if not 0.0 <= anomaly_probability <= 1.0:
            raise ValueError("anomaly_probability must be in [0, 1]")
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 < lr_decay <= 1.0:
            raise ValueError("lr_decay must be in (0, 1]")
        if batch_size <= 0 or accumulate_batches <= 0 or epoch_size_multiplier <= 0:
            raise ValueError("batch and epoch multipliers must be positive")
        if epochs < 0:
            raise ValueError("epochs must be non-negative")
        if gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be positive")

        self.pretrained = bool(pretrained)
        self.freeze_features = self.pretrained if freeze_features is None else bool(freeze_features)
        self.synthetic_anomalies = bool(synthetic_anomalies)
        self.anomaly_probability = float(anomaly_probability)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lr_decay = float(lr_decay)
        self.batch_size = int(batch_size)
        self.accumulate_batches = int(accumulate_batches)
        self.epoch_size_multiplier = int(epoch_size_multiplier)
        self.epochs = int(epochs)
        self.gaussian_sigma = float(gaussian_sigma)
        self.random_state = int(random_state)

        self.network_: Optional[FCDDNetwork] = None
        self.optimizer_: Optional[torch.optim.Optimizer] = None
        self.scheduler_: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.normalization_min_: Optional[torch.Tensor] = None
        self.normalization_scale_: Optional[torch.Tensor] = None
        self.history_: list[float] = []

    @staticmethod
    def _as_image_tensor(x: NDArray) -> torch.Tensor:
        array = np.asarray(x)
        if array.size == 0:
            raise ValueError("x must contain at least one image")
        if array.ndim == 3:
            array = array[None] if array.shape[-1] in {1, 3} else array[..., None]
        if array.ndim != 4:
            raise ValueError("x must have shape (N,H,W,C), (N,C,H,W), or (N,H,W)")
        if array.shape[-1] in {1, 3}:
            array = np.transpose(array, (0, 3, 1, 2))
        elif array.shape[1] not in {1, 3}:
            raise ValueError("FCDD expects one or three image channels")

        array = np.ascontiguousarray(array)
        tensor = torch.from_numpy(array).float()
        if not torch.isfinite(tensor).all():
            raise ValueError("x contains NaN or infinite values")
        minimum, maximum = float(tensor.min()), float(tensor.max())
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError("image values must be in [0, 1] or [0, 255]")
        if maximum > 1.0:
            tensor = tensor / 255.0
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        return tensor

    @staticmethod
    def _resize(images: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(images, size=(size, size), mode="nearest")

    @staticmethod
    def _local_contrast_normalize(images: torch.Tensor) -> torch.Tensor:
        centered = images - images.mean(dim=(1, 2, 3), keepdim=True)
        scale = centered.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return centered / scale

    def _fit_normalization(self, images: torch.Tensor) -> None:
        mins: list[torch.Tensor] = []
        maxs: list[torch.Tensor] = []
        for batch in images.split(max(1, self.batch_size * self.accumulate_batches)):
            normalized = self._local_contrast_normalize(self._resize(batch, _PAPER_INPUT_SIZE))
            mins.append(normalized.amin(dim=(0, 2, 3)))
            maxs.append(normalized.amax(dim=(0, 2, 3)))
        minimum = torch.stack(mins).amin(dim=0)
        maximum = torch.stack(maxs).amax(dim=0)
        self.normalization_min_ = minimum
        self.normalization_scale_ = (maximum - minimum).clamp_min(1e-12)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        if self.normalization_min_ is None or self.normalization_scale_ is None:
            raise RuntimeError("FCDD normalization has not been fitted")
        minimum = self.normalization_min_.to(images.device)[None, :, None, None]
        scale = self.normalization_scale_.to(images.device)[None, :, None, None]
        return (self._local_contrast_normalize(images) - minimum) / scale

    @staticmethod
    def _color_jitter(image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        limits = (
            (0.04, 0.04, 0.04, 0.04)
            if rng.random() < 0.5
            else (
                0.005,
                0.0005,
                0.0005,
                0.0005,
            )
        )
        brightness, contrast, saturation, hue = limits
        operations = [
            (TVF.adjust_brightness, rng.uniform(1.0 - brightness, 1.0 + brightness)),
            (TVF.adjust_contrast, rng.uniform(1.0 - contrast, 1.0 + contrast)),
            (TVF.adjust_saturation, rng.uniform(1.0 - saturation, 1.0 + saturation)),
            (TVF.adjust_hue, rng.uniform(-hue, hue)),
        ]
        rng.shuffle(operations)
        for operation, factor in operations:
            image = operation(image, float(factor))
        return image

    def _augment(self, images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        augmented = []
        for image in images:
            if rng.random() < 0.5:
                top = int(rng.integers(0, _PAPER_RAW_SIZE - _PAPER_INPUT_SIZE + 1))
                left = int(rng.integers(0, _PAPER_RAW_SIZE - _PAPER_INPUT_SIZE + 1))
                image = image[:, top : top + _PAPER_INPUT_SIZE, left : left + _PAPER_INPUT_SIZE]
            else:
                image = self._resize(image.unsqueeze(0), _PAPER_INPUT_SIZE).squeeze(0)
            image = self._color_jitter(image, rng)
            if rng.random() < 0.5:
                noise = torch.from_numpy(rng.standard_normal(image.shape).astype(np.float32))
                image = (image + noise * image.std() * 0.1).clamp(0.0, 1.0)
            augmented.append(image)
        return self._normalize(torch.stack(augmented))

    @staticmethod
    def _confetti_image(image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        pixels = np.ascontiguousarray(image.permute(1, 2, 0).numpy() * 255.0)
        height, width = pixels.shape[:2]
        noise = np.zeros(pixels.shape, dtype=np.float32)
        for probability, colored in ((0.000018, True), (0.000012, False)):
            blobs = max(1, int(rng.binomial(height * width, probability)))
            for _ in range(blobs):
                side = int(rng.integers(8, 55))
                center = (float(rng.integers(0, width)), float(rng.integers(0, height)))
                angle = float(rng.uniform(-45.0, 45.0))
                points = cv2.boxPoints((center, (float(side), float(side)), angle)).astype(np.int32)
                color = (
                    tuple(float(value) for value in rng.integers(-256, 0, size=3))
                    if colored
                    else (-255.0, -255.0, -255.0)
                )
                cv2.fillConvexPoly(noise, points, color)
        noise = cv2.GaussianBlur(noise, (25, 25), sigmaX=5.0)
        anomalous = np.clip(pixels + noise, 0.0, 255.0)
        difference = abs(float(np.mean(anomalous - pixels)))
        inverted = np.clip(pixels - noise, 0.0, 255.0)
        if difference < 0.025 and abs(float(np.mean(inverted - pixels))) > difference:
            anomalous = inverted
        return torch.from_numpy(anomalous).permute(2, 0, 1).float() / 255.0

    def _inject_confetti(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.synthetic_anomalies:
            return images, labels
        selected = (labels == 0).numpy() & (rng.random(len(labels)) < self.anomaly_probability)
        if not selected.any():
            return images, labels
        images = images.clone()
        labels = labels.clone()
        for index in np.flatnonzero(selected):
            images[index] = self._confetti_image(images[index], rng)
            labels[index] = 1
        return images, labels

    def _build_network(self) -> FCDDNetwork:
        self.network_ = FCDDNetwork(
            pretrained=self.pretrained,
            freeze_features=self.freeze_features,
        ).to(self.device)
        return self.network_

    def fit(self, x: NDArray, y: Optional[NDArray] = None) -> "FCDD":
        images = self._as_image_tensor(x)
        labels = torch.zeros(len(images), dtype=torch.long)
        if y is not None:
            labels_array = np.asarray(y).reshape(-1)
            if len(labels_array) != len(images) or not np.isin(labels_array, (0, 1)).all():
                raise ValueError("y must contain one binary label per image")
            labels = torch.from_numpy(labels_array.astype(np.int64, copy=False))

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        raw_images = self._resize(images, _PAPER_RAW_SIZE)
        self._fit_normalization(images)
        network = self._build_network()
        self.optimizer_ = torch.optim.SGD(
            network.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=self.weight_decay,
        )
        self.scheduler_ = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_,
            lr_lambda=lambda epoch: self.lr_decay**epoch,
        )
        base_dataset = TensorDataset(raw_images, labels)
        dataset = ConcatDataset([base_dataset] * self.epoch_size_multiplier)
        generator = torch.Generator().manual_seed(self.random_state)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.accumulate_batches,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        self.history_ = []
        network.train()
        for epoch in range(self.epochs):
            losses = []
            for batch_images, batch_labels in dataloader:
                batch_images, batch_labels = self._inject_confetti(batch_images, batch_labels, rng)
                batch_images = self._augment(batch_images, rng).to(self.device)
                batch_labels = batch_labels.to(self.device)
                outputs = network(batch_images)
                loss = _fcdd_loss(outputs, batch_labels)
                self.optimizer_.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer_.step()
                losses.append(float(loss.detach()))
            self.scheduler_.step()
            mean_loss = float(np.mean(losses)) if losses else 0.0
            self.history_.append(mean_loss)
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, self.epochs, mean_loss)

        self.is_fitted_ = True
        self.decision_scores_ = self.decision_function(x)
        self._process_decision_scores()
        self._set_n_classes(None)
        return self

    @staticmethod
    def receptive_upsample(score_map: torch.Tensor, sigma: float = 12.0) -> torch.Tensor:
        if score_map.shape[-2:] != (28, 28):
            raise ValueError("the paper MVTec network must produce a 28x28 score map")
        kernel = _gaussian_kernel(_PAPER_RECEPTIVE_FIELD, float(sigma), score_map)
        expanded = F.conv_transpose2d(
            score_map,
            kernel[None, None],
            stride=_PAPER_RECEPTIVE_STRIDE,
            output_padding=6,
        )
        return expanded[:, :, 27:-33, 27:-33]

    def _predict(self, x: NDArray, *, include_maps: bool) -> tuple[NDArray, Optional[NDArray]]:
        self._check_is_fitted()
        if self.network_ is None:
            raise RuntimeError("FCDD network is unavailable")
        images = self._as_image_tensor(x)
        original_size = images.shape[-2:]
        scores = []
        maps = []
        self.network_.eval()
        with torch.no_grad():
            for batch in images.split(self.batch_size):
                batch = self._normalize(self._resize(batch, _PAPER_INPUT_SIZE)).to(self.device)
                score_map = _pseudo_huber(self.network_(batch))
                scores.append(score_map.flatten(1).mean(1).cpu().numpy())
                if include_maps:
                    full_map = self.receptive_upsample(score_map, self.gaussian_sigma)
                    if original_size != (_PAPER_INPUT_SIZE, _PAPER_INPUT_SIZE):
                        full_map = F.interpolate(full_map, size=original_size, mode="nearest")
                    maps.append(full_map.squeeze(1).cpu().numpy())
        score_array = np.concatenate(scores)
        map_array = np.concatenate(maps) if include_maps else None
        return score_array, map_array

    def decision_function(self, x: NDArray) -> NDArray:
        return self._predict(x, include_maps=False)[0]

    def predict_with_map(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        scores, maps = self._predict(x, include_maps=True)
        assert maps is not None
        return scores, maps

    def predict_anomaly_map(self, x: NDArray) -> NDArray:
        return self.predict_with_map(x)[1]

    def get_params(self) -> dict:
        return {
            "pretrained": self.pretrained,
            "freeze_features": self.freeze_features,
            "synthetic_anomalies": self.synthetic_anomalies,
            "anomaly_probability": self.anomaly_probability,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_decay": self.lr_decay,
            "batch_size": self.batch_size,
            "accumulate_batches": self.accumulate_batches,
            "epoch_size_multiplier": self.epoch_size_multiplier,
            "epochs": self.epochs,
            "gaussian_sigma": self.gaussian_sigma,
            "contamination": self.contamination,
            "device": str(self.device),
            "random_state": self.random_state,
        }


__all__ = ["FCDD", "FCDDNetwork"]
