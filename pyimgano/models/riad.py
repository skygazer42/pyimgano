"""RIAD reconstruction-by-inpainting adaptation for industrial images.

Reference: "Reconstruction by inpainting for visual anomaly detection",
Pattern Recognition 112 (2021), DOI 10.1016/j.patcog.2020.107706.

The implementation follows the paper's MVTec RGB path: disjoint ``k x k``
region masks, a five-level U-Net, assembled partial inpaintings, the combined
L2/SSIM/MSGMS objective, and multi-region-size MSGMS anomaly maps.  The
authors did not publish reference code, so this is a paper adaptation rather
than a claim of source-identical reproduction.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, TensorDataset

from pyimgano.utils.random_state import isolated_random_state, isolated_random_state_method

from ._image_batch import coerce_rgb_image_batch
from .baseCv import BaseVisionDeepDetector
from .draem import _ssim_loss
from .registry import register_model

logger = logging.getLogger(__name__)


class ImageDecomposer:
    """Create RIAD's random, disjoint sets of square image regions."""

    def __init__(
        self,
        num_disjoint_masks: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        if num_disjoint_masks <= 0:
            raise ValueError("num_disjoint_masks must be positive.")
        self.num_disjoint_masks = int(num_disjoint_masks)
        self.rng = np.random.default_rng(random_state)

    def create_disjoint_masks(
        self,
        image_size: Tuple[int, int],
        region_size: int,
    ) -> NDArray:
        """Return keep-masks whose removed regions partition the whole image."""

        height, width = (int(image_size[0]), int(image_size[1]))
        region_size = int(region_size)
        if height <= 0 or width <= 0 or region_size <= 0:
            raise ValueError("image dimensions and region_size must be positive.")

        grid_height = math.ceil(height / region_size)
        grid_width = math.ceil(width / region_size)
        region_ids = self.rng.permutation(grid_height * grid_width)
        masks = []
        for subset in np.array_split(region_ids, self.num_disjoint_masks):
            grid = np.ones(grid_height * grid_width, dtype=np.float32)
            grid[subset] = 0.0
            mask = grid.reshape(grid_height, grid_width)
            mask = np.repeat(np.repeat(mask, region_size, axis=0), region_size, axis=1)
            masks.append(mask[:height, :width])
        return np.stack(masks, axis=0)[:, np.newaxis]


class UNetDownBlock(nn.Module):
    """Two convolution-BatchNorm-ReLU stages from the RIAD encoder."""

    def __init__(self, in_channels: int, out_channels: int, *, downsample: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4 if downsample else 3,
            stride=2 if downsample else 1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(x)), inplace=True)


class UNetUpBlock(nn.Module):
    """RIAD decoder stage with transposed convolution and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.up(x), skip), dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(x)), inplace=True)


class UNet(nn.Module):
    """Five-level U-Net shown in the RIAD paper's network diagram."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        channels = (64, 128, 256, 512, 512)
        self.down_blocks = nn.ModuleList(
            [
                UNetDownBlock(in_channels, channels[0], downsample=False),
                *(
                    UNetDownBlock(previous, current, downsample=True)
                    for previous, current in zip(channels, channels[1:])
                ),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UNetUpBlock(512, 512),
                UNetUpBlock(512, 256),
                UNetUpBlock(256, 128),
                UNetUpBlock(128, 64),
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            x = block(x, skip)
        return self.output(x)


class MSGMSLoss(nn.Module):
    """Multi-scale gradient-magnitude-similarity loss and anomaly map."""

    def __init__(self, num_scales: int = 4, stability: float = 0.0026) -> None:
        super().__init__()
        if num_scales <= 0 or stability <= 0:
            raise ValueError("num_scales and stability must be positive.")
        self.num_scales = int(num_scales)
        self.stability = float(stability)
        self.register_buffer(
            "prewitt_x",
            torch.tensor([[[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]]) / 3.0,
        )
        self.register_buffer(
            "prewitt_y",
            torch.tensor([[[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]]]) / 3.0,
        )

    def _gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        grayscale = image.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(grayscale, self.prewitt_x, padding=1)
        grad_y = F.conv2d(grayscale, self.prewitt_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)

    def _gms(self, image: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        source_gradient = self._gradient_magnitude(image)
        reconstruction_gradient = self._gradient_magnitude(reconstruction)
        numerator = 2.0 * source_gradient * reconstruction_gradient + self.stability
        denominator = source_gradient.square() + reconstruction_gradient.square() + self.stability
        return numerator / denominator

    def forward(
        self,
        image: torch.Tensor,
        reconstruction: torch.Tensor,
        *,
        as_loss: bool = True,
    ) -> torch.Tensor:
        if image.shape != reconstruction.shape or image.ndim != 4:
            raise ValueError("MSGMS inputs must have matching NCHW shapes.")

        output_size = image.shape[-2:]
        distance = torch.zeros(
            (image.shape[0], 1, *output_size),
            dtype=image.dtype,
            device=image.device,
        )
        source, restored = image, reconstruction
        for scale in range(self.num_scales):
            if scale:
                source = F.avg_pool2d(source, kernel_size=2, stride=2)
                restored = F.avg_pool2d(restored, kernel_size=2, stride=2)
            scale_distance = 1.0 - self._gms(source, restored)
            if scale_distance.shape[-2:] != output_size:
                scale_distance = F.interpolate(
                    scale_distance,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            distance = distance + scale_distance
        distance = distance / self.num_scales
        return distance.mean() if as_loss else distance


@register_model(
    "vision_riad",
    tags=("vision", "deep", "riad", "reconstruction", "self-supervised", "pixel_map"),
    metadata={
        "description": "RIAD disjoint masked-inpainting MVTec RGB adaptation with MSGMS scoring",
        "paper": "Reconstruction by inpainting for visual anomaly detection",
        "paper_url": "https://doi.org/10.1016/j.patcog.2020.107706",
        "year": 2021,
        "supervision": "self-supervised",
        "implementation_status": "paper-network-mask-loss-and-score-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
@register_model(
    "riad",
    tags=("vision", "deep", "riad", "reconstruction", "self-supervised", "pixel_map"),
    metadata={
        "description": "Legacy alias for the RIAD disjoint masked-inpainting adaptation",
        "paper": "Reconstruction by inpainting for visual anomaly detection",
        "paper_url": "https://doi.org/10.1016/j.patcog.2020.107706",
        "year": 2021,
        "supervision": "self-supervised",
        "implementation_status": "paper-network-mask-loss-and-score-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
class RIADDetector(BaseVisionDeepDetector):
    """Paper-adapted RIAD detector for the MVTec RGB protocol.

    Defaults follow the published setup: 256-pixel RGB inputs, region sizes
    ``(2, 4, 8, 16)``, three disjoint masks, four MSGMS scales, 300 epochs,
    batch size 4, and Adam with learning rate ``1e-4`` and weight decay
    ``1e-5``.  The learning rate is reduced by ten at epoch 250.
    """

    def __init__(
        self,
        region_sizes: Sequence[int] = (2, 4, 8, 16),
        num_disjoint_masks: int = 3,
        image_size: Tuple[int, int] = (256, 256),
        epochs: int = 300,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        msgms_scales: int = 4,
        gaussian_sigma: float = 7.0,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        legacy = {"n_splits", "mask_ratio"}.intersection(kwargs)
        if legacy:
            raise TypeError(
                "RIAD's legacy random-grid proxy parameters were removed; use "
                "region_sizes and num_disjoint_masks."
            )
        super().__init__(**kwargs)

        self.region_sizes = tuple(int(value) for value in region_sizes)
        self.num_disjoint_masks = int(num_disjoint_masks)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.msgms_scales = int(msgms_scales)
        self.gaussian_sigma = float(gaussian_sigma)
        self.random_state = random_state

        if not self.region_sizes or any(value <= 0 for value in self.region_sizes):
            raise ValueError("region_sizes must contain positive integers.")
        if self.num_disjoint_masks <= 0:
            raise ValueError("num_disjoint_masks must be positive.")
        if any(value <= 0 or value % 16 for value in self.image_size):
            raise ValueError("image_size dimensions must be positive multiples of 16.")
        if max(self.region_sizes) > min(self.image_size):
            raise ValueError("region_sizes cannot exceed the smaller image dimension.")
        if self.msgms_scales <= 0 or min(self.image_size) < 2 ** (self.msgms_scales - 1):
            raise ValueError("image_size is too small for msgms_scales.")
        if self.epochs < 0 or self.batch_size <= 0:
            raise ValueError("epochs must be non-negative and batch_size positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gaussian_sigma < 0:
            raise ValueError("learning_rate must be positive; weight_decay/sigma non-negative.")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        with isolated_random_state(random_state):
            self.model = UNet().to(self.device)
        self.decomposer = ImageDecomposer(self.num_disjoint_masks, random_state=random_state)
        self.msgms = MSGMSLoss(self.msgms_scales).to(self.device)
        self.loss_history_: list[float] = []

    def _prepare_images(self, x: object) -> Tuple[torch.Tensor, list[Tuple[int, int]]]:
        import cv2

        images = coerce_rgb_image_batch(x).astype(np.float32, copy=False)
        if not np.all(np.isfinite(images)):
            raise ValueError("RIAD images must contain only finite values.")
        minimum, maximum = float(images.min()), float(images.max())
        if maximum > 1.0:
            if minimum < 0.0 or maximum > 255.0:
                raise ValueError("RIAD image values must lie in [0, 1], [0, 255], or [-1, 1].")
            images = images / 255.0
        elif minimum < 0.0:
            if minimum < -1.0:
                raise ValueError("RIAD image values must lie in [0, 1], [0, 255], or [-1, 1].")
            images = (images + 1.0) / 2.0

        original_sizes = [tuple(int(value) for value in image.shape[:2]) for image in images]
        resized = [
            (
                cv2.resize(
                    image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA
                )
                if image.shape[:2] != self.image_size
                else image
            )
            for image in images
        ]
        tensor = torch.from_numpy(np.ascontiguousarray(np.stack(resized)))
        tensor = tensor.permute(0, 3, 1, 2).float().mul(2.0).sub(1.0)
        return tensor, original_sizes

    def _reconstruct(self, images: torch.Tensor, region_size: int) -> torch.Tensor:
        masks = self.decomposer.create_disjoint_masks(images.shape[-2:], region_size)
        keep_masks = torch.from_numpy(masks).to(device=images.device, dtype=images.dtype)
        reconstruction = torch.zeros_like(images)
        for keep_mask in keep_masks:
            prediction = self.model(images * keep_mask)
            reconstruction = reconstruction + prediction * (1.0 - keep_mask)
        return reconstruction

    @isolated_random_state_method
    def fit(self, x: object, y: Optional[NDArray] = None, **kwargs: object) -> "RIADDetector":
        """Train on anomaly-free RGB images."""

        del y, kwargs
        images, _ = self._prepare_images(x)
        loader = DataLoader(
            TensorDataset(images),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.loss_history_ = []
        self.model.train()
        for epoch in range(self.epochs):
            if epoch == 249:
                for group in optimizer.param_groups:
                    group["lr"] = self.learning_rate * 0.1

            total_loss = 0.0
            for (target,) in loader:
                target = target.to(self.device)
                region_size = int(self.decomposer.rng.choice(self.region_sizes))
                reconstruction = self._reconstruct(target, region_size)
                loss = (
                    F.mse_loss(reconstruction, target)
                    + _ssim_loss(reconstruction, target)
                    + self.msgms(target, reconstruction)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())

            average_loss = total_loss / len(loader)
            self.loss_history_.append(average_loss)
            if (epoch + 1) % 10 == 0 or epoch + 1 == self.epochs:
                logger.info("RIAD epoch %d/%d loss %.6f", epoch + 1, self.epochs, average_loss)

        self.model.eval()
        return self

    def _predict_model_map(self, image: torch.Tensor) -> torch.Tensor:
        anomaly_map = torch.zeros(
            (image.shape[0], 1, *self.image_size),
            dtype=image.dtype,
            device=image.device,
        )
        for region_size in self.region_sizes:
            reconstruction = self._reconstruct(image, region_size)
            anomaly_map = anomaly_map + self.msgms(image, reconstruction, as_loss=False)
        return anomaly_map / len(self.region_sizes)

    def predict_anomaly_map(self, x: object) -> list[NDArray]:
        """Return Gaussian-smoothed MSGMS maps at each input image's original size."""

        import cv2

        images, original_sizes = self._prepare_images(x)
        self.model.eval()
        output = []
        with torch.no_grad():
            for image, original_size in zip(images, original_sizes):
                anomaly_map = self._predict_model_map(image.unsqueeze(0).to(self.device))
                array = anomaly_map[0, 0].cpu().numpy()
                if self.gaussian_sigma:
                    array = gaussian_filter(array, sigma=self.gaussian_sigma)
                if original_size != self.image_size:
                    array = cv2.resize(
                        array,
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                output.append(np.asarray(array, dtype=np.float32))
        return output

    def predict_proba(self, x: object, **kwargs: object) -> NDArray:
        """Return the paper's poorest-region (maximum-map) image scores."""

        del kwargs
        return np.asarray([float(amap.max()) for amap in self.predict_anomaly_map(x)])

    def decision_function(
        self,
        x: object,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        del batch_size
        return np.asarray(self.predict_proba(x, **kwargs), dtype=np.float64).reshape(-1)


__all__ = ["ImageDecomposer", "MSGMSLoss", "RIADDetector", "UNet"]
