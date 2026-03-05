# -*- coding: utf-8 -*-
"""Autoencoder-based anomaly detector (modernized).

This module provides a lightweight, contract-aligned reconstruction baseline:
- registry name: `ae_resnet_unet` (kept for backward compatibility)
- base contract: `BaseVisionDeepDetector` / `BaseDeepLearningDetector`

Design constraints:
- No implicit weight downloads by default.
- Keep module import lightweight (avoid importing matplotlib/cv2 at import time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


def _to_hw(image_size: int | Tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, tuple):
        h, w = int(image_size[0]), int(image_size[1])
    else:
        h = w = int(image_size)
    if h <= 0 or w <= 0:
        raise ValueError(f"image_size must be positive, got {image_size!r}")
    return h, w


@dataclass(frozen=True)
class AEConfig:
    image_size: tuple[int, int]
    latent_channels: int
    base_channels: int


def _make_transforms(image_size: tuple[int, int]):
    # Import torchvision lazily: keep `import pyimgano.models` light.
    from torchvision import transforms

    # Use deterministic transforms for unit tests + stable behavior.
    # Note: transforms.Normalize expects float tensors.
    return transforms.Compose(
        [
            transforms.Resize((int(image_size[0]), int(image_size[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _build_autoencoder(*, cfg: AEConfig):
    import torch
    import torch.nn as nn

    c = int(cfg.base_channels)
    z = int(cfg.latent_channels)

    # Tiny convolutional AE (fast; good enough for smoke tests and baselines).
    encoder = nn.Sequential(
        nn.Conv2d(3, c, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c * 2, z, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(z, c * 2, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c, 3, kernel_size=4, stride=2, padding=1),
    )

    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            zt = self.encoder(x)
            out = self.decoder(zt)
            # Note: output is in normalized space; caller compares in same space.
            return out

    return _AE()


@register_model(
    "ae_resnet_unet",
    tags=("vision", "deep", "autoencoder", "reconstruction"),
    metadata={
        "description": "Reconstruction baseline (contract-aligned autoencoder; legacy name kept)",
    },
    overwrite=True,
)
class OptimizedAEDetector(BaseVisionDeepDetector):
    """Contract-aligned reconstruction anomaly detector.

    Parameters
    ----------
    tiny:
        When true, uses a smaller network and fewer defaults suitable for unit tests.
    image_size:
        Resize inputs to this spatial size (int or (H,W)).
    """

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str | None = None,
        random_state: int = 0,
        verbose: int = 0,
        tiny: bool = False,
        image_size: int | Tuple[int, int] = 256,
        latent_channels: int = 64,
        base_channels: int = 32,
    ) -> None:
        self.tiny = bool(tiny)
        hw = _to_hw(image_size)

        if self.tiny:
            latent_channels = min(int(latent_channels), 32)
            base_channels = min(int(base_channels), 16)

        self.cfg = AEConfig(
            image_size=hw,
            latent_channels=int(latent_channels),
            base_channels=int(base_channels),
        )

        train_transform = _make_transforms(hw)
        eval_transform = _make_transforms(hw)

        super().__init__(
            contamination=float(contamination),
            preprocessing=True,
            lr=float(lr),
            epoch_num=int(epochs),
            batch_size=int(batch_size),
            optimizer_name="adam",
            criterion_name="mse",
            device=device,
            random_state=int(random_state),
            verbose=int(verbose),
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    def build_model(self):
        # Called inside BaseDeepLearningDetector.fit().
        return _build_autoencoder(cfg=self.cfg)

    def training_forward(self, batch) -> float:  # noqa: ANN001
        import torch

        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)

        recon = self.model(images)  # type: ignore[operator]
        loss = self.criterion(recon, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def evaluating_forward(self, batch):  # noqa: ANN001
        import torch

        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            recon = self.model(images)  # type: ignore[operator]
            err = (recon - targets) ** 2
            # Mean squared error per sample.
            score = err.flatten(1).mean(dim=1)
            return score.detach().cpu().numpy()

    @staticmethod
    def _to_single_path(x):  # noqa: ANN001, ANN201 - utility
        return x[0] if isinstance(x, (list, tuple)) and x else x

    def get_anomaly_map(
        self, x, *, image_size: tuple[int, int] | None = None
    ):  # noqa: ANN001, ANN201
        """Best-effort pixel-level reconstruction error map (H,W)."""

        import torch
        from torch.utils.data import DataLoader

        from pyimgano.datasets import VisionArrayDataset, VisionImageDataset

        if image_size is None:
            image_size = self.cfg.image_size

        item = self._to_single_path(x)
        if isinstance(item, np.ndarray):
            ds = VisionArrayDataset([item], transform=self.eval_transform)
        else:
            ds = VisionImageDataset([str(item)], transform=self.eval_transform)
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        self.evaluating_prepare()
        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            with torch.no_grad():
                recon = self.model(images)  # type: ignore[operator]
                err = (recon - targets).abs().mean(dim=1)  # (B,H,W)
                m = err[0].detach().cpu().numpy().astype(np.float32)
                return m

        return np.zeros(image_size, dtype=np.float32)
