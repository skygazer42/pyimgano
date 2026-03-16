# -*- coding: utf-8 -*-
"""Variational Autoencoder (VAE) anomaly detector (modernized).

Registry name: `vae_conv` (kept stable for backward compatibility).

This implementation is contract-aligned with `BaseVisionDeepDetector`:
- `fit(X)` accepts image paths or numpy images (via BaseVisionDeepDetector)
- `decision_function(X)` returns one score per sample (higher = more anomalous)

Design constraints:
- Keep import-time lightweight (no matplotlib/cv2 at module import time).
- No implicit network weight downloads by default.
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
class VAEConfig:
    image_size: tuple[int, int]
    latent_dim: int
    base_channels: int
    beta_kl: float


def _make_transforms(image_size: tuple[int, int]):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((int(image_size[0]), int(image_size[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _build_conv_vae(*, cfg: VAEConfig):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    c = int(cfg.base_channels)
    zdim = int(cfg.latent_dim)

    # Encoder downsamples 3 times -> spatial /8
    enc = nn.Sequential(
        nn.Conv2d(3, c, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
    )

    # Compute flattened dim from config image_size
    h, w = int(cfg.image_size[0]), int(cfg.image_size[1])
    fh, fw = max(1, h // 8), max(1, w // 8)
    flat_dim = int(c * 4 * fh * fw)

    fc_mu = nn.Linear(flat_dim, zdim)
    fc_logvar = nn.Linear(flat_dim, zdim)
    fc_dec = nn.Linear(zdim, flat_dim)

    dec = nn.Sequential(
        nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1),
    )

    class _VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.fc_mu = fc_mu
            self.fc_logvar = fc_logvar
            self.fc_dec = fc_dec
            self.dec = dec
            self.fh = fh
            self.fw = fw
            self.c4 = c * 4

        def encode(self, x: torch.Tensor):
            feat = self.enc(x)
            flat = feat.flatten(1)
            mu = self.fc_mu(flat)
            logvar = self.fc_logvar(flat)
            return mu, logvar

        def reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor):
            flat = self.fc_dec(z)
            feat = flat.view(flat.size(0), self.c4, self.fh, self.fw)
            out = self.dec(feat)
            return out

        def forward(self, x: torch.Tensor):
            mu, logvar = self.encode(x)
            z = self.reparam(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

        @staticmethod
        def loss(
            recon: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            *,
            beta: float,
        ):
            # Reconstruction in normalized space.
            rec = F.mse_loss(recon, x, reduction="none").flatten(1).mean(dim=1)
            kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl = kl / float(mu.shape[1])  # normalize by latent dim
            loss = rec + float(beta) * kl
            return loss, rec, kl

    return _VAE()


@register_model(
    "vae_conv",
    tags=("vision", "deep", "vae", "reconstruction"),
    metadata={"description": "Convolutional VAE reconstruction baseline (contract-aligned)"},
    overwrite=True,
)
class VAEAnomalyDetector(BaseVisionDeepDetector):
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
        latent_dim: int = 64,
        base_channels: int = 32,
        beta_kl: float = 1e-3,
    ) -> None:
        self.tiny = bool(tiny)
        hw = _to_hw(image_size)

        if self.tiny:
            latent_dim = min(int(latent_dim), 32)
            base_channels = min(int(base_channels), 16)

        self.cfg = VAEConfig(
            image_size=hw,
            latent_dim=int(latent_dim),
            base_channels=int(base_channels),
            beta_kl=float(beta_kl),
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
        return _build_conv_vae(cfg=self.cfg)

    def training_forward(self, batch) -> float:  # noqa: ANN001

        images, _targets = batch
        images = images.to(self.device)

        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)

        recon, mu, logvar = self.model(images)  # type: ignore[operator]
        loss_vec, _rec, _kl = self.model.loss(recon, images, mu, logvar, beta=float(self.cfg.beta_kl))  # type: ignore[attr-defined]
        loss = loss_vec.mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def evaluating_forward(self, batch):  # noqa: ANN001
        import torch

        images, _targets = batch
        images = images.to(self.device)

        with torch.no_grad():
            recon, mu, logvar = self.model(images)  # type: ignore[operator]
            _, rec, _ = self.model.loss(recon, images, mu, logvar, beta=float(self.cfg.beta_kl))  # type: ignore[attr-defined]
            # Use reconstruction error only as the anomaly score (more interpretable).
            score = rec
            return score.detach().cpu().numpy()

    def get_anomaly_map(
        self, x, *, image_size: tuple[int, int] | None = None
    ):  # noqa: ANN001, ANN201
        """Best-effort pixel-level reconstruction error map (H,W)."""

        import torch
        from torch.utils.data import DataLoader

        from pyimgano.datasets import VisionArrayDataset, VisionImageDataset

        if image_size is None:
            image_size = self.cfg.image_size

        item = x[0] if isinstance(x, (list, tuple)) and x else x
        if isinstance(item, np.ndarray):
            ds = VisionArrayDataset([item], transform=self.eval_transform)
        else:
            ds = VisionImageDataset([str(item)], transform=self.eval_transform)
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        self.evaluating_prepare()
        for images, _targets in loader:
            images = images.to(self.device)
            with torch.no_grad():
                recon, _mu, _logvar = self.model(images)  # type: ignore[operator]
                err = (recon - images).abs().mean(dim=1)
                return err[0].detach().cpu().numpy().astype(np.float32)

        return np.zeros(image_size, dtype=np.float32)
