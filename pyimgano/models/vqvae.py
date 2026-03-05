# -*- coding: utf-8 -*-
"""Lightweight VQ-VAE reconstruction anomaly detector (industrial tiny variant).

Registry name: `vqvae_conv`.

Design goals:
- contract-aligned with `BaseVisionDeepDetector`
- small, dependency-stable implementation (torch/torchvision only)
- no implicit downloads by default
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
class VQVAEConfig:
    image_size: tuple[int, int]
    codebook_size: int
    embedding_dim: int
    base_channels: int
    beta_commit: float


def _make_transforms(image_size: tuple[int, int]):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((int(image_size[0]), int(image_size[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _build_conv_vqvae(*, cfg: VQVAEConfig):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    c = int(cfg.base_channels)
    k = int(cfg.codebook_size)
    d = int(cfg.embedding_dim)

    # Encoder downsamples 3 times -> spatial /8
    enc = nn.Sequential(
        nn.Conv2d(3, c, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
    )
    proj = nn.Conv2d(c * 4, d, kernel_size=1)

    codebook = nn.Embedding(k, d)
    nn.init.uniform_(codebook.weight, -1.0 / k, 1.0 / k)

    dec_in = nn.Conv2d(d, c * 4, kernel_size=1)
    dec = nn.Sequential(
        nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1),
    )

    class _VQVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.proj = proj
            self.codebook = codebook
            self.dec_in = dec_in
            self.dec = dec

        def _quantize(self, z_e: torch.Tensor, *, beta: float):
            # z_e: (N,D,H,W) -> (N*H*W, D)
            z = z_e.permute(0, 2, 3, 1).contiguous()
            flat = z.view(-1, z.shape[-1])

            # Compute nearest codebook entry (L2).
            # dist = ||x||^2 + ||e||^2 - 2 x.e
            flat_sq = (flat * flat).sum(dim=1, keepdim=True)
            emb = self.codebook.weight
            emb_sq = (emb * emb).sum(dim=1).view(1, -1)
            dist = flat_sq + emb_sq - 2.0 * (flat @ emb.t())
            indices = torch.argmin(dist, dim=1)

            z_q = self.codebook(indices).view(z.shape)

            # VQ losses (per-element mean).
            # - codebook: move embeddings towards encoder outputs
            # - commitment: keep encoder outputs close to embeddings
            codebook_loss = F.mse_loss(z_q, z.detach(), reduction="mean")
            commit_loss = F.mse_loss(z_q.detach(), z, reduction="mean")
            vq_loss = codebook_loss + float(beta) * commit_loss

            # Straight-through estimator.
            z_q_st = z + (z_q - z).detach()
            z_q_st = z_q_st.permute(0, 3, 1, 2).contiguous()
            return z_q_st, vq_loss

        def forward(self, x: torch.Tensor, *, beta: float):
            feat = self.enc(x)
            z_e = self.proj(feat)
            z_q, vq_loss = self._quantize(z_e, beta=float(beta))
            recon = self.dec(self.dec_in(z_q))
            return recon, vq_loss

        @staticmethod
        def recon_error(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # Reconstruction error in normalized space.
            rec = F.mse_loss(recon, x, reduction="none")
            return rec.flatten(1).mean(dim=1)

    return _VQVAE()


@register_model(
    "vqvae_conv",
    tags=("vision", "deep", "vqvae", "reconstruction"),
    metadata={"description": "Convolutional VQ-VAE reconstruction baseline (tiny-capable)"},
)
class VQVAEAnomalyDetector(BaseVisionDeepDetector):
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
        image_size: int | Tuple[int, int] = 128,
        codebook_size: int = 128,
        embedding_dim: int = 64,
        base_channels: int = 32,
        beta_commit: float = 0.25,
    ) -> None:
        self.tiny = bool(tiny)
        hw = _to_hw(image_size)

        if self.tiny:
            codebook_size = min(int(codebook_size), 64)
            embedding_dim = min(int(embedding_dim), 32)
            base_channels = min(int(base_channels), 16)

        self.cfg = VQVAEConfig(
            image_size=hw,
            codebook_size=int(codebook_size),
            embedding_dim=int(embedding_dim),
            base_channels=int(base_channels),
            beta_commit=float(beta_commit),
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
        return _build_conv_vqvae(cfg=self.cfg)

    def training_forward(self, batch) -> float:  # noqa: ANN001
        import torch
        import torch.nn.functional as F

        images, _targets = batch
        images = images.to(self.device)

        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)

        recon, vq_loss = self.model(images, beta=float(self.cfg.beta_commit))  # type: ignore[operator]
        rec_vec = self.model.recon_error(recon, images)  # type: ignore[attr-defined]
        rec_loss = rec_vec.mean()
        loss = rec_loss + vq_loss
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def evaluating_forward(self, batch):  # noqa: ANN001
        import torch

        images, _targets = batch
        images = images.to(self.device)

        with torch.no_grad():
            recon, _vq_loss = self.model(images, beta=float(self.cfg.beta_commit))  # type: ignore[operator]
            rec_vec = self.model.recon_error(recon, images)  # type: ignore[attr-defined]
            # Use reconstruction error only as the anomaly score (more interpretable).
            return rec_vec.detach().cpu().numpy()

    def get_anomaly_map(
        self, x, *, image_size: tuple[int, int] | None = None
    ):  # noqa: ANN001, ANN201
        """Best-effort pixel-level reconstruction error map (H,W)."""

        import torch
        import torch.nn.functional as F
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

        for batch in loader:
            images, _targets = batch
            images = images.to(self.device)
            with torch.no_grad():
                recon, _vq_loss = self.model(images, beta=float(self.cfg.beta_commit))  # type: ignore[operator]
                err = F.mse_loss(recon, images, reduction="none")
                # Mean over channels -> (H,W)
                m = err.mean(dim=1)[0]
                return m.detach().cpu().numpy().astype(np.float32)

        raise RuntimeError("No data produced for get_anomaly_map")


__all__ = ["VQVAEAnomalyDetector"]
