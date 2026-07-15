"""ALAD image detector aligned with the paper's CIFAR-10/SVHN network."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .baseCv import BaseVisionDeepDetector
from .registry import register_model

_PAPER_IMAGE_SIZE = 32
_PAPER_INIT_STD = 0.01


def _spectral(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


def _paper_initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        weight = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(weight, mean=0.0, std=_PAPER_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _batch_norm(channels: int) -> nn.BatchNorm2d:
    # TensorFlow's released implementation uses epsilon=1e-3 and momentum=0.99.
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class ConvEncoder(nn.Module):
    """Paper encoder: 128-256-512 strided convolutions and a 100-D code."""

    def __init__(
        self,
        latent_dim: int = 100,
        *,
        in_ch: int = 3,
        spectral_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = _spectral(nn.Conv2d(in_ch, 128, 4, 2, 1), spectral_normalization)
        self.bn1 = _batch_norm(128)
        self.conv2 = _spectral(nn.Conv2d(128, 256, 4, 2, 1), spectral_normalization)
        self.bn2 = _batch_norm(256)
        self.conv3 = _spectral(nn.Conv2d(256, 512, 4, 2, 1), spectral_normalization)
        self.bn3 = _batch_norm(512)
        # The released model intentionally leaves the final latent convolution unnormalized.
        self.conv4 = nn.Conv2d(512, latent_dim, 4, 1, 0)
        self.apply(_paper_initialize)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (_PAPER_IMAGE_SIZE, _PAPER_IMAGE_SIZE):
            images = F.interpolate(
                images,
                size=(_PAPER_IMAGE_SIZE, _PAPER_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        hidden = F.leaky_relu(self.bn1(self.conv1(images)), negative_slope=0.2)
        hidden = F.leaky_relu(self.bn2(self.conv2(hidden)), negative_slope=0.2)
        hidden = F.leaky_relu(self.bn3(self.conv3(hidden)), negative_slope=0.2)
        return self.conv4(hidden).flatten(1)


class ConvDecoder(nn.Module):
    """Paper generator: 512-256-128 transposed convolutions and tanh RGB output."""

    def __init__(self, latent_dim: int = 100, *, out_ch: int = 3) -> None:
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 512, 4, 2, 0)
        self.bn1 = _batch_norm(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn2 = _batch_norm(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn3 = _batch_norm(128)
        self.deconv4 = nn.ConvTranspose2d(128, out_ch, 4, 2, 1)
        self.apply(_paper_initialize)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = latent.reshape(latent.shape[0], latent.shape[1], 1, 1)
        hidden = F.relu(self.bn1(self.deconv1(hidden)))
        hidden = F.relu(self.bn2(self.deconv2(hidden)))
        hidden = F.relu(self.bn3(self.deconv3(hidden)))
        return torch.tanh(self.deconv4(hidden))


class DiscXZ(nn.Module):
    """Joint data/latent discriminator from the released image architecture."""

    def __init__(
        self,
        latent_dim: int = 100,
        *,
        dropout_rate: float = 0.2,
        spectral_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.x_branch = nn.Sequential(
            _spectral(nn.Conv2d(3, 128, 4, 2, 1), spectral_normalization),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral(nn.Conv2d(128, 256, 4, 2, 1), spectral_normalization),
            _batch_norm(256),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral(nn.Conv2d(256, 512, 4, 2, 1), spectral_normalization),
            _batch_norm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        self.z_branch = nn.Sequential(
            _spectral(nn.Linear(latent_dim, 512), spectral_normalization),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            _spectral(nn.Linear(512, 512), spectral_normalization),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.joint = nn.Sequential(
            _spectral(nn.Linear(512 * 4 * 4 + 512, 1024), spectral_normalization),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.logit = nn.Linear(1024, 1)
        self.apply(_paper_initialize)

    def features(self, images: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (_PAPER_IMAGE_SIZE, _PAPER_IMAGE_SIZE):
            images = F.interpolate(
                images,
                size=(_PAPER_IMAGE_SIZE, _PAPER_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        return self.joint(torch.cat([self.x_branch(images), self.z_branch(latent)], dim=1))

    def forward(self, images: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.logit(self.features(images, latent)).flatten()


class DiscXX(nn.Module):
    """Image cycle discriminator; its pre-logit activations define the ALAD score."""

    def __init__(
        self,
        *,
        dropout_rate: float = 0.2,
        spectral_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = _spectral(nn.Conv2d(6, 64, 5, 2, 2), spectral_normalization)
        self.conv2 = _spectral(nn.Conv2d(64, 128, 5, 2, 2), spectral_normalization)
        self.dropout = nn.Dropout(dropout_rate)
        self.logit = nn.Linear(128 * 8 * 8, 1)
        self.apply(_paper_initialize)

    def features(self, images: torch.Tensor, paired_images: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([images, paired_images], dim=1)
        if pair.shape[-2:] != (_PAPER_IMAGE_SIZE, _PAPER_IMAGE_SIZE):
            pair = F.interpolate(
                pair,
                size=(_PAPER_IMAGE_SIZE, _PAPER_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        hidden = self.dropout(F.leaky_relu(self.conv1(pair), negative_slope=0.2))
        hidden = self.dropout(F.leaky_relu(self.conv2(hidden), negative_slope=0.2))
        return hidden.flatten(1)

    def forward(self, images: torch.Tensor, paired_images: torch.Tensor) -> torch.Tensor:
        return self.logit(self.features(images, paired_images)).flatten()


class DiscZZ(nn.Module):
    """Latent cycle discriminator: 2z -> 64 -> 32 -> 1."""

    def __init__(
        self,
        latent_dim: int = 100,
        *,
        dropout_rate: float = 0.2,
        spectral_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            _spectral(nn.Linear(2 * latent_dim, 64), spectral_normalization),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            _spectral(nn.Linear(64, 32), spectral_normalization),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.logit = nn.Linear(32, 1)
        self.apply(_paper_initialize)

    def forward(self, latent: torch.Tensor, paired_latent: torch.Tensor) -> torch.Tensor:
        return self.logit(self.hidden(torch.cat([latent, paired_latent], dim=1))).flatten()


def _paper_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((_PAPER_IMAGE_SIZE, _PAPER_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


@register_model(
    "vision_alad",
    tags=("vision", "deep", "gan", "cycle-consistency", "spectral-normalization"),
    metadata={
        "description": "ALAD with the paper image network, losses, EMA, and feature score",
        "paper": "Adversarially Learned Anomaly Detection",
        "paper_url": "https://arxiv.org/abs/1812.02288",
        "year": 2018,
        "supervision": "one-class",
        "implementation_status": "paper-image-network-industrial-input-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
class ALAD(BaseVisionDeepDetector):
    """ALAD's CIFAR-10/SVHN network exposed through the vision detector contract."""

    def __init__(
        self,
        *,
        latent_dim: int = 100,
        dropout_rate: float = 0.2,
        learning_rate_gen: float = 2e-4,
        learning_rate_disc: float = 2e-4,
        add_recon_loss: bool = False,
        lambda_recon_loss: float = 0.1,
        add_disc_zz_loss: bool = True,
        spectral_normalization: bool = True,
        score_degree: float = 1.0,
        ema_decay: float = 0.999,
        contamination: float = 0.1,
        preprocessing: bool = True,
        lr: float = 2e-4,
        epoch_num: int = 100,
        batch_size: int = 32,
        optimizer_name: str = "adam",
        criterion_name: str = "mse",
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
        verbose: int = 1,
        train_transform=None,
        eval_transform=None,
        **kwargs,
    ) -> None:
        if preprocessing:
            train_transform = _paper_transform() if train_transform is None else train_transform
            eval_transform = _paper_transform() if eval_transform is None else eval_transform
        super().__init__(
            contamination=contamination,
            preprocessing=preprocessing,
            lr=lr,
            epoch_num=epoch_num,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            criterion_name=criterion_name,
            device=device,
            random_state=random_state,
            verbose=verbose,
            train_transform=train_transform,
            eval_transform=eval_transform,
            **kwargs,
        )
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if learning_rate_gen <= 0.0 or learning_rate_disc <= 0.0:
            raise ValueError("learning rates must be positive")
        if score_degree <= 0.0:
            raise ValueError("score_degree must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")

        self.latent_dim = int(latent_dim)
        self.dropout_rate = float(dropout_rate)
        self.learning_rate_gen = float(learning_rate_gen)
        self.learning_rate_disc = float(learning_rate_disc)
        self.add_recon_loss = bool(add_recon_loss)
        self.lambda_recon_loss = float(lambda_recon_loss)
        self.add_disc_zz_loss = bool(add_disc_zz_loss)
        self.spectral_normalization = bool(spectral_normalization)
        self.score_degree = float(score_degree)
        self.ema_decay = float(ema_decay)
        self.ema_enabled = True
        self.ema_start_epoch = 1

        self.enc: Optional[ConvEncoder] = None
        self.dec: Optional[ConvDecoder] = None
        self.disc_xx: Optional[DiscXX] = None
        self.disc_xz: Optional[DiscXZ] = None
        self.disc_zz: Optional[DiscZZ] = None
        self.img_feat: Optional[nn.Module] = None
        self.opt_gen: Optional[torch.optim.Optimizer] = None
        self.opt_enc: Optional[torch.optim.Optimizer] = None
        self.opt_disc: Optional[torch.optim.Optimizer] = None
        self.hist_loss_disc: List[float] = []
        self.hist_loss_gen: List[float] = []
        self.hist_loss_enc: List[float] = []

    def build_model(self) -> nn.ModuleList:
        self.enc = ConvEncoder(
            self.latent_dim,
            spectral_normalization=self.spectral_normalization,
        ).to(self.device)
        self.dec = ConvDecoder(self.latent_dim).to(self.device)
        self.disc_xz = DiscXZ(
            self.latent_dim,
            dropout_rate=self.dropout_rate,
            spectral_normalization=self.spectral_normalization,
        ).to(self.device)
        self.disc_xx = DiscXX(
            dropout_rate=self.dropout_rate,
            spectral_normalization=self.spectral_normalization,
        ).to(self.device)
        self.disc_zz = DiscZZ(
            self.latent_dim,
            dropout_rate=self.dropout_rate,
            spectral_normalization=self.spectral_normalization,
        ).to(self.device)
        self.img_feat = self.disc_xz.x_branch

        self.opt_gen = torch.optim.Adam(
            self.dec.parameters(),
            lr=self.learning_rate_gen,
            betas=(0.5, 0.999),
        )
        self.opt_enc = torch.optim.Adam(
            self.enc.parameters(),
            lr=self.learning_rate_gen,
            betas=(0.5, 0.999),
        )
        discriminator_parameters = list(self.disc_xz.parameters()) + list(
            self.disc_xx.parameters()
        )
        if self.add_disc_zz_loss:
            discriminator_parameters.extend(self.disc_zz.parameters())
        self.opt_disc = torch.optim.Adam(
            discriminator_parameters,
            lr=self.learning_rate_disc,
            betas=(0.5, 0.999),
        )
        # Prevent BaseDeepLearningDetector from constructing an unused optimizer.
        self.optimizer = self.opt_gen

        return nn.ModuleList([self.enc, self.dec, self.disc_xz, self.disc_xx, self.disc_zz])

    def _modules_or_raise(self) -> tuple[ConvEncoder, ConvDecoder, DiscXZ, DiscXX, DiscZZ]:
        modules = (self.enc, self.dec, self.disc_xz, self.disc_xx, self.disc_zz)
        if any(module is None for module in modules):
            raise RuntimeError("ALAD model has not been built")
        return modules  # type: ignore[return-value]

    @staticmethod
    def _set_discriminator_grad(
        discriminators: tuple[nn.Module, ...],
        *,
        enabled: bool,
    ) -> None:
        for discriminator in discriminators:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(enabled)

    def training_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        encoder, decoder, disc_xz, disc_xx, disc_zz = self._modules_or_raise()
        encoder.train()
        decoder.train()
        disc_xz.train()
        disc_xx.train()
        disc_zz.train()
        images, _targets = batch
        images = images.to(self.device)
        batch_size = images.shape[0]
        criterion = nn.BCEWithLogitsLoss()
        discriminators = (disc_xz, disc_xx, disc_zz)

        self._set_discriminator_grad(discriminators, enabled=True)
        self.opt_disc.zero_grad(set_to_none=True)
        latent_disc = torch.randn(batch_size, self.latent_dim, device=self.device)
        with torch.no_grad():
            generated_disc = decoder(latent_disc)
            encoded_disc = encoder(images)
            reconstructed_disc = decoder(encoded_disc)
            recycled_latent_disc = encoder(generated_disc)

        loss_disc_xz = criterion(disc_xz(images, encoded_disc), torch.ones(batch_size, device=self.device))
        loss_disc_xz = loss_disc_xz + criterion(
            disc_xz(generated_disc, latent_disc),
            torch.zeros(batch_size, device=self.device),
        )
        loss_disc_xx = criterion(disc_xx(images, images), torch.ones(batch_size, device=self.device))
        loss_disc_xx = loss_disc_xx + criterion(
            disc_xx(images, reconstructed_disc),
            torch.zeros(batch_size, device=self.device),
        )
        loss_disc = loss_disc_xz + loss_disc_xx
        if self.add_disc_zz_loss:
            loss_disc_zz = criterion(
                disc_zz(latent_disc, latent_disc),
                torch.ones(batch_size, device=self.device),
            )
            loss_disc_zz = loss_disc_zz + criterion(
                disc_zz(latent_disc, recycled_latent_disc),
                torch.zeros(batch_size, device=self.device),
            )
            loss_disc = loss_disc + loss_disc_zz
        loss_disc.backward()
        self.opt_disc.step()

        self._set_discriminator_grad(discriminators, enabled=False)
        self.opt_gen.zero_grad(set_to_none=True)
        self.opt_enc.zero_grad(set_to_none=True)
        latent = torch.randn(batch_size, self.latent_dim, device=self.device)
        generated = decoder(latent)
        encoded = encoder(images)
        reconstructed = decoder(encoded)
        recycled_latent = encoder(generated)

        real_xz_logits = disc_xz(images, encoded)
        fake_xz_logits = disc_xz(generated, latent)
        real_xx_logits = disc_xx(images, images)
        fake_xx_logits = disc_xx(images, reconstructed)
        cost_x = criterion(real_xx_logits, torch.zeros_like(real_xx_logits)) + criterion(
            fake_xx_logits,
            torch.ones_like(fake_xx_logits),
        )
        cycle_cost = cost_x
        if self.add_disc_zz_loss:
            real_zz_logits = disc_zz(latent, latent)
            fake_zz_logits = disc_zz(latent, recycled_latent)
            cost_z = criterion(real_zz_logits, torch.zeros_like(real_zz_logits)) + criterion(
                fake_zz_logits,
                torch.ones_like(fake_zz_logits),
            )
            cycle_cost = cycle_cost + cost_z

        loss_gen = criterion(fake_xz_logits, torch.ones_like(fake_xz_logits)) + cycle_cost
        loss_enc = criterion(real_xz_logits, torch.zeros_like(real_xz_logits)) + cycle_cost
        if self.add_recon_loss:
            reconstruction_loss = F.mse_loss(reconstructed, images)
            loss_gen = loss_gen + self.lambda_recon_loss * reconstruction_loss
            loss_enc = loss_enc + self.lambda_recon_loss * reconstruction_loss

        decoder_parameters = tuple(decoder.parameters())
        encoder_parameters = tuple(encoder.parameters())
        decoder_gradients = torch.autograd.grad(
            loss_gen,
            decoder_parameters,
            retain_graph=True,
        )
        encoder_gradients = torch.autograd.grad(loss_enc, encoder_parameters)
        for parameter, gradient in zip(decoder_parameters, decoder_gradients):
            parameter.grad = gradient
        for parameter, gradient in zip(encoder_parameters, encoder_gradients):
            parameter.grad = gradient
        self.opt_gen.step()
        self.opt_enc.step()
        self._set_discriminator_grad(discriminators, enabled=True)

        disc_value = float(loss_disc.detach())
        gen_value = float(loss_gen.detach())
        enc_value = float(loss_enc.detach())
        self.hist_loss_disc.append(disc_value)
        self.hist_loss_gen.append(gen_value)
        self.hist_loss_enc.append(enc_value)
        return disc_value + gen_value + enc_value

    @torch.no_grad()
    def evaluating_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        encoder, decoder, _disc_xz, disc_xx, _disc_zz = self._modules_or_raise()
        encoder.eval()
        decoder.eval()
        disc_xx.eval()
        images, _targets = batch
        images = images.to(self.device)
        reconstructed = decoder(encoder(images))
        real_features = disc_xx.features(images, images)
        reconstructed_features = disc_xx.features(images, reconstructed)
        score = torch.norm(
            real_features - reconstructed_features,
            p=self.score_degree,
            dim=1,
        )
        return score.cpu().numpy()

    def get_history(self) -> Dict[str, List[float]]:
        return {
            "loss_disc": self.hist_loss_disc,
            "loss_gen": self.hist_loss_gen,
            "loss_enc": self.hist_loss_enc,
        }


__all__ = ["ALAD", "ConvDecoder", "ConvEncoder", "DiscXX", "DiscXZ", "DiscZZ"]
