"""Compact SimpleNet adaptation for anomaly detection and localization."""

from __future__ import annotations

import logging
from itertools import chain
from typing import Iterable, Optional, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


class SimpleAdapter(nn.Module):
    """Shallow patch-feature adapter used by SimpleNet."""

    def __init__(self, in_channels: int = 1536, out_channels: int = 384) -> None:
        super().__init__()
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        return x.permute(0, 3, 1, 2)


class AnomalyDiscriminator(nn.Module):
    """Patch discriminator separating normal and noise-perturbed features."""

    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden = int(hidden_dim or feature_dim)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.network(patches).squeeze(-1)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Iterable[str], transform) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)


@register_model(
    "vision_simplenet",
    tags=("vision", "deep", "simplenet", "self-supervised", "pixel_map"),
    metadata={
        "description": "Compact SimpleNet adaptation with patch features and a noise-trained discriminator",
        "paper": "SimpleNet: A Simple Network for Image Anomaly Detection and Localization",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html",
        "year": 2023,
        "supervision": "self-supervised",
        "implementation_status": "compact-feature-pipeline-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
class VisionSimpleNet(BaseVisionDeepDetector):
    """Compact SimpleNet adaptation for Gaussian feature-noise discrimination."""

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        pretrained: bool = False,
        feature_dim: int = 384,
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 0.001,
        noise_std: float = 0.05,
        discriminator_margin: float = 0.5,
        image_size: int = 224,
        gaussian_sigma: float = 4.0,
        device: str = "cpu",
        random_state: Optional[int] = 42,
        **kwargs: object,
    ) -> None:
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if discriminator_margin <= 0:
            raise ValueError("discriminator_margin must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if image_size < 32:
            raise ValueError("image_size must be >= 32")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")

        requested_random_state = None if random_state is None else int(random_state)
        super().__init__(random_state=None, **kwargs)
        self.random_state = requested_random_state
        self.backbone_name = str(backbone)
        self.pretrained = bool(pretrained)
        self.feature_dim = int(feature_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.noise_std = float(noise_std)
        self.discriminator_margin = float(discriminator_margin)
        self.image_size = int(image_size)
        self.gaussian_sigma = float(gaussian_sigma)
        self.device = torch.device(device)
        self.is_fitted_ = False

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._build_model()

    def _build_model(self) -> None:
        if self.backbone_name not in {"wide_resnet50", "resnet50"}:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                "Choose 'wide_resnet50' or 'resnet50'."
            )
        with torch.random.fork_rng(devices=[]):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            backbone, _ = load_torchvision_model(
                self.backbone_name,
                pretrained=self.pretrained,
            )
            self.feature_extractor = nn.ModuleDict(
                {
                    "stem": nn.Sequential(
                        backbone.conv1,
                        backbone.bn1,
                        backbone.relu,
                        backbone.maxpool,
                        backbone.layer1,
                    ),
                    "layer2": backbone.layer2,
                    "layer3": backbone.layer3,
                }
            )
            self.adapter = SimpleAdapter(1536, self.feature_dim)
            self.discriminator = AnomalyDiscriminator(self.feature_dim)

        self.feature_extractor.to(self.device).eval()
        self.adapter.to(self.device)
        self.discriminator.to(self.device)
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.feature_extractor["stem"](images)
            layer2 = self.feature_extractor["layer2"](base)
            layer3 = self.feature_extractor["layer3"](layer2)
            layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
            layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
            layer3 = F.interpolate(
                layer3,
                size=layer2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return torch.cat((layer2, layer3), dim=1)

    def _adapted_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.adapter(self._extract_features(images))

    @staticmethod
    def _flatten_patches(features: torch.Tensor) -> torch.Tensor:
        return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionSimpleNet":
        del y
        paths = list(
            cast(
                Iterable[str],
                resolve_legacy_x_keyword(x, kwargs, method_name="fit"),
            )
        )
        if not paths:
            raise ValueError("Training set cannot be empty")

        loader = DataLoader(
            ImagePathDataset(paths, self.transform),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        optimizer = Adam(
            chain(self.adapter.parameters(), self.discriminator.parameters()),
            lr=self.lr,
            weight_decay=0.0,
        )
        noise_generator = torch.Generator(device=self.device.type)
        if self.random_state is not None:
            noise_generator.manual_seed(self.random_state)

        self.adapter.train()
        self.discriminator.train()
        for _epoch in range(self.epochs):
            for images in loader:
                images = images.to(self.device)
                normal = self._flatten_patches(self._adapted_features(images))
                noise = torch.randn(
                    normal.shape,
                    generator=noise_generator,
                    device=self.device,
                    dtype=normal.dtype,
                )
                synthetic_anomaly = normal + self.noise_std * noise
                normal_score = self.discriminator(normal)
                anomaly_score = self.discriminator(synthetic_anomaly)
                loss = F.relu(self.discriminator_margin - normal_score).mean() + F.relu(
                    self.discriminator_margin + anomaly_score
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.is_fitted_ = True
        self.decision_scores_ = self.decision_function(paths)
        self._process_decision_scores()
        return self

    @staticmethod
    def _load_rgb(path: str) -> tuple[NDArray, tuple[int, int]]:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        original_size = (int(image.shape[1]), int(image.shape[0]))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), original_size

    def _score_map(self, path: str) -> NDArray:
        image, original_size = self._load_rgb(path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._adapted_features(image_tensor)
            height, width = features.shape[-2:]
            scores = -self.discriminator(self._flatten_patches(features))
            score_map = scores.reshape(1, 1, height, width)
            score_map = F.interpolate(
                score_map,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False,
            )
        output = score_map.squeeze().cpu().numpy().astype(np.float32, copy=False)
        if self.gaussian_sigma > 0:
            output = cv2.GaussianBlur(output, (0, 0), sigmaX=self.gaussian_sigma)
        return np.asarray(output, dtype=np.float32)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        paths = list(
            cast(
                Iterable[str],
                resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            )
        )
        self.adapter.eval()
        self.discriminator.eval()
        return np.asarray([float(self._score_map(path).max()) for path in paths], dtype=np.float64)

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        if return_confidence:
            raise NotImplementedError("return_confidence is not implemented for VisionSimpleNet")
        if not self.is_fitted_ or not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")
        paths = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        return (self.decision_function(paths) > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image_path: str) -> NDArray:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self.adapter.eval()
        self.discriminator.eval()
        return self._score_map(str(image_path))

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        paths = list(
            cast(
                Iterable[str],
                resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
            )
        )
        return np.stack([self.get_anomaly_map(path) for path in paths])


__all__ = ["AnomalyDiscriminator", "SimpleAdapter", "VisionSimpleNet"]
