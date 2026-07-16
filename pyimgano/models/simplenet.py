"""Paper-aligned SimpleNet anomaly detection and localization."""

from __future__ import annotations

import logging
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
    """Paper's single bias-free fully connected feature adapter."""

    def __init__(self, in_channels: int = 1536, out_channels: int | None = None) -> None:
        super().__init__()
        output_dim = int(in_channels if out_channels is None else out_channels)
        self.projection = nn.Linear(int(in_channels), output_dim, bias=False)
        nn.init.xavier_normal_(self.projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class AnomalyDiscriminator(nn.Module):
    """Paper's Linear-BN-LeakyReLU-Linear normality discriminator."""

    def __init__(self, feature_dim: int = 1536, hidden_dim: Optional[int] = 1024) -> None:
        super().__init__()
        hidden = int(feature_dim if hidden_dim is None else hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1, bias=False),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

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
        "description": "Paper-aligned SimpleNet patch adapter and Gaussian-noise discriminator",
        "paper": "SimpleNet: A Simple Network for Image Anomaly Detection and Localization",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html",
        "year": 2023,
        "supervision": "self-supervised",
        "implementation_status": "paper-network-and-training-defaults-aligned",
        "paper_fidelity": "core-aligned",
    },
)
class VisionSimpleNet(BaseVisionDeepDetector):
    """SimpleNet with the paper's patch embedding, adapter, and discriminator."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        pretrained: bool = False,
        feature_dim: int = 1536,
        discriminator_hidden_dim: int = 1024,
        patch_size: int = 3,
        patch_stride: int = 1,
        epochs: int = 160,
        batch_size: int = 4,
        lr: float = 1e-4,
        discriminator_lr: float = 2e-4,
        weight_decay: float = 1e-5,
        noise_std: float = 0.015,
        discriminator_margin: float = 0.5,
        image_size: int = 224,
        resize_size: int = 256,
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
        if discriminator_hidden_dim < 1:
            raise ValueError("discriminator_hidden_dim must be positive")
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if patch_stride < 1:
            raise ValueError("patch_stride must be positive")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if discriminator_margin <= 0:
            raise ValueError("discriminator_margin must be positive")
        if lr <= 0 or discriminator_lr <= 0:
            raise ValueError("lr and discriminator_lr must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if image_size < 32:
            raise ValueError("image_size must be >= 32")
        if resize_size < image_size:
            raise ValueError("resize_size must be >= image_size")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")

        requested_random_state = None if random_state is None else int(random_state)
        super().__init__(random_state=None, **kwargs)
        self.random_state = requested_random_state
        backbone_aliases = {
            "wide_resnet50": "wide_resnet50_2",
            "wideresnet50": "wide_resnet50_2",
        }
        self.backbone_name = backbone_aliases.get(str(backbone), str(backbone))
        self.pretrained = bool(pretrained)
        self.feature_dim = int(feature_dim)
        self.discriminator_hidden_dim = int(discriminator_hidden_dim)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.discriminator_lr = float(discriminator_lr)
        self.weight_decay = float(weight_decay)
        self.noise_std = float(noise_std)
        self.discriminator_margin = float(discriminator_margin)
        self.image_size = int(image_size)
        self.resize_size = int(resize_size)
        self.gaussian_sigma = float(gaussian_sigma)
        self.device = torch.device(device)
        self.is_fitted_ = False

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._build_model()

    def _build_model(self) -> None:
        if self.backbone_name not in {"wide_resnet50_2", "resnet50"}:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                "Choose 'wide_resnet50_2' or 'resnet50'."
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
            self.adapter = SimpleAdapter(self.feature_dim)
            self.discriminator = AnomalyDiscriminator(
                self.feature_dim, self.discriminator_hidden_dim
            )

        self.feature_extractor.to(self.device).eval()
        self.adapter.to(self.device)
        self.discriminator.to(self.device)
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _patchify(
        features: torch.Tensor, *, patch_size: int, patch_stride: int
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        padding = (patch_size - 1) // 2
        unfolded = F.unfold(
            features,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=padding,
        )
        height = (features.shape[-2] + 2 * padding - patch_size) // patch_stride + 1
        width = (features.shape[-1] + 2 * padding - patch_size) // patch_stride + 1
        patches = unfolded.transpose(1, 2).reshape(
            features.shape[0], height * width, features.shape[1], patch_size, patch_size
        )
        return patches, (int(height), int(width))

    def _embed_feature_maps(
        self, feature_maps: list[torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        patchified = [
            self._patchify(
                item,
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
            )
            for item in feature_maps
        ]
        patches = [item[0] for item in patchified]
        grids = [item[1] for item in patchified]
        reference_grid = grids[0]

        for index in range(1, len(patches)):
            current = patches[index]
            grid = grids[index]
            current = current.reshape(
                current.shape[0], grid[0], grid[1], *current.shape[2:]
            ).permute(0, 3, 4, 5, 1, 2)
            base_shape = current.shape
            current = current.reshape(-1, 1, grid[0], grid[1])
            current = F.interpolate(
                current,
                size=reference_grid,
                mode="bilinear",
                align_corners=False,
            )
            current = current.reshape(*base_shape[:-2], *reference_grid)
            patches[index] = current.permute(0, 4, 5, 1, 2, 3).reshape(
                current.shape[0], reference_grid[0] * reference_grid[1], *current.shape[1:4]
            )

        flattened = [item.reshape(-1, *item.shape[-3:]) for item in patches]
        pooled = [
            F.adaptive_avg_pool1d(item.reshape(len(item), 1, -1), self.feature_dim).squeeze(1)
            for item in flattened
        ]
        stacked = torch.stack(pooled, dim=1).reshape(len(pooled[0]), 1, -1)
        embedded = F.adaptive_avg_pool1d(stacked, self.feature_dim).reshape(
            len(stacked), self.feature_dim
        )
        return embedded, reference_grid

    def _extract_features(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        with torch.no_grad():
            base = self.feature_extractor["stem"](images)
            layer2 = self.feature_extractor["layer2"](base)
            layer3 = self.feature_extractor["layer3"](layer2)
            return self._embed_feature_maps([layer2, layer3])

    def _adapted_features(self, images: torch.Tensor) -> torch.Tensor:
        features, _ = self._extract_features(images)
        return self.adapter(features)

    @staticmethod
    def _flatten_patches(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            return features
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

        loader_generator = torch.Generator()
        if self.random_state is not None:
            loader_generator.manual_seed(self.random_state)
        loader = DataLoader(
            ImagePathDataset(paths, self.transform),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=loader_generator,
        )
        adapter_optimizer = Adam(
            self.adapter.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        discriminator_optimizer = Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            weight_decay=self.weight_decay,
        )
        noise_generator = torch.Generator(device=self.device)
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
                loss = (
                    F.relu(self.discriminator_margin - normal_score).mean()
                    + F.relu(self.discriminator_margin + anomaly_score).mean()
                )
                adapter_optimizer.zero_grad(set_to_none=True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                adapter_optimizer.step()
                discriminator_optimizer.step()

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
            features, grid = self._extract_features(image_tensor)
            features = self.adapter(features)
            scores = -self.discriminator(features)
            height, width = grid
            score_map = scores.reshape(1, 1, height, width)
            score_map = F.interpolate(
                score_map,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False,
            )
        output = score_map.squeeze().cpu().numpy().astype(np.float32, copy=False)
        if self.gaussian_sigma > 0:
            from scipy.ndimage import gaussian_filter

            output = gaussian_filter(output, sigma=self.gaussian_sigma)
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
