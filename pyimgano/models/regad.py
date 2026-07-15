"""RegAD paper adaptation for category-agnostic few-shot anomaly detection.

The detector follows the ECCV 2022 RegAD path: an ImageNet ResNet-18 with
three spatial-transformer blocks, a convolutional SimSiam objective, and a
per-location Gaussian model fitted from an augmented target support set.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from pyimgano.models._image_batch import coerce_rgb_image_batch
from pyimgano.utils.random_state import isolated_random_state_method
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

_STN_PARAMETERS = {
    "affine": 6,
    "translation": 2,
    "rotation": 1,
    "scale": 2,
    "shear": 2,
    "rotation_scale": 3,
    "translation_scale": 4,
    "rotation_translation": 3,
    "rotation_translation_scale": 5,
}


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


class RegADSpatialTransformer(nn.Module):
    """The localization network used after each of RegAD's first three blocks."""

    def __init__(self, in_channels: int, feature_size: int, mode: str = "rotation_scale"):
        super().__init__()
        if mode not in _STN_PARAMETERS:
            raise ValueError(f"Unsupported stn_mode: {mode!r}")
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")

        self.mode = mode
        pooled_size = (int(feature_size) + 3) // 4
        self.localization = nn.Sequential(
            _conv3x3(in_channels, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _conv3x3(64, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(16 * pooled_size * pooled_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, _STN_PARAMETERS[mode]),
        )
        nn.init.zeros_(self.regressor[2].weight)
        identity = {
            "affine": (1, 0, 0, 0, 1, 0),
            "translation": (0, 0),
            "rotation": (0,),
            "scale": (1, 1),
            "shear": (0, 0),
            "rotation_scale": (0, 1, 1),
            "translation_scale": (0, 0, 1, 1),
            "rotation_translation": (0, 0, 0),
            "rotation_translation_scale": (0, 0, 0, 1, 1),
        }[mode]
        with torch.no_grad():
            self.regressor[2].bias.copy_(torch.tensor(identity, dtype=torch.float32))

    def _matrix(self, parameters: torch.Tensor) -> torch.Tensor:
        if self.mode == "affine":
            return parameters.reshape(-1, 2, 3)

        zero = torch.zeros_like(parameters[:, 0])
        one = torch.ones_like(parameters[:, 0])
        if self.mode == "translation":
            row1 = torch.stack((one, zero, parameters[:, 0]), dim=1)
            row2 = torch.stack((zero, one, parameters[:, 1]), dim=1)
        elif self.mode == "rotation":
            cosine, sine = torch.cos(parameters[:, 0]), torch.sin(parameters[:, 0])
            row1 = torch.stack((cosine, -sine, zero), dim=1)
            row2 = torch.stack((sine, cosine, zero), dim=1)
        elif self.mode == "scale":
            row1 = torch.stack((parameters[:, 0], zero, zero), dim=1)
            row2 = torch.stack((zero, parameters[:, 1], zero), dim=1)
        elif self.mode == "shear":
            row1 = torch.stack((one, parameters[:, 0], zero), dim=1)
            row2 = torch.stack((parameters[:, 1], one, zero), dim=1)
        elif self.mode == "rotation_scale":
            cosine, sine = torch.cos(parameters[:, 0]), torch.sin(parameters[:, 0])
            row1 = torch.stack((cosine * parameters[:, 1], -sine, zero), dim=1)
            row2 = torch.stack((sine, cosine * parameters[:, 2], zero), dim=1)
        elif self.mode == "translation_scale":
            row1 = torch.stack((parameters[:, 2], zero, parameters[:, 0]), dim=1)
            row2 = torch.stack((zero, parameters[:, 3], parameters[:, 1]), dim=1)
        elif self.mode == "rotation_translation":
            cosine, sine = torch.cos(parameters[:, 0]), torch.sin(parameters[:, 0])
            row1 = torch.stack((cosine, -sine, parameters[:, 1]), dim=1)
            row2 = torch.stack((sine, cosine, parameters[:, 2]), dim=1)
        else:  # rotation_translation_scale
            cosine, sine = torch.cos(parameters[:, 0]), torch.sin(parameters[:, 0])
            row1 = torch.stack((cosine * parameters[:, 3], -sine, parameters[:, 1]), dim=1)
            row2 = torch.stack((sine, cosine * parameters[:, 4], parameters[:, 2]), dim=1)
        return torch.stack((row1, row2), dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = self.regressor(self.localization(x).flatten(1))
        theta = self._matrix(parameters)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        transformed = F.grid_sample(
            x,
            grid,
            padding_mode="reflection",
            align_corners=False,
        )
        return transformed, theta


class RegADRegistrationNetwork(nn.Module):
    """ImageNet ResNet-18 stages 1--3, each followed by an STN."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        image_size: int = 224,
        stn_mode: str = "rotation_scale",
    ):
        super().__init__()
        resnet, _ = load_torchvision_model(
            "resnet18",
            pretrained=bool(pretrained),
            weights_name="IMAGENET1K_V1" if pretrained else None,
        )
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.stn1 = RegADSpatialTransformer(64, image_size // 4, stn_mode)
        self.stn2 = RegADSpatialTransformer(128, image_size // 8, stn_mode)
        self.stn3 = RegADSpatialTransformer(256, image_size // 16, stn_mode)

    @staticmethod
    def _warp(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(
            x,
            grid,
            padding_mode="reflection",
            align_corners=False,
        )

    @staticmethod
    def _inverse(theta: torch.Tensor) -> torch.Tensor:
        bottom = theta.new_zeros((theta.shape[0], 1, 3))
        bottom[:, :, 2] = 1
        return torch.linalg.inv(torch.cat((theta, bottom), dim=1))[:, :2, :]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x, _ = self.stn1(self.layer1(x))
        x, _ = self.stn2(self.layer2(x))
        out, _ = self.stn3(self.layer3(x))
        return out

    def aligned_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the network while retaining the transformed outputs for inference."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        stage1, theta1 = self.stn1(self.layer1(x))
        stage2, theta2 = self.stn2(self.layer2(stage1))
        stage3, theta3 = self.stn3(self.layer3(stage2))
        inverse1 = self._inverse(theta1.detach())
        inverse2 = self._inverse(theta2.detach())
        inverse3 = self._inverse(theta3.detach())
        aligned1 = self._warp(stage1.detach(), inverse1)
        aligned2 = self._warp(self._warp(stage2.detach(), inverse2), inverse1)
        aligned3 = self._warp(
            self._warp(self._warp(stage3.detach(), inverse3), inverse2),
            inverse1,
        )
        return aligned1, aligned2, aligned3


class RegADEncoder(nn.Module):
    """Three spatial 1x1 convolutions from the paper's Siamese encoder."""

    def __init__(self):
        super().__init__()
        self.conv1 = _conv1x1(256, 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = _conv1x1(256, 256)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = _conv1x1(256, 256)
        # Present in the authors' checkpoint even though their forward omits it.
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.conv3(x)


class RegADPredictor(nn.Module):
    """Two spatial 1x1 convolutions from the paper's prediction head."""

    def __init__(self):
        super().__init__()
        self.conv1 = _conv1x1(256, 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = _conv1x1(256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.bn1(self.conv1(x))))


class RegADModel(nn.Module):
    """Registration backbone and convolutional SimSiam heads."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        image_size: int = 224,
        stn_mode: str = "rotation_scale",
    ):
        super().__init__()
        self.registration = RegADRegistrationNetwork(
            pretrained=pretrained,
            image_size=image_size,
            stn_mode=stn_mode,
        )
        self.encoder = RegADEncoder()
        self.predictor = RegADPredictor()

    @staticmethod
    def _negative_cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -F.cosine_similarity(prediction, target.detach(), dim=1).mean()

    def registration_loss(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        if support.ndim != 5 or support.shape[0] != query.shape[0]:
            raise ValueError("support must have shape (batch, shot, channels, height, width)")
        batch, shot, channels, height, width = support.shape
        query_features = self.registration(query)
        support_features = self.registration(support.reshape(batch * shot, channels, height, width))
        support_features = support_features.reshape(batch, shot, *support_features.shape[1:]).mean(
            dim=1
        )
        query_embedding = self.encoder(query_features)
        support_embedding = self.encoder(support_features)
        return 0.5 * (
            self._negative_cosine(self.predictor(query_embedding), support_embedding)
            + self._negative_cosine(self.predictor(support_embedding), query_embedding)
        )

    def aligned_features(self, x: torch.Tensor) -> torch.Tensor:
        stage1, stage2, stage3 = self.registration.aligned_outputs(x)
        size = stage1.shape[-2:]
        return torch.cat(
            (
                stage1,
                F.interpolate(stage2, size=size, mode="nearest"),
                F.interpolate(stage3, size=size, mode="nearest"),
            ),
            dim=1,
        )


@register_model(
    "vision_regad",
    tags=(
        "vision",
        "deep",
        "regad",
        "registration",
        "few-shot",
        "self-supervised",
        "pixel_map",
        "paper",
    ),
    metadata={
        "description": "RegAD ResNet-18/STN registration with few-shot Gaussian scoring",
        "paper": "Registration based Few-Shot Anomaly Detection",
        "paper_url": "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840300.pdf",
        "year": 2022,
        "implementation_status": "paper-resnet18-stn-siamese-gaussian-path-aligned",
        "paper_fidelity": "paper-adaptation",
        "supervision": "few-shot",
        "type": "registration",
        "supports_pixel_map": True,
    },
)
class VisionRegAD(BaseVisionDeepDetector):
    """Category-agnostic RegAD adapted to the PyImgAno detector contract.

    ``fit`` trains on normal source-category images and requires their category
    labels plus a separate target-category normal support set. ``set_support``
    can then replace that target support set without retraining the network.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        image_size: int = 224,
        stn_mode: str = "rotation_scale",
        learning_rate: float = 1e-4,
        momentum: float = 0.9,
        batch_size: int = 32,
        epochs: int = 50,
        shot: int = 2,
        covariance_regularization: float = 0.01,
        gaussian_sigma: float = 4.0,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        if backbone != "resnet18":
            raise ValueError("The RegAD paper architecture supports only backbone='resnet18'.")
        if image_size < 32 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if stn_mode not in _STN_PARAMETERS:
            raise ValueError(f"Unsupported stn_mode: {stn_mode!r}")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if batch_size <= 0 or epochs < 0 or shot <= 0:
            raise ValueError("batch_size and shot must be positive; epochs must be nonnegative")
        if covariance_regularization <= 0:
            raise ValueError("covariance_regularization must be positive")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be nonnegative")

        self.backbone = backbone
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)
        self.stn_mode = stn_mode
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.shot = int(shot)
        self.covariance_regularization = float(covariance_regularization)
        self.gaussian_sigma = float(gaussian_sigma)
        requested_device = torch.device(device)
        self.device = (
            requested_device
            if requested_device.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.random_state = random_state

        self.model_: Optional[RegADModel] = None
        self.support_mean_: Optional[torch.Tensor] = None
        self.support_centered_: Optional[torch.Tensor] = None
        self.support_cholesky_: Optional[torch.Tensor] = None

    def _build_model(self) -> RegADModel:
        return RegADModel(
            pretrained=self.pretrained,
            image_size=self.image_size,
            stn_mode=self.stn_mode,
        ).to(self.device)

    def _preprocess(self, x: object) -> torch.Tensor:
        """Match the authors' Resize + ToTensor path (no ImageNet normalization)."""
        images = coerce_rgb_image_batch(x)
        integer_input = np.issubdtype(images.dtype, np.integer)
        images = images.astype(np.float32)
        if not np.isfinite(images).all():
            raise ValueError("RegAD images must contain only finite values")
        if images.size == 0:
            raise ValueError("RegAD requires at least one image")
        if float(images.min()) < 0:
            raise ValueError("RegAD images must be nonnegative")
        if integer_input or float(images.max()) > 1.0:
            if float(images.max()) > 255.0:
                raise ValueError("RegAD image values must be in [0, 1] or [0, 255]")
            images /= 255.0
        tensor = torch.from_numpy(np.transpose(images, (0, 3, 1, 2)).copy()).float()
        return F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _affine_images(images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        return F.grid_sample(
            images,
            grid,
            padding_mode="reflection",
            align_corners=False,
        )

    def _augment_support(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        dtype, device = images.dtype, images.device
        augmented = [images]
        for angle in (
            -math.pi / 4,
            -3 * math.pi / 16,
            -math.pi / 8,
            -math.pi / 16,
            math.pi / 16,
            math.pi / 8,
            3 * math.pi / 16,
            math.pi / 4,
        ):
            cosine, sine = math.cos(angle), math.sin(angle)
            theta = torch.tensor(
                ((cosine, -sine, 0), (sine, cosine, 0)), dtype=dtype, device=device
            ).expand(batch, -1, -1)
            augmented.append(self._affine_images(images, theta))
        for horizontal, vertical in (
            (0.2, 0.2),
            (-0.2, 0.2),
            (-0.2, -0.2),
            (0.2, -0.2),
            (0.1, 0.1),
            (-0.1, 0.1),
            (-0.1, -0.1),
            (0.1, -0.1),
        ):
            theta = torch.tensor(
                ((1, 0, horizontal), (0, 1, vertical)), dtype=dtype, device=device
            ).expand(batch, -1, -1)
            augmented.append(self._affine_images(images, theta))
        augmented.append(torch.flip(images, dims=(-1,)))
        gray = images[:, 0:1] * 0.299 + images[:, 1:2] * 0.587 + images[:, 2:3] * 0.114
        augmented.append(gray.expand(-1, 3, -1, -1))
        augmented.extend(torch.rot90(images, turns, dims=(-2, -1)) for turns in (1, 2, 3))
        return torch.cat(augmented, dim=0)

    @staticmethod
    def _category_groups(labels: NDArray, sample_count: int) -> list[NDArray[np.int64]]:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != sample_count:
            raise ValueError("y must contain one source-category label per training image")
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        if len(groups) < 2:
            raise ValueError(
                "RegAD category-agnostic training requires at least two source categories"
            )
        if any(len(group) < 2 for group in groups):
            raise ValueError("Each RegAD source category must contain at least two normal images")
        return [group.astype(np.int64, copy=False) for group in groups]

    def _train_registration(self, images: torch.Tensor, labels: NDArray) -> None:
        if self.model_ is None:
            raise RuntimeError("RegAD model is not initialized")
        groups = self._category_groups(np.asarray(labels), len(images))
        group_for_index = {int(index): group for group in groups for index in group.tolist()}
        rng = np.random.default_rng(self.random_state)
        optimizer = torch.optim.SGD(
            self.model_.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=0.0,
        )
        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            learning_rate = (
                self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * epoch / self.epochs))
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            batches = []
            for group in groups:
                shuffled = rng.permutation(group)
                batches.extend(
                    shuffled[start : start + self.batch_size]
                    for start in range(0, len(shuffled), self.batch_size)
                )
            rng.shuffle(batches)
            for query_indices in batches:
                support_indices = []
                for query_index in query_indices:
                    candidates = group_for_index[int(query_index)]
                    candidates = candidates[candidates != query_index]
                    support_indices.append(rng.choice(candidates, size=self.shot, replace=True))
                query = images[query_indices].to(self.device)
                support = images[np.asarray(support_indices)].to(self.device)
                loss = self.model_.registration_loss(query, support)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _fit_gaussian(self, features: torch.Tensor) -> None:
        if features.ndim != 4 or features.shape[0] < 2:
            raise ValueError("RegAD Gaussian estimation requires at least two support features")
        count, channels = features.shape[:2]
        mean = features.mean(dim=0)
        centered = (features - mean).permute(2, 3, 0, 1).reshape(-1, count, channels)
        identity = torch.eye(count, dtype=features.dtype, device=features.device)
        dual_covariance = (count - 1) * identity.unsqueeze(0) + centered @ centered.transpose(
            1, 2
        ) / self.covariance_regularization
        self.support_mean_ = mean
        self.support_centered_ = centered
        self.support_cholesky_ = torch.linalg.cholesky(dual_covariance)

    def _mahalanobis_map(self, features: torch.Tensor) -> torch.Tensor:
        if (
            self.support_mean_ is None
            or self.support_centered_ is None
            or self.support_cholesky_ is None
        ):
            raise RuntimeError("Call fit(..., support_images=...) or set_support() first")
        batch, _, height, width = features.shape
        delta = (
            (features - self.support_mean_)
            .permute(0, 2, 3, 1)
            .reshape(batch, -1, features.shape[1])
        )
        projection = torch.einsum("pnc,bpc->pnb", self.support_centered_, delta)
        solved = torch.cholesky_solve(projection, self.support_cholesky_)
        regularization = self.covariance_regularization
        squared = delta.square().sum(dim=2) / regularization
        squared -= (projection * solved).sum(dim=1).transpose(0, 1) / (regularization**2)
        return squared.clamp_min(0).sqrt().reshape(batch, height, width)

    def _score_tensor(self, images: torch.Tensor) -> NDArray[np.float32]:
        if self.model_ is None:
            raise RuntimeError("RegAD model has not been trained")
        maps = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                features = self.model_.aligned_features(
                    images[start : start + self.batch_size].to(self.device)
                )
                score_map = self._mahalanobis_map(features).unsqueeze(1)
                score_map = F.interpolate(
                    score_map,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                maps.append(score_map.cpu())
        result = torch.cat(maps).numpy().astype(np.float32, copy=False)
        if self.gaussian_sigma:
            for index in range(len(result)):
                result[index] = gaussian_filter(result[index], sigma=self.gaussian_sigma)
        return result

    def set_support(self, x: object) -> "VisionRegAD":
        """Estimate the target-category normal distribution without fine-tuning."""
        if self.model_ is None:
            raise RuntimeError("Train or load the RegAD registration model before set_support()")
        support = self._preprocess(x)
        augmented = self._augment_support(support.to(self.device))
        features = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(augmented), self.batch_size):
                features.append(
                    self.model_.aligned_features(augmented[start : start + self.batch_size])
                )
        self._fit_gaussian(torch.cat(features))
        self.decision_scores_ = self._score_tensor(support).reshape(len(support), -1).max(axis=1)
        self._process_decision_scores()
        self.is_fitted_ = True
        return self

    @isolated_random_state_method
    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        *,
        support_images: object = MISSING,
        **kwargs: object,
    ) -> "VisionRegAD":
        """Train on labeled source categories, then fit the target support distribution."""
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        if support_images is MISSING:
            raise ValueError(
                "RegAD requires target-category normal support_images; it cannot use a same-category mean proxy."
            )
        if self.epochs and y is None:
            raise ValueError(
                "RegAD category-agnostic training requires source-category labels in y."
            )
        if not self.epochs and self.model_ is None:
            raise ValueError("epochs=0 requires an explicitly preloaded RegAD model_ checkpoint")
        source = self._preprocess(x_value)
        if y is not None:
            self._category_groups(np.asarray(y), len(source))
        if self.model_ is None:
            self.model_ = self._build_model()
        if self.epochs:
            self._train_registration(source, np.asarray(y))
        self.set_support(support_images)
        return self

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        images = coerce_rgb_image_batch(x_value)
        original_size = images.shape[1:3]
        maps = self._score_tensor(self._preprocess(images))
        if original_size != (self.image_size, self.image_size):
            maps = (
                F.interpolate(
                    torch.from_numpy(maps).unsqueeze(1),
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .numpy()
            )
        return maps.astype(np.float32, copy=False)

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        maps = self.predict_anomaly_map(x_value)
        return maps.reshape(len(maps), -1).max(axis=1).astype(np.float64)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        if batch_size is None:
            return self.predict(x_value)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        previous = self.batch_size
        try:
            self.batch_size = batch_size
            return self.predict(x_value)
        finally:
            self.batch_size = previous

    def get_registration_map(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="get_registration_map")
        return self.predict_anomaly_map(x_value)
