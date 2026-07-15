"""Image DevNet for weakly supervised anomaly detection.

This module follows the image formulation from "Explainable Deep Few-shot
Anomaly Detection with Deviation Networks" (2021): a trainable ResNet feature
map, a 1x1 patch-score head, top-K multiple-instance aggregation, and the
Gaussian-reference deviation loss.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from pyimgano.utils.random_state import isolated_random_state_method
from pyimgano.utils.torchvision_safe import load_torchvision_model

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class DeviationLoss(nn.Module):
    """Z-score deviation loss with the paper's Gaussian reference sample."""

    def __init__(self, margin: float = 5.0, reference_size: int = 5000) -> None:
        super().__init__()
        if reference_size < 2:
            raise ValueError("reference_size must be at least 2.")
        self.margin = float(margin)
        self.reference_size = int(reference_size)

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        ref_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if scores.shape != labels.shape:
            raise ValueError(f"scores and labels must share shape; got {scores.shape} and {labels.shape}")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must contain only 0 (normal) and 1 (anomaly)")

        reference = (
            torch.randn(self.reference_size, device=scores.device, dtype=scores.dtype)
            if ref_scores is None
            else ref_scores.to(device=scores.device, dtype=scores.dtype).reshape(-1)
        )
        if reference.numel() < 2:
            raise ValueError("ref_scores must contain at least two values")

        deviation = (scores - reference.mean()) / reference.std(unbiased=False).clamp_min(1e-6)
        normal_loss = deviation.abs()
        anomaly_loss = F.relu(self.margin - deviation)
        labels_float = labels.to(dtype=scores.dtype)
        return ((1.0 - labels_float) * normal_loss + labels_float * anomaly_loss).mean()


class FeatureExtractor(nn.Module):
    """ResNet convolutional feature map used by image DevNet."""

    _FEATURE_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False) -> None:
        super().__init__()
        if backbone not in self._FEATURE_DIMS:
            raise ValueError(f"Unknown backbone: {backbone}")
        resnet, _ = load_torchvision_model(backbone, pretrained=bool(pretrained))
        self.feature_dim = self._FEATURE_DIMS[backbone]
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class DevNetModel(nn.Module):
    """Paper image network: ResNet patches, linear scores, and top-K MIL."""

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        pretrained: bool = False,
        image_size: int = 448,
        n_scales: int = 2,
        topk_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size <= 0 or n_scales <= 0:
            raise ValueError("image_size and n_scales must be positive.")
        if not 0.0 <= topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in [0, 1].")

        self.image_size = int(image_size)
        self.n_scales = int(n_scales)
        self.topk_ratio = float(topk_ratio)
        self.feature_extractor = FeatureExtractor(backbone=backbone, pretrained=pretrained)
        self.score_head = nn.Conv2d(self.feature_extractor.feature_dim, 1, kernel_size=1)

    @staticmethod
    def aggregate_patch_scores(patch_scores: torch.Tensor, topk_ratio: float) -> torch.Tensor:
        flattened = patch_scores.flatten(1)
        if topk_ratio > 0:
            count = max(int(flattened.shape[1] * float(topk_ratio)), 1)
            # Equation 6 selects the largest signed anomaly scores, not magnitudes.
            flattened = torch.topk(flattened, count, dim=1).values
        return flattened.mean(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        scale_scores = []
        for scale in range(self.n_scales):
            scaled = images
            if scale:
                side = self.image_size // (2**scale)
                scaled = F.interpolate(images, size=(side, side), mode="nearest")
            patch_scores = self.score_head(self.feature_extractor(scaled))
            scale_scores.append(self.aggregate_patch_scores(patch_scores, self.topk_ratio))
        return torch.stack(scale_scores, dim=1).mean(dim=1)


class BalancedBatchSampler(Sampler[list[int]]):
    """Yield the paper's half-normal, half-anomaly batches for a fixed step count."""

    def __init__(
        self,
        labels: NDArray,
        *,
        batch_size: int,
        steps_per_epoch: int,
        random_state: Optional[int],
    ) -> None:
        if batch_size < 2 or steps_per_epoch <= 0:
            raise ValueError("batch_size must be at least 2 and steps_per_epoch must be positive.")
        labels = np.asarray(labels).reshape(-1)
        self.normal_indices = np.flatnonzero(labels == 0)
        self.anomaly_indices = np.flatnonzero(labels == 1)
        if not self.normal_indices.size or not self.anomaly_indices.size:
            raise ValueError("Balanced batches require both normal and anomaly samples.")
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(random_state)

    def _cycle(self, indices: NDArray) -> Iterator[int]:
        while True:
            yield from (int(index) for index in self.rng.permutation(indices))

    def __iter__(self) -> Iterator[list[int]]:
        normal_stream = self._cycle(self.normal_indices)
        anomaly_stream = self._cycle(self.anomaly_indices)
        normal_count = self.batch_size // 2
        anomaly_count = self.batch_size - normal_count
        for _ in range(self.steps_per_epoch):
            yield [next(normal_stream) for _ in range(normal_count)] + [
                next(anomaly_stream) for _ in range(anomaly_count)
            ]

    def __len__(self) -> int:
        return self.steps_per_epoch


class _DevNetArrayDataset(Dataset):
    def __init__(self, images: NDArray, labels: NDArray, transform) -> None:
        self.images = np.asarray(images)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return self.transform(np.ascontiguousarray(image)), int(self.labels[index])


@register_model(
    "vision_devnet",
    tags=("vision", "deep", "devnet", "weakly-supervised", "few-shot"),
    metadata={
        "description": "Image DevNet with multi-scale top-K patch scoring and deviation loss",
        "paper": "Explainable Deep Few-shot Anomaly Detection with Deviation Networks",
        "paper_url": "https://arxiv.org/abs/2108.00462",
        "year": 2021,
        "supervision": "weakly-supervised",
        "implementation_status": "paper-image-network-aligned-no-localization",
        "paper_fidelity": "paper-adaptation",
    },
)
@register_model(
    "devnet",
    tags=("vision", "deep", "devnet", "weakly-supervised", "few-shot"),
    metadata={
        "description": "Legacy alias for the image DevNet adaptation",
        "paper": "Explainable Deep Few-shot Anomaly Detection with Deviation Networks",
        "paper_url": "https://arxiv.org/abs/2108.00462",
        "year": 2021,
        "supervision": "weakly-supervised",
        "implementation_status": "paper-image-network-aligned-no-localization",
        "paper_fidelity": "paper-adaptation",
    },
)
class DevNetDetector(BaseVisionDeepDetector):
    """Few-shot image DevNet with the paper network and training sampler."""

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        margin: float = 5.0,
        reference_size: int = 5000,
        pretrained: bool = False,
        image_size: int = 448,
        n_scales: int = 2,
        topk_ratio: float = 0.1,
        epochs: int = 50,
        batch_size: int = 48,
        steps_per_epoch: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
        gradient_clip_norm: float = 1.0,
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
        contamination: float = 0.1,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            contamination=contamination,
            device=device,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )
        self.backbone = str(backbone)
        self.backbone_name = self.backbone
        self.margin = float(margin)
        self.reference_size = int(reference_size)
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)
        self.n_scales = int(n_scales)
        self.topk_ratio = float(topk_ratio)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.scheduler_step_size = int(scheduler_step_size)
        self.scheduler_gamma = float(scheduler_gamma)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.random_state = None if random_state is None else int(random_state)
        self.model: DevNetModel
        self.optimizer_: Optional[torch.optim.Optimizer] = None
        self.scheduler_: Optional[object] = None
        self._build_model()

    def _build_model(self) -> None:
        with torch.random.fork_rng(devices=[]):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            self.model = DevNetModel(
                backbone=self.backbone,
                pretrained=self.pretrained,
                image_size=self.image_size,
                n_scales=self.n_scales,
                topk_ratio=self.topk_ratio,
            ).to(self.device)
        self.feature_extractor = self.model.feature_extractor
        self.scoring_model = self.model

    def _transform(self, *, training: bool):
        operations: list[object] = [
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if training:
            operations.append(transforms.RandomRotation(180))
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        return transforms.Compose(operations)

    @isolated_random_state_method
    def fit(self, x: NDArray, y: NDArray, **kwargs) -> "DevNetDetector":
        del kwargs
        images = np.asarray(x)
        labels = np.asarray(y).reshape(-1) if y is not None else np.asarray([])
        if len(images) != len(labels) or set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError(
                "DevNet requires labeled data with both normal (0) and anomaly (1) samples."
            )

        self._build_model()
        dataset = _DevNetArrayDataset(images, labels, self._transform(training=True))
        sampler = BalancedBatchSampler(
            labels,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            random_state=self.random_state,
        )
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

        self.optimizer_ = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler_ = torch.optim.lr_scheduler.StepLR(
            self.optimizer_,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        criterion = DeviationLoss(margin=self.margin, reference_size=self.reference_size)

        for _epoch in range(self.epochs):
            self.model.train()
            for batch_images, batch_labels in dataloader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                scores = self.model(batch_images)
                loss = criterion(scores, batch_labels)
                self.optimizer_.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer_.step()
            self.scheduler_.step()

        self.model.eval()
        self.is_fitted_ = True
        self.fitted_ = True
        self.decision_scores_ = self.decision_function(images)
        self._process_decision_scores()
        self._set_n_classes(labels, warn_on_labeled_y=False)
        return self

    @torch.no_grad()
    def predict_proba(self, x: NDArray, *, batch_size: Optional[int] = None, **kwargs) -> NDArray:
        del kwargs
        self._check_is_fitted()
        images = np.asarray(x)
        if len(images) == 0:
            return np.zeros((0,), dtype=np.float64)
        dataset = _DevNetArrayDataset(
            images,
            np.zeros(len(images), dtype=np.int64),
            self._transform(training=False),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size if batch_size is None else int(batch_size),
            shuffle=False,
            num_workers=0,
        )
        self.model.eval()
        scores = [self.model(batch.to(self.device)).cpu() for batch, _labels in dataloader]
        return torch.cat(scores).numpy().astype(np.float64, copy=False)

    def decision_function(
        self,
        x: NDArray,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> NDArray:
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        return self.predict_proba(x, batch_size=batch_size, **kwargs).reshape(-1)

    def get_feature_importance(self, x: NDArray) -> NDArray:
        """Return mean absolute gradient importance for each backbone channel."""
        self._check_is_fitted()
        images = np.asarray(x)
        if len(images) == 0:
            return np.zeros((self.feature_extractor.feature_dim,), dtype=np.float32)
        dataset = _DevNetArrayDataset(
            images,
            np.zeros(len(images), dtype=np.int64),
            self._transform(training=False),
        )
        batch = torch.stack([dataset[index][0] for index in range(len(dataset))]).to(self.device)
        self.feature_extractor.eval()
        features = self.feature_extractor(batch).detach().requires_grad_(True)
        patch_scores = self.model.score_head(features)
        scores = self.model.aggregate_patch_scores(patch_scores, self.topk_ratio)
        gradients = torch.autograd.grad(scores.sum(), features)[0]
        return gradients.abs().mean(dim=(0, 2, 3)).detach().cpu().numpy()


__all__ = [
    "BalancedBatchSampler",
    "DeviationLoss",
    "DevNetDetector",
    "DevNetModel",
    "FeatureExtractor",
]
