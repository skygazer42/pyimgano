from __future__ import annotations

"""
CutPaste: Self-Supervised Learning for Anomaly Detection and Localization.

Paper: https://arxiv.org/abs/2104.04015
Conference: CVPR 2021

CutPaste cuts and pastes rectangular patches within an image, creating
synthetic anomalies for self-supervised representation learning.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tv_functional

from pyimgano.utils.random_state import check_random_state, isolated_random_state_method
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._image_batch import coerce_rgb_image_batch
from .baseCv import BaseVisionDeepDetector
from .deep_io import export_module_state_dict, safe_torch_load
from .registry import register_model

logger = logging.getLogger(__name__)


class CutPasteAugmentation:
    """CutPaste augmentation strategies."""

    def __init__(
        self,
        area_ratio: Tuple[float, float] = (0.02, 0.15),
        aspect_ratio: Tuple[float, float] = (0.3, 1 / 0.3),
        type: str = "normal",  # "normal", "scar", "3way"
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize CutPaste augmentation.

        Args:
            area_ratio: Range of area ratio for cut patch.
            aspect_ratio: Range of aspect ratio for cut patch.
            type: Type of CutPaste ("normal", "scar", "3way").
        """
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.type = type
        self.rng = check_random_state(rng)

    @staticmethod
    def _sample_location(rng: np.random.Generator, full: int, extent: int) -> int:
        """Sample a start coordinate while keeping the full patch in-frame."""

        return int(rng.integers(0, full - extent + 1))

    def _color_jitter(self, patch: NDArray, intensity: float = 0.1) -> NDArray:
        """Apply the paper's randomly ordered patch ColorJitter operations."""

        original_dtype = patch.dtype
        scale = 255.0 if np.issubdtype(original_dtype, np.integer) or patch.max() > 1.0 else 1.0
        tensor = (
            torch.from_numpy(np.ascontiguousarray(patch))
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            / scale
        )
        operations = [
            (tv_functional.adjust_brightness, float(self.rng.uniform(1 - intensity, 1 + intensity))),
            (tv_functional.adjust_contrast, float(self.rng.uniform(1 - intensity, 1 + intensity))),
            (tv_functional.adjust_saturation, float(self.rng.uniform(1 - intensity, 1 + intensity))),
            (tv_functional.adjust_hue, float(self.rng.uniform(-intensity, intensity))),
        ]
        for index in self.rng.permutation(len(operations)):
            operation, factor = operations[int(index)]
            tensor = operation(tensor, factor)

        result = tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * scale
        if np.issubdtype(original_dtype, np.integer):
            result = np.rint(result)
        return result.astype(original_dtype, copy=False)

    @staticmethod
    def _rotate_patch(patch: NDArray, angle: float) -> tuple[NDArray, NDArray]:
        """Rotate a scar with expansion and return its alpha-like paste mask."""

        patch_h, patch_w = patch.shape[:2]
        matrix = cv2.getRotationMatrix2D((patch_w / 2.0, patch_h / 2.0), angle, 1.0)
        cosine = abs(float(matrix[0, 0]))
        sine = abs(float(matrix[0, 1]))
        rotated_w = max(1, int(np.ceil(patch_h * sine + patch_w * cosine)))
        rotated_h = max(1, int(np.ceil(patch_h * cosine + patch_w * sine)))
        matrix[0, 2] += rotated_w / 2.0 - patch_w / 2.0
        matrix[1, 2] += rotated_h / 2.0 - patch_h / 2.0
        rotated = cv2.warpAffine(
            patch,
            matrix,
            (rotated_w, rotated_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = cv2.warpAffine(
            np.full((patch_h, patch_w), 255, dtype=np.uint8),
            matrix,
            (rotated_w, rotated_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return rotated, mask

    def __call__(self, image: NDArray) -> NDArray:
        """Apply CutPaste augmentation.

        Args:
            image: Input image (H, W, C).

        Returns:
            Augmented image.
        """
        if self.type == "normal":
            return self.cutpaste_normal(image)
        elif self.type == "scar":
            return self.cutpaste_scar(image)
        elif self.type == "3way":
            # 3-way classification: normal, normal cutpaste, scar cutpaste
            if self.rng.random() < 0.5:
                return self.cutpaste_normal(image)
            else:
                return self.cutpaste_scar(image)
        else:
            raise ValueError(f"Unknown CutPaste type: {self.type}")

    def cutpaste_normal(self, image: NDArray) -> NDArray:
        """Apply normal CutPaste augmentation.

        Cuts a rectangular patch and pastes it at a random location.
        """
        h, w = image.shape[:2]
        rng = self.rng

        # Sample patch size
        area = h * w
        target_area = rng.uniform(*self.area_ratio) * area
        if rng.random() < 0.5:
            aspect = rng.uniform(self.aspect_ratio[0], 1.0)
        else:
            aspect = rng.uniform(1.0, self.aspect_ratio[1])

        patch_w = max(1, int(np.sqrt(target_area * aspect)))
        patch_h = max(1, int(np.sqrt(target_area / aspect)))

        # Ensure patch fits in image
        patch_h = min(patch_h, h)
        patch_w = min(patch_w, w)

        # Random source location
        src_y = self._sample_location(rng, h, patch_h)
        src_x = self._sample_location(rng, w, patch_w)

        # Random target location
        dst_y = self._sample_location(rng, h, patch_h)
        dst_x = self._sample_location(rng, w, patch_w)

        # Cut and paste
        patch = image[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()

        patch = self._color_jitter(patch)

        result = image.copy()
        result[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = patch

        return result

    def cutpaste_scar(self, image: NDArray) -> NDArray:
        """Apply scar CutPaste augmentation.

        Cuts a thin elongated patch and pastes it at a random location.
        """
        h, w = image.shape[:2]
        rng = self.rng

        # Supplemental A.2: width [2,16], length [10,25], angle (-45,45).
        patch_w = min(int(rng.integers(2, 17)), w)
        patch_h = min(int(rng.integers(10, 26)), h)

        # Random source and target
        src_y = self._sample_location(rng, h, patch_h)
        src_x = self._sample_location(rng, w, patch_w)

        # Cut and paste
        patch = image[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()

        patch = self._color_jitter(patch)
        patch, mask = self._rotate_patch(patch, float(rng.uniform(-45.0, 45.0)))
        rotated_h, rotated_w = patch.shape[:2]
        if rotated_h > h or rotated_w > w:
            scale = min(h / rotated_h, w / rotated_w)
            rotated_w = max(1, int(np.floor(rotated_w * scale)))
            rotated_h = max(1, int(np.floor(rotated_h * scale)))
            patch = cv2.resize(patch, (rotated_w, rotated_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (rotated_w, rotated_h), interpolation=cv2.INTER_NEAREST)
        dst_y = self._sample_location(rng, h, rotated_h)
        dst_x = self._sample_location(rng, w, rotated_w)

        result = image.copy()
        target = result[dst_y : dst_y + rotated_h, dst_x : dst_x + rotated_w]
        target[mask > 0] = patch[mask > 0]

        return result


class CutPasteDataset(Dataset):
    """Dataset for CutPaste training."""

    def __init__(
        self,
        images: NDArray,
        transform=None,
        augment_type: str = "3way",
        random_state: int | np.random.Generator | None = None,
    ):
        """Initialize dataset.

        Args:
            images: Array of images (N, H, W, C).
            transform: Transform to apply to images.
            augment_type: Type of CutPaste augmentation.
            random_state: Local random generator or seed for CutPaste geometry.
        """
        self.images = images
        self.transform = transform
        self.augment_type = augment_type
        rng = check_random_state(random_state)
        self.normal_augmenter = CutPasteAugmentation(type="normal", rng=rng)
        self.scar_augmenter = CutPasteAugmentation(type="scar", rng=rng)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Original image (label 0)
        original = image.copy()

        variants = [original]
        if self.augment_type == "3way":
            variants.extend([self.normal_augmenter(image), self.scar_augmenter(image)])
        elif self.augment_type == "normal":
            variants.append(self.normal_augmenter(image))
        elif self.augment_type == "scar":
            variants.append(self.scar_augmenter(image))
        else:
            raise ValueError(f"Unknown CutPaste type: {self.augment_type}")

        if self.transform:
            variants = [self.transform(item) for item in variants]

        return torch.stack(variants), torch.arange(len(variants), dtype=torch.long)


class ProjectionHead(nn.Module):
    """Projection head for CutPaste."""

    def __init__(self, in_features: int, hidden_dim: int = 512, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@register_model(
    "vision_cutpaste",
    tags=("vision", "deep", "cutpaste", "self-supervised", "cvpr2021"),
    metadata={
        "description": "CutPaste - self-supervised anomaly detection via synthetic cut/paste (CVPR 2021)",
        "paper": "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html",
        "year": 2021,
        "supervision": "self-supervised",
        "implementation_status": "native-core-augmentation-and-classification",
        "paper_fidelity": "core-aligned",
    },
)
@register_model(
    "cutpaste",
    tags=("vision", "deep", "cutpaste", "self-supervised", "cvpr2021"),
    metadata={
        "description": "CutPaste (legacy alias) - self-supervised anomaly detection via synthetic cut/paste",
        "paper": "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html",
        "year": 2021,
        "supervision": "self-supervised",
        "implementation_status": "native-core-augmentation-and-classification",
        "paper_fidelity": "core-aligned",
    },
)
class CutPasteDetector(BaseVisionDeepDetector):
    """CutPaste anomaly detector.

    Self-supervised learning using synthetic anomalies created by
    cutting and pasting image patches.

    Args:
        backbone: Backbone architecture ("resnet18", "resnet50", "efficientnet").
        embedding_dim: Dimension of feature embeddings.
        augment_type: Type of CutPaste ("normal", "scar", "3way").
        pretrained: Whether to use pretrained backbone.
        freeze_backbone: Whether to freeze backbone during training.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        device: Device to use ("cuda" or "cpu").

    References:
        Li et al. "CutPaste: Self-Supervised Learning for Anomaly Detection
        and Localization." CVPR 2021.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 512,
        augment_type: str = "3way",
        pretrained: bool = False,
        freeze_backbone: bool = False,
        epochs: int = 256,
        batch_size: int = 96,
        learning_rate: float = 0.03,
        image_size: int = 256,
        steps_per_epoch: int = 256,
        translation_ratio: float = 0.05,
        global_jitter: float = 0.1,
        random_state: int = 42,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.augment_type = augment_type
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.image_size = int(image_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.translation_ratio = float(translation_ratio)
        self.global_jitter = float(global_jitter)
        self.random_state = int(random_state)

        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if self.augment_type not in {"normal", "scar", "3way"}:
            raise ValueError("augment_type must be 'normal', 'scar', or '3way'")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        if not 0.0 <= self.translation_ratio <= 1.0:
            raise ValueError("translation_ratio must be in [0, 1]")
        if not 0.0 <= self.global_jitter <= 0.5:
            raise ValueError("global_jitter must be in [0, 0.5]")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

    @isolated_random_state_method
    def _build_model(self):
        """Build the CutPaste model."""
        # Load backbone
        if self.backbone_name == "resnet18":
            self.backbone, _ = load_torchvision_model(
                "resnet18",
                pretrained=bool(self.pretrained),
            )
            feature_dim = 512
        elif self.backbone_name == "resnet50":
            self.backbone, _ = load_torchvision_model(
                "resnet50",
                pretrained=bool(self.pretrained),
            )
            feature_dim = 2048
        elif self.backbone_name == "wide_resnet50":
            self.backbone, _ = load_torchvision_model(
                "wide_resnet50",
                pretrained=bool(self.pretrained),
            )
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # The paper trains an MLP projection head followed by the final
        # binary/three-way classification layer.
        num_classes = 2 if self.augment_type != "3way" else 3
        self.projection_head = ProjectionHead(
            feature_dim,
            hidden_dim=self.embedding_dim,
            out_features=num_classes,
        )

        self.backbone.to(self.device)
        self.projection_head.to(self.device)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Feature tensor (B, D).
        """
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)
        return features

    @isolated_random_state_method
    def fit(self, x: NDArray, y: Optional[NDArray] = None, **kwargs):
        """Train the CutPaste detector.

        Args:
            X: Training images (N, H, W, C) or (N, C, H, W).
            y: Not used (unsupervised).
        """
        del kwargs
        x = coerce_rgb_image_batch(x)
        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x.astype(np.float32) / 255.0

        # Create dataset
        dataset = CutPasteDataset(
            x,
            transform=self._get_transform(training=True),
            augment_type=self.augment_type,
            random_state=self.random_state,
        )

        num_variants = 3 if self.augment_type == "3way" else 2
        source_batch_size = max(1, self.batch_size // num_variants)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        dataloader = DataLoader(
            dataset,
            batch_size=source_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

        # Optimizer and loss
        optimizer = torch.optim.SGD(
            list(self.backbone.parameters()) + list(self.projection_head.parameters()),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.00003,
        )

        criterion = nn.CrossEntropyLoss()
        total_steps = self.epochs * self.steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=0.0,
        )

        # Training loop
        self.backbone.train()
        self.projection_head.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            loader_iter = iter(dataloader)
            for _ in range(self.steps_per_epoch):
                try:
                    images, labels = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(dataloader)
                    images, labels = next(loader_iter)
                images = images.flatten(0, 1).to(self.device)
                labels = labels.flatten().to(self.device)

                # Forward
                features = self._extract_features(images)
                logits = self.projection_head(features)

                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Statistics
                epoch_loss += loss.detach().item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            if (epoch + 1) % 10 == 0:
                acc = 100.0 * correct / total
                logger.info(
                    "Epoch [%d/%d] Loss: %.4f Acc: %.2f%%",
                    epoch + 1,
                    self.epochs,
                    epoch_loss / self.steps_per_epoch,
                    acc,
                )

        # Switch to eval mode
        self.backbone.eval()
        self.projection_head.eval()
        self.training_steps_ = total_steps
        self.final_learning_rate_ = float(optimizer.param_groups[0]["lr"])

        # Build reference statistics from training data
        self._build_reference(x)
        self.decision_scores_ = self.decision_function(x)
        self._process_decision_scores()
        self._set_n_classes(y)
        return self

    def _build_reference(self, x: NDArray):
        """Build reference feature distribution.

        Args:
            X: Training images.
        """
        self.backbone.eval()

        features_list = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch = x[i : i + self.batch_size]
                batch_tensor = self._preprocess(batch).to(self.device)
                features = self._extract_features(batch_tensor)
                features_list.append(features.cpu().numpy())

        features = np.vstack(features_list)

        density = LedoitWolf().fit(features)
        self.reference_mean = np.asarray(density.location_, dtype=np.float32)
        self.reference_precision = np.asarray(density.precision_, dtype=np.float32)

    def _score_images(self, x: NDArray, **kwargs) -> NDArray:
        """Predict anomaly scores.

        Args:
            X: Test images (N, H, W, C) or (N, C, H, W).

        Returns:
            Anomaly scores for each sample.
        """
        del kwargs
        x = coerce_rgb_image_batch(x)
        if x.max() > 1.0:
            x = x.astype(np.float32) / 255.0

        self.backbone.eval()
        scores = []

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch = x[i : i + self.batch_size]
                batch_tensor = self._preprocess(batch).to(self.device)

                # Extract features
                features = self._extract_features(batch_tensor).cpu().numpy()

                delta = features - self.reference_mean
                dist = np.sqrt(
                    np.maximum(
                        np.einsum("bi,ij,bj->b", delta, self.reference_precision, delta),
                        0.0,
                    )
                )

                scores.append(dist)

        return np.concatenate(scores)

    def decision_function(self, x: NDArray, batch_size: int | None = None, **kwargs) -> NDArray:
        del batch_size
        return np.asarray(self._score_images(x, **kwargs), dtype=np.float64).reshape(-1)

    def save_checkpoint(self, path: str | Path) -> Path:
        if not hasattr(self, "reference_mean") or not hasattr(self, "reference_precision"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        from pyimgano.utils.optional_deps import require

        torch_runtime = require("torch", extra="torch", purpose="save CutPaste checkpoint")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "detector": "vision_cutpaste",
            "config": {
                "contamination": float(self.contamination),
                "backbone": str(self.backbone_name),
                "embedding_dim": int(self.embedding_dim),
                "augment_type": str(self.augment_type),
                "pretrained": bool(self.pretrained),
                "freeze_backbone": bool(self.freeze_backbone),
                "epochs": int(self.epochs),
                "batch_size": int(self.batch_size),
                "learning_rate": float(self.learning_rate),
                "image_size": int(self.image_size),
                "steps_per_epoch": int(self.steps_per_epoch),
                "translation_ratio": float(self.translation_ratio),
                "global_jitter": float(self.global_jitter),
                "random_state": int(self.random_state),
                "device": str(self.device),
            },
            "state": {
                "backbone_state_dict": export_module_state_dict(self.backbone),
                "projection_head_state_dict": export_module_state_dict(self.projection_head),
                "reference_mean": np.asarray(self.reference_mean, dtype=np.float32),
                "reference_precision": np.asarray(self.reference_precision, dtype=np.float32),
                "decision_scores_": (
                    np.asarray(self.decision_scores_, dtype=np.float64)
                    if getattr(self, "decision_scores_", None) is not None
                    else None
                ),
                "threshold_": (
                    float(self.threshold_)
                    if getattr(self, "threshold_", None) is not None
                    else None
                ),
            },
        }
        torch_runtime.save(payload, out_path)
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Invalid CutPaste checkpoint payload: expected a dict.")
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError(
                "Unsupported legacy CutPaste checkpoint: expected the MLP-head schema version 2."
            )
        if str(payload.get("detector", "")) not in {"vision_cutpaste", "cutpaste"}:
            raise ValueError("Invalid CutPaste checkpoint payload: detector marker mismatch.")

        config = payload.get("config", None)
        if not isinstance(config, dict):
            raise ValueError("Invalid CutPaste checkpoint payload: missing config.")

        self.contamination = float(config.get("contamination", self.contamination))
        self.backbone_name = str(config.get("backbone", self.backbone_name))
        self.embedding_dim = int(config.get("embedding_dim", self.embedding_dim))
        self.augment_type = str(config.get("augment_type", self.augment_type))
        self.pretrained = bool(config.get("pretrained", self.pretrained))
        self.freeze_backbone = bool(config.get("freeze_backbone", self.freeze_backbone))
        self.epochs = int(config.get("epochs", self.epochs))
        self.batch_size = int(config.get("batch_size", self.batch_size))
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))
        self.image_size = int(config.get("image_size", self.image_size))
        self.steps_per_epoch = int(config.get("steps_per_epoch", self.steps_per_epoch))
        self.translation_ratio = float(
            config.get("translation_ratio", self.translation_ratio)
        )
        self.global_jitter = float(config.get("global_jitter", self.global_jitter))
        self.random_state = int(config.get("random_state", self.random_state))
        self.device = torch.device(str(config.get("device", self.device)))
        self._build_model()

        state = payload.get("state", None)
        if not isinstance(state, dict):
            raise ValueError("Invalid CutPaste checkpoint payload: missing state.")

        backbone_state = state.get("backbone_state_dict", None)
        projection_head_state = state.get("projection_head_state_dict", None)
        if not isinstance(backbone_state, dict) or not isinstance(projection_head_state, dict):
            raise ValueError("Invalid CutPaste checkpoint payload: missing model state.")

        self.backbone.load_state_dict(dict(backbone_state), strict=False)
        self.projection_head.load_state_dict(dict(projection_head_state), strict=False)
        self.backbone.to(self.device)
        self.projection_head.to(self.device)
        self.backbone.eval()
        self.projection_head.eval()

        self.reference_mean = np.asarray(state["reference_mean"], dtype=np.float32)
        if state.get("reference_precision", None) is not None:
            self.reference_precision = np.asarray(state["reference_precision"], dtype=np.float32)
        elif state.get("reference_std", None) is not None:
            std = np.asarray(state["reference_std"], dtype=np.float32)
            self.reference_precision = np.diag(1.0 / np.square(std + 1e-8))
        else:
            raise ValueError("CutPaste checkpoint is missing Gaussian reference statistics.")
        if state.get("decision_scores_", None) is not None:
            self.decision_scores_ = np.asarray(state["decision_scores_"], dtype=np.float64)
        if state.get("threshold_", None) is not None:
            self.threshold_ = float(state["threshold_"])

    def _get_transform(self, *, training: bool = False):
        """Build the 256px paper preprocessing, with train-only invariance jitter."""
        from torchvision import transforms

        operations = [
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if training:
            operations.extend(
                [
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(self.translation_ratio, self.translation_ratio),
                    ),
                    transforms.ColorJitter(
                        brightness=self.global_jitter,
                        contrast=self.global_jitter,
                        saturation=self.global_jitter,
                        hue=self.global_jitter,
                    ),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transforms.Compose(operations)

    def _preprocess(self, images: NDArray) -> torch.Tensor:
        """Preprocess images for inference.

        Args:
            images: Input images (N, H, W, C).

        Returns:
            Preprocessed tensor (N, C, H, W).
        """
        transform = self._get_transform(training=False)

        batch = []
        for img in images:
            img_tensor = transform(img)
            batch.append(img_tensor)

        return torch.stack(batch)
