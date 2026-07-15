"""Compact DRAEM adaptation.

The detector keeps the defining DRAEM path, but uses smaller U-Net variants and
a simplified texture pipeline.  It is therefore an adaptation, not the exact
network and augmentation recipe released by the paper authors.

Reference:
    Zavrtanik, V., Kristan, M., & Skočaj, D. (2021).
    DRAEM-A discriminatively trained reconstruction embedding for surface anomaly detection.
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 8330-8339).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Union, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pyimgano.synthesis.perlin import perlin_noise_2d

from .baseCv import BaseVisionDeepDetector
from .deep_io import safe_torch_load
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not fitted. Call fit() first."
_MISSING = object()


logger = logging.getLogger(__name__)

ImageInput = Union[str, np.ndarray]


def _resolve_legacy_x_keyword(
    x: object, kwargs: Dict[str, object], *, method_name: str
) -> Iterable[ImageInput]:
    """Accept legacy `X=` inputs while keeping lowercase local naming."""

    legacy_x = kwargs.pop("X", _MISSING)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"{method_name}() got an unexpected keyword argument {unexpected!r}")
    if x is _MISSING:
        if legacy_x is _MISSING:
            raise TypeError(f"{method_name}() missing 1 required positional argument: 'x'")
        return cast(Iterable[ImageInput], legacy_x)
    if legacy_x is not _MISSING:
        raise TypeError(f"{method_name}() got multiple values for argument 'x'")
    return cast(Iterable[ImageInput], x)


class SimpleUNet(nn.Module):
    """Small U-Net used by the two DRAEM subnetworks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int = 32,
        output_activation: bool = False,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        if c <= 0:
            raise ValueError("base_channels must be positive")
        self.output_activation = bool(output_activation)

        # Encoder
        self.enc1 = self._conv_block(in_channels, c)
        self.enc2 = self._conv_block(c, c * 2)
        self.enc3 = self._conv_block(c * 2, c * 4)
        self.enc4 = self._conv_block(c * 4, c * 8)

        # Decoder
        self.dec4 = self._upconv_block(c * 8, c * 4)
        self.dec3 = self._upconv_block(c * 8, c * 2)
        self.dec2 = self._upconv_block(c * 4, c)
        self.dec1 = self._conv_block(c * 2, c)
        self.final = nn.Conv2d(c, out_channels, 1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _upconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Decoder with skip connections
        d4 = self.dec4(e4)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)

        output = self.final(d1)
        return torch.sigmoid(output) if self.output_activation else output


class DRAEMNetwork(nn.Module):
    """Reconstructive and discriminative subnetworks from the DRAEM pipeline."""

    def __init__(self, *, base_channels: int = 32) -> None:
        super().__init__()
        self.reconstructor = SimpleUNet(
            3,
            3,
            base_channels=base_channels,
            output_activation=True,
        )
        self.segmentor = SimpleUNet(
            6,
            2,
            base_channels=base_channels,
            output_activation=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction = self.reconstructor(x)
        logits = self.segmentor(torch.cat((reconstruction, x), dim=1))
        return reconstruction, logits


def _ssim_loss(x: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable local SSIM loss for tensors in ``[0, 1]``."""

    size = min(int(window_size), int(x.shape[-2]), int(x.shape[-1]))
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)
    coords = torch.arange(size, dtype=x.dtype, device=x.device) - (size - 1) / 2
    gaussian = torch.exp(-(coords**2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    window = window.expand(x.shape[1], 1, size, size).contiguous()
    padding = size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=x.shape[1])
    mu_y = F.conv2d(target, window, padding=padding, groups=target.shape[1])
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(x.square(), window, padding=padding, groups=x.shape[1]) - mu_x_sq
    sigma_y = (
        F.conv2d(target.square(), window, padding=padding, groups=target.shape[1]) - mu_y_sq
    )
    sigma_xy = F.conv2d(x * target, window, padding=padding, groups=x.shape[1]) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - score.mean()


def _focal_loss(logits: torch.Tensor, mask: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    target = mask[:, 0].long()
    cross_entropy = F.cross_entropy(logits, target, reduction="none")
    probability = torch.exp(-cross_entropy)
    return ((1.0 - probability) ** gamma * cross_entropy).mean()


class ImagePathDataset(Dataset):
    """DRAEM training data with Perlin-masked synthetic texture anomalies."""

    def __init__(
        self,
        image_paths: Iterable[ImageInput],
        transform=None,
        *,
        anomaly_source_images: Optional[Iterable[ImageInput]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.image_paths = list(image_paths)
        self.anomaly_source_images = (
            list(self.image_paths)
            if anomaly_source_images is None
            else list(anomaly_source_images)
        )
        if not self.anomaly_source_images:
            raise ValueError("anomaly_source_images cannot be empty")
        self.transform = transform
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _load_rgb(item: ImageInput) -> np.ndarray:
        if isinstance(item, np.ndarray):
            if item.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB image, got dtype={item.dtype}")
            if item.ndim != 3 or item.shape[2] != 3:
                raise ValueError(f"Expected shape (H,W,3), got {item.shape}")
            return np.ascontiguousarray(item)

        img_path = str(item)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self._load_rgb(self.image_paths[idx])
        source_idx = int(self.rng.integers(0, len(self.anomaly_source_images)))
        texture = self._load_rgb(self.anomaly_source_images[source_idx])

        if self.transform:
            img = self.transform(img)
            texture = self.transform(texture)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            texture = torch.from_numpy(texture).permute(2, 0, 1).float() / 255.0

        augmented, mask = self._add_synthetic_anomaly(img, texture)
        return augmented, img, mask

    def _add_synthetic_anomaly(
        self, img: torch.Tensor, texture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend an augmented texture through a randomized Perlin mask."""

        rng = self.rng
        height, width = int(img.shape[-2]), int(img.shape[-1])
        if rng.random() > 0.5:
            return img.clone(), torch.zeros((1, height, width), dtype=img.dtype)

        max_power = max(0, min(5, int(np.floor(np.log2(max(1, min(height, width)))))))
        resolution = (
            2 ** int(rng.integers(0, max_power + 1)),
            2 ** int(rng.integers(0, max_power + 1)),
        )
        noise = perlin_noise_2d((height, width), resolution, rng=rng)
        noise = np.rot90(noise, k=int(rng.integers(0, 4))).copy()
        mask = torch.from_numpy((noise > 0.5).astype(np.float32)).unsqueeze(0)

        if rng.random() > 0.5:
            texture = torch.flip(texture, dims=(2,))
        if rng.random() > 0.5:
            texture = torch.flip(texture, dims=(1,))
        texture = texture[torch.as_tensor(rng.permutation(3), dtype=torch.long)]
        gamma = float(rng.uniform(0.5, 2.0))
        texture = texture.clamp(0.0, 1.0).pow(gamma)
        brightness = float(rng.uniform(0.75, 1.25))
        texture = (texture * brightness).clamp(0.0, 1.0)

        beta = float(rng.uniform(0.0, 0.8))
        blended = (1.0 - beta) * texture + beta * img
        augmented = img * (1.0 - mask) + blended * mask
        return augmented, mask


@register_model(
    "vision_draem",
    tags=("vision", "deep", "draem", "reconstruction", "synthetic", "numpy", "pixel_map"),
    metadata={
        "description": "Compact DRAEM adaptation with reconstruction and discriminative segmentation",
        "paper": "DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection",
        "paper_url": "https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html",
        "year": 2021,
        "supervision": "self-supervised",
        "implementation_status": "compact-two-network-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
class VisionDRAEM(BaseVisionDeepDetector):
    """
    DRAEM anomaly detector using synthetic anomalies.

    Parameters
    ----------
    image_size : int, default=256
        Input image size
    epochs : int, default=100
        Number of training epochs
    batch_size : int, default=8
        Training batch size
    lr : float, default=0.0001
        Learning rate
    num_workers : int, default=0
        Number of workers for the training DataLoader.
    device : str, default='cpu'
        Device to run model on

    Examples
    --------
    >>> detector = VisionDRAEM(epochs=50, device='cuda')
    >>> detector.fit(train_images)
    >>> scores = detector.decision_function(test_images)
    >>> labels = detector.predict(test_images)  # 0=normal, 1=anomaly
    """

    def __init__(
        self,
        image_size: int = 256,
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 0.0001,
        num_workers: int = 0,
        device: str = "cpu",
        anomaly_source_images: Optional[Iterable[ImageInput]] = None,
        base_channels: int = 32,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize DRAEM detector."""
        super().__init__(random_state=random_state, **kwargs)

        if image_size < 8 or image_size % 8:
            raise ValueError("image_size must be >= 8 and divisible by 8")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")

        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.device = device
        self.anomaly_source_images = (
            None if anomaly_source_images is None else list(anomaly_source_images)
        )
        self.base_channels = int(base_channels)
        self.random_state = int(random_state)
        self._is_fitted = False

        # Build model
        self.model = DRAEMNetwork(base_channels=self.base_channels)
        self.model.to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        logger.info(
            "Initialized DRAEM with image_size=%d, epochs=%d, batch_size=%d, device=%s",
            image_size,
            epochs,
            batch_size,
            device,
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        if not self._is_fitted or not hasattr(self, "threshold_"):
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        raw_state = self.model.state_dict()
        model_state_dict: dict[str, object] = {}
        for key, value in dict(raw_state).items():
            detach = getattr(value, "detach", None)
            cpu = getattr(value, "cpu", None)
            if callable(detach) and callable(cpu):
                model_state_dict[str(key)] = detach().cpu()
            else:
                model_state_dict[str(key)] = value

        torch.save(
            {
                "schema_version": 2,
                "model_type": "draem_two_network",
                "model_state_dict": model_state_dict,
                "decision_scores_": torch.as_tensor(
                    np.asarray(self.decision_scores_, dtype=np.float64),
                    dtype=torch.float64,
                ),
                "threshold_": float(self.threshold_),
                "is_fitted": True,
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        state = safe_torch_load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Invalid VisionDRAEM checkpoint payload.")
        if state.get("schema_version") != 2 or state.get("model_type") != "draem_two_network":
            raise ValueError(
                "Unsupported legacy DRAEM checkpoint: expected the two-network schema version 2."
            )

        model_state_dict = state.get("model_state_dict", None)
        if not isinstance(model_state_dict, dict):
            raise ValueError("VisionDRAEM checkpoint is missing model_state_dict.")

        self.model.load_state_dict(dict(model_state_dict), strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.decision_scores_ = np.asarray(state["decision_scores_"], dtype=np.float64)
        self.threshold_ = float(state["threshold_"])
        self._is_fitted = bool(state.get("is_fitted", True))

    def fit(
        self,
        x: object = _MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionDRAEM":
        """
        Train DRAEM on normal images.

        Parameters
        ----------
        X : iterable of str
            Paths to normal training images
        y : array-like, optional
            Ignored

        Returns
        -------
        self : VisionDRAEM
        """
        del y
        logger.info("Training DRAEM detector")

        x_iter = _resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        x_list = list(x_iter)
        if not x_list:
            raise ValueError("Training set cannot be empty")

        # Create dataset
        dataset = ImagePathDataset(
            x_list,
            transform=self.transform,
            anomaly_source_images=self.anomaly_source_images,
            rng=np.random.default_rng(self.random_state),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=str(self.device).startswith("cuda"),
        )

        # Setup optimizer
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)

        # Training loop
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for augmented, original, anomaly_mask in dataloader:
                augmented = augmented.to(self.device)
                original = original.to(self.device)
                anomaly_mask = anomaly_mask.to(self.device)

                # Forward pass
                reconstructed, mask_logits = self.model(augmented)

                loss = (
                    F.mse_loss(reconstructed, original)
                    + _ssim_loss(reconstructed, original)
                    + _focal_loss(mask_logits, anomaly_mask)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.info("Epoch %d/%d, Loss: %.6f", epoch + 1, self.epochs, avg_loss)

        logger.info("DRAEM training completed")

        # Mark as fitted and compute training scores to establish a threshold.
        self._is_fitted = True
        self.decision_scores_ = self.decision_function(x_list)
        self._process_decision_scores()

        return self

    def predict(
        self,
        x: object = _MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        """
        Predict binary anomaly labels for test images.

        Parameters
        ----------
        X : iterable of str
            Paths to test images

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Binary labels (0 = normal, 1 = anomaly)
        """
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )

        if not self._is_fitted or not hasattr(self, "threshold_"):
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        x_iter = _resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        scores = self.decision_function(x_iter)
        return (scores >= self.threshold_).astype(int)

    def _load_item_rgb(self, item: ImageInput) -> np.ndarray:  # pragma: no cover
        """Load an input item as a contiguous RGB uint8 numpy array."""

        if isinstance(item, np.ndarray):
            if item.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB image, got dtype={item.dtype}")
            if item.ndim != 3 or item.shape[2] != 3:
                raise ValueError(f"Expected shape (H,W,3), got {item.shape}")
            return np.ascontiguousarray(item)

        img = cv2.imread(str(item))
        if img is None:
            raise ValueError(f"Failed to load image: {item}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _anomaly_map_tensor(self, item: ImageInput) -> tuple[np.ndarray, tuple[int, int]]:
        img = self._load_item_rgb(item)
        original_size = (int(img.shape[1]), int(img.shape[0]))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        _, logits = self.model(img_tensor)
        anomaly_map = torch.softmax(logits, dim=1)[:, 1:2]
        return anomaly_map.squeeze(0).squeeze(0).cpu().numpy(), original_size

    def _score_item(self, item: ImageInput) -> float:
        anomaly_map, _ = self._anomaly_map_tensor(item)
        map_tensor = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0)
        smoothed = F.avg_pool2d(map_tensor, kernel_size=21, stride=1, padding=10)
        return float(smoothed.max().item())

    def decision_function(
        self,
        x: object = _MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        """Compute anomaly scores."""
        # This detector scores one image at a time. Keep `batch_size` for
        # interface compatibility with BaseDeepLearningDetector.
        if batch_size is not None:
            batch_size_int = int(batch_size)
            if batch_size_int <= 0:
                raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")

        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        self.model.eval()

        x_iter = _resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        x_list = list(x_iter)
        scores = np.zeros(len(x_list), dtype=np.float64)

        logger.info("Computing anomaly scores for %d images", len(x_list))

        with torch.no_grad():
            for idx, item in enumerate(x_list):
                scores[idx] = self._score_item(item)

        return scores

    def get_anomaly_map(self, image_path: ImageInput) -> NDArray:
        """Generate pixel-level anomaly heatmap."""
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)

        self.model.eval()
        with torch.no_grad():
            anomaly_map, original_size = self._anomaly_map_tensor(image_path)

        anomaly_map = anomaly_map.astype(np.float32, copy=False)
        anomaly_map = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_CUBIC)
        return anomaly_map

    def predict_anomaly_map(self, x: object = _MISSING, **kwargs: object) -> NDArray:
        """Generate pixel-level anomaly maps for a batch of images."""
        x_iter = _resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        maps = [self.get_anomaly_map(path) for path in x_iter]
        return np.stack(maps)
