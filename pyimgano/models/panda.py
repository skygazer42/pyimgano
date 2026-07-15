"""PANDA-Early image anomaly detection.

Paper: "PANDA: Adapting Pretrained Features for Anomaly Detection and
Segmentation" (CVPR 2021).

This module implements the paper's image-level fixed-iteration early-stopping
variant.  PANDA-EWC needs an external ImageNet Fisher diagonal and PANDA-SES
needs a checkpoint ensemble; neither is silently approximated here.  The
paper's separate SPADE segmentation path is implemented by ``vision_spade``.
"""

from __future__ import annotations

import logging
from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import functional as tv_functional

from pyimgano.models._imagenet_preprocess import preprocess_imagenet_batch
from pyimgano.utils.random_state import isolated_random_state_method
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._image_batch import coerce_rgb_image_batch
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


class PANDAEncoder(nn.Module):
    """ImageNet ResNet pooled features with only blocks 3 and 4 trainable."""

    def __init__(
        self,
        backbone: str = "resnet152",
        *,
        pretrained: bool = True,
        weights_name: str = "IMAGENET1K_V1",
    ) -> None:
        super().__init__()
        model, _ = load_torchvision_model(
            backbone,
            pretrained=pretrained,
            weights_name=weights_name if pretrained else None,
        )
        required = (
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
        )
        if any(not hasattr(model, name) for name in required):
            raise ValueError(f"PANDA requires a torchvision ResNet backbone, got {backbone!r}.")

        self.backbone = model
        self.feature_dim = int(model.fc.in_features)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for block in (model.layer3, model.layer4):
            for parameter in block.parameters():
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return torch.flatten(model.avgpool(x), 1)


@register_model(
    "vision_panda",
    tags=("vision", "deep", "panda", "feature-adaptation", "knn", "paper"),
    metadata={
        "description": "PANDA-Early ResNet152 compactness adaptation with squared-L2 2-NN scoring",
        "paper": "PANDA: Adapting Pretrained Features for Anomaly Detection and Segmentation",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2021/html/Reiss_PANDA_Adapting_Pretrained_Features_for_Anomaly_Detection_and_Segmentation_CVPR_2021_paper.html",
        "year": 2021,
        "implementation_status": "paper-image-panda-early-path-aligned",
        "paper_fidelity": "core-aligned",
        "type": "feature-adaptation",
    },
)
class VisionPANDA(BaseVisionDeepDetector):
    """Paper-aligned PANDA-Early image-level detector.

    Defaults reproduce the published fixed-iteration path: ImageNet-pretrained
    ResNet152, blocks 3/4 fine-tuning, 2,300 minibatches of 32 images, SGD with
    ``lr=1e-2``, momentum ``0.9``, weight decay ``5e-5``, gradient clipping at
    ``1e-3``, and summed squared distances to two nearest normal features.

    ``pretrained=False`` and non-default backbones are useful for tests but are
    not paper configurations.
    """

    def __init__(
        self,
        backbone: str = "resnet152",
        *,
        pretrained: bool = True,
        weights_name: str = "IMAGENET1K_V1",
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        training_steps: int = 2300,
        momentum: float = 0.9,
        weight_decay: float = 5e-5,
        grad_clip_norm: float = 1e-3,
        n_neighbors: int = 2,
        resize_size: int = 256,
        image_size: int = 224,
        contamination: float = 0.1,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(training_steps) <= 0:
            raise ValueError("training_steps must be positive.")
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be positive.")
        if int(image_size) <= 0 or int(resize_size) < int(image_size):
            raise ValueError("resize_size must be at least image_size > 0.")
        if float(learning_rate) <= 0 or float(grad_clip_norm) <= 0:
            raise ValueError("learning_rate and grad_clip_norm must be positive.")

        resolved_device = device
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        super().__init__(
            contamination=contamination,
            batch_size=int(batch_size),
            device=resolved_device,
            random_state=random_state,
            **kwargs,
        )
        self.backbone = str(backbone)
        self.pretrained = bool(pretrained)
        self.weights_name = str(weights_name)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.training_steps = int(training_steps)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = float(grad_clip_norm)
        self.n_neighbors = int(n_neighbors)
        self.resize_size = int(resize_size)
        self.image_size = int(image_size)
        self.random_state = random_state

        self.encoder_: PANDAEncoder | None = None
        self.center_: torch.Tensor | None = None
        self.memory_bank_: torch.Tensor | None = None
        self.is_fitted_ = False

    def _preprocess(self, x: object) -> torch.Tensor:
        images = coerce_rgb_image_batch(x)
        cropped = [
            np.asarray(
                tv_functional.center_crop(
                    tv_functional.resize(tv_functional.to_pil_image(image), self.resize_size),
                    [self.image_size, self.image_size],
                )
            )
            for image in images
        ]
        return preprocess_imagenet_batch(np.stack(cropped))

    def _extract_features(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.encoder_ is None:
            raise RuntimeError("PANDA encoder is not initialized.")
        self.encoder_.eval()  # The reference training path keeps BatchNorm statistics fixed.
        features: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                batch = tensor[start : start + self.batch_size].to(self.device)
                features.append(self.encoder_(batch).detach().cpu())
        return torch.cat(features, dim=0)

    def _score_features(
        self,
        features: torch.Tensor,
        *,
        exclude_same_index: bool = False,
    ) -> NDArray[np.float32]:
        if self.memory_bank_ is None:
            raise RuntimeError("PANDA memory bank is not initialized.")
        gallery = self.memory_bank_.to(self.device)
        if self.n_neighbors > len(gallery) - int(exclude_same_index):
            raise ValueError("Not enough normal samples for the configured n_neighbors.")

        scores: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                query = features[start : start + self.batch_size].to(self.device)
                distances = (
                    query.square().sum(dim=1, keepdim=True)
                    + gallery.square().sum(dim=1).unsqueeze(0)
                    - 2.0 * query @ gallery.T
                ).clamp_min_(0.0)
                if exclude_same_index:
                    rows = torch.arange(len(query), device=self.device)
                    distances[rows, rows + start] = torch.inf
                scores.append(
                    distances.topk(self.n_neighbors, dim=1, largest=False).values.sum(dim=1).cpu()
                )
        return torch.cat(scores).numpy().astype(np.float32, copy=False)

    @isolated_random_state_method
    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionPANDA":
        values = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        tensor = self._preprocess(values)
        if len(tensor) <= self.n_neighbors:
            raise ValueError(
                f"PANDA needs more than n_neighbors={self.n_neighbors} normal samples."
            )

        self.encoder_ = PANDAEncoder(
            self.backbone,
            pretrained=self.pretrained,
            weights_name=self.weights_name,
        ).to(self.device)
        self.model = self.encoder_
        initial_features = self._extract_features(tensor)
        self.center_ = initial_features.mean(dim=0).to(self.device)

        trainable = [
            parameter for parameter in self.encoder_.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("PANDA backbone exposes no trainable layer3/layer4 parameters.")
        optimizer = torch.optim.SGD(
            trainable,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.encoder_.eval()
        step = 0
        last_loss = 0.0
        while step < self.training_steps:
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                features = self.encoder_(batch)
                loss = F.mse_loss(features, self.center_.expand_as(features))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip_norm)
                optimizer.step()
                step += 1
                last_loss = float(loss.detach().cpu())
                if step >= self.training_steps:
                    break

        logger.info("PANDA-Early finished %d minibatches; loss=%.6f", step, last_loss)
        self.training_steps_completed_ = step
        self.training_loss_ = last_loss
        self.memory_bank_ = self._extract_features(tensor)
        self.decision_scores_ = self._score_features(
            self.memory_bank_, exclude_same_index=True
        ).astype(np.float64)
        self._process_decision_scores()
        self._set_n_classes(y)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        self._check_is_fitted()
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        return self._score_features(self._extract_features(self._preprocess(values)))

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        values = cast(object, resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
        if batch_size is None:
            return self.predict(values)
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        old_batch_size = self.batch_size
        try:
            self.batch_size = int(batch_size)
            return self.predict(values)
        finally:
            self.batch_size = old_batch_size


__all__ = ["PANDAEncoder", "VisionPANDA"]
