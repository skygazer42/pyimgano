from __future__ import annotations

"""DifferNet normalizing-flow anomaly detector.

Implements the core architecture from "Same Same But DifferNet" (WACV 2021):
multi-scale AlexNet features, an invertible density model, and likelihood
aggregation over fixed image transformations.
"""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .cflow import ConditionalFlow
from .deep_io import export_module_state_dict, safe_torch_load
from .registry import register_model


class DifferNetNetwork(nn.Module):
    """Frozen multi-scale AlexNet encoder followed by a vector normalizing flow."""

    def __init__(
        self, *, pretrained: bool, image_size: int, n_scales: int, n_flow_steps: int
    ) -> None:
        super().__init__()
        if n_scales <= 0:
            raise ValueError("n_scales must be positive.")

        alexnet, _ = load_torchvision_model("alexnet", pretrained=bool(pretrained))
        self.feature_extractor = alexnet.features
        self.image_size = int(image_size)
        self.n_scales = int(n_scales)
        self.feature_dim = 256 * self.n_scales
        self.flow = ConditionalFlow(self.feature_dim, condition_dim=0, n_flows=n_flow_steps)

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        self.feature_extractor.eval()

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(
            images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        features = []
        for scale in range(self.n_scales):
            scaled = images
            if scale:
                scaled = F.interpolate(
                    images,
                    scale_factor=1.0 / (2**scale),
                    mode="bilinear",
                    align_corners=False,
                )
            encoded = self.feature_extractor(scaled)
            features.append(encoded.mean(dim=(-2, -1)))
        return torch.cat(features, dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.extract_features(images)
        condition = features.new_empty((features.shape[0], 0))
        return self.flow(features, condition)


@register_model(
    "vision_differnet",
    tags=("vision", "deep", "flow"),
    metadata={
        "description": "DifferNet adaptation with multi-scale features, flow density, and quarter-turn aggregation",
        "paper": "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows",
        "paper_url": "https://arxiv.org/abs/2008.12577",
        "year": 2021,
        "supervision": "one-class",
        "implementation_status": "native-flow-with-simplified-transform-ensemble",
        "paper_fidelity": "paper-adaptation",
    },
)
@register_model(
    "differnet",
    tags=("vision", "deep", "flow"),
    metadata={
        "description": "Legacy alias for the DifferNet flow adaptation",
        "paper": "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows",
        "paper_url": "https://arxiv.org/abs/2008.12577",
        "year": 2021,
        "supervision": "one-class",
        "implementation_status": "native-flow-with-simplified-transform-ensemble",
        "paper_fidelity": "paper-adaptation",
    },
)
class DifferNetDetector(BaseVisionDeepDetector):
    def __init__(
        self,
        *,
        pretrained: bool = False,
        image_size: int = 256,
        n_scales: int = 3,
        n_flow_steps: int = 8,
        n_transforms: int = 4,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        if "train_difference" in kwargs:
            raise TypeError(
                "train_difference belonged to the removed kNN implementation and is not "
                "a DifferNet paper parameter."
            )
        if n_transforms <= 0 or n_transforms > 4:
            raise ValueError("n_transforms must be between 1 and 4.")
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)
        self.n_scales = int(n_scales)
        self.n_flow_steps = int(n_flow_steps)
        self.n_transforms = int(n_transforms)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        requested_random_state = None if random_state is None else int(random_state)
        super().__init__(
            lr=self.learning_rate,
            epoch_num=self.epochs,
            batch_size=int(batch_size),
            device=device,
            random_state=None,
            verbose=0,
            **kwargs,
        )
        self.random_state = requested_random_state

    def build_model(self) -> DifferNetNetwork:
        with torch.random.fork_rng(devices=[]):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            model = DifferNetNetwork(
                pretrained=self.pretrained,
                image_size=self.image_size,
                n_scales=self.n_scales,
                n_flow_steps=self.n_flow_steps,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            model.flow.parameters(),
            lr=self.learning_rate,
            betas=(0.8, 0.8),
            eps=1e-4,
            weight_decay=1e-5,
        )
        return model

    def _transformed_batch(self, images: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.rot90(images, turns, dims=(-2, -1)) for turns in range(self.n_transforms)],
            dim=0,
        )

    @staticmethod
    def _nll(z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        return (0.5 * z.square().sum(dim=1) - logdet) / z.shape[1]

    def training_forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        images, _ = batch
        images = self._transformed_batch(images.to(self.device))
        self.model.train()
        self.model.feature_extractor.eval()
        self.optimizer.zero_grad(set_to_none=True)
        z, logdet = self.model(images)
        loss = self._nll(z, logdet).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluating_forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> NDArray:
        images, _ = batch
        batch_size = images.shape[0]
        images = self._transformed_batch(images.to(self.device))
        self.model.eval()
        z, _logdet = self.model(images)
        # The authors train with flow likelihood, but rank anomalies by the
        # mean latent energy across image transformations.
        scores = z.square().mean(dim=1).view(self.n_transforms, batch_size).mean(dim=0)
        return scores.cpu().numpy()

    def fit(
        self,
        x: object = MISSING,
        y: Optional[Iterable[int]] = None,
        **kwargs: object,
    ) -> "DifferNetDetector":
        value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        fitted = super().fit(value, y)
        self.is_fitted_ = True
        return fitted

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        items = list(resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
        if not items:
            return np.zeros((0,), dtype=np.float64)
        return np.asarray(super().decision_function(items, batch_size=batch_size), dtype=np.float64)

    def save_checkpoint(self, path: str | Path) -> Path:
        if getattr(self, "model", None) is None or not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 2,
                "detector": "vision_differnet",
                "config": {
                    "pretrained": self.pretrained,
                    "image_size": self.image_size,
                    "n_scales": self.n_scales,
                    "n_flow_steps": self.n_flow_steps,
                    "n_transforms": self.n_transforms,
                    "learning_rate": self.learning_rate,
                },
                "model_state_dict": export_module_state_dict(self.model),
                "decision_scores_": torch.as_tensor(self.decision_scores_, dtype=torch.float64),
                "threshold_": float(self.threshold_),
                "labels_": torch.as_tensor(self.labels_, dtype=torch.int64),
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            raise ValueError(
                "Unsupported DifferNet checkpoint. Retrain checkpoints created by the former kNN implementation."
            )
        if str(payload.get("detector", "")) not in {"vision_differnet", "differnet"}:
            raise ValueError("Invalid DifferNet checkpoint: detector marker mismatch.")

        config = payload.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("Invalid DifferNet checkpoint: missing config.")
        self.pretrained = bool(config.get("pretrained", self.pretrained))
        self.image_size = int(config.get("image_size", self.image_size))
        self.n_scales = int(config.get("n_scales", self.n_scales))
        self.n_flow_steps = int(config.get("n_flow_steps", self.n_flow_steps))
        self.n_transforms = int(config.get("n_transforms", self.n_transforms))
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))

        self.model = self.build_model()
        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Invalid DifferNet checkpoint: missing model state.")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.decision_scores_ = np.asarray(payload["decision_scores_"], dtype=np.float64)
        self.threshold_ = float(payload["threshold_"])
        self.labels_ = np.asarray(payload["labels_"], dtype=np.int64)
        self.is_fitted_ = True


__all__ = ["DifferNetDetector", "DifferNetNetwork"]
