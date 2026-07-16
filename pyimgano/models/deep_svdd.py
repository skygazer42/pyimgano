# -*- coding: utf-8 -*-
"""DeepSVDD 异常检测实现 (PyTorch 版本)."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from torch.utils.data import DataLoader, TensorDataset

from ..utils.param_check import check_parameter
from ..utils.random_state import isolated_random_state_method
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


def _get_activation(name: str) -> nn.Module:
    if str(name).strip().lower() == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)

    from ..utils.torch_activations import get_activation_by_name

    return get_activation_by_name(name)


class InnerDeepSVDD(nn.Module):
    """DeepSVDD 神经网络主体。"""

    def __init__(
        self,
        n_features: int,
        use_autoencoder: bool,
        hidden_neurons,
        hidden_activation: str,
        output_activation: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.use_autoencoder = use_autoencoder
        self.hidden_neurons = list(hidden_neurons or [64, 32])
        if len(self.hidden_neurons) < 2:
            raise ValueError("hidden_neurons 至少包含两个元素，以便构建输出层")
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder() if self.use_autoencoder else None
        self.center = None

    # ------------------------------------------------------------------
    def _build_encoder(self) -> nn.Sequential:
        layers = nn.Sequential()
        layers.add_module("linear0", nn.Linear(self.n_features, self.hidden_neurons[0], bias=False))
        layers.add_module("act0", _get_activation(self.hidden_activation))

        for idx in range(1, len(self.hidden_neurons) - 1):
            layers.add_module(
                f"linear{idx}",
                nn.Linear(self.hidden_neurons[idx - 1], self.hidden_neurons[idx], bias=False),
            )
            layers.add_module(f"act{idx}", _get_activation(self.hidden_activation))
            if self.dropout_rate > 0:
                layers.add_module(f"drop{idx}", nn.Dropout(self.dropout_rate))

        layers.add_module(
            "net_output",
            nn.Linear(self.hidden_neurons[-2], self.hidden_neurons[-1], bias=False),
        )
        return layers

    @torch.no_grad()
    def init_center(self, features: torch.Tensor, eps: float = 0.1) -> None:
        self.eval()
        center = self.encode(features).mean(dim=0)
        center = torch.where(
            (center.abs() < eps) & (center < 0), torch.full_like(center, -eps), center
        )
        center = torch.where(
            (center.abs() < eps) & (center > 0), torch.full_like(center, eps), center
        )
        self.center = center.detach()

    def _build_decoder(self) -> nn.Sequential:
        layers = nn.Sequential()

        # Decode from representation space back to the input feature space.
        for idx in range(len(self.hidden_neurons) - 1, 0, -1):
            layers.add_module(
                f"linear_d{idx}",
                nn.Linear(self.hidden_neurons[idx], self.hidden_neurons[idx - 1], bias=False),
            )
            layers.add_module(f"act_d{idx}", _get_activation(self.hidden_activation))
            if self.dropout_rate > 0:
                layers.add_module(f"drop_d{idx}", nn.Dropout(self.dropout_rate))

        layers.add_module("recon", nn.Linear(self.hidden_neurons[0], self.n_features, bias=False))
        layers.add_module("recon_act", _get_activation(self.output_activation))
        return layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the representation vector used for SVDD distance scoring."""

        return self.encoder(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("reconstruct() requires use_autoencoder=True")
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.use_autoencoder:
            return self.reconstruct(z)
        return z


@register_model(
    "core_deep_svdd",
    tags=("deep", "core", "features", "torch", "one-class"),
    metadata={
        "description": "核心 DeepSVDD 异常检测器",
        "paper": "Deep One-Class Classification",
        "paper_url": "https://proceedings.mlr.press/v80/ruff18a.html",
        "year": 2018,
        "supervision": "one-class",
        "implementation_status": "paper-objectives-generic-feature-network",
        "paper_fidelity": "paper-adaptation",
    },
)
class CoreDeepSVDD(BaseDetector):
    """核心 DeepSVDD 实现（native BaseDetector contract）。"""

    def __init__(
        self,
        n_features: int | None = None,
        *,
        center=None,
        objective: str = "one-class",
        nu: float = 0.1,
        warm_up_epochs: int = 10,
        radius_update_interval: int = 5,
        use_autoencoder: bool = False,
        hidden_neurons=None,
        hidden_activation: str = "leaky_relu",
        output_activation: str = "identity",
        optimizer: str = "adam",
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 128,
        dropout_rate: float = 0.0,
        l2_weight: float = 1e-6,
        preprocessing: bool = True,
        device: str | None = None,
        verbose: int = 1,
        random_state: int | None = None,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_features = int(n_features) if n_features is not None else None
        self.center = center
        self.center_ = None
        self.objective = str(objective).strip().lower()
        self.nu = float(nu)
        self.warm_up_epochs = int(warm_up_epochs)
        self.radius_update_interval = int(radius_update_interval)
        self.radius_ = 0.0
        self.use_autoencoder = bool(use_autoencoder)
        self.hidden_neurons = list(hidden_neurons or [64, 32])
        self.hidden_activation = str(hidden_activation)
        self.output_activation = str(output_activation)
        self.optimizer = str(optimizer).strip().lower()
        self.optimizer_name = self.optimizer
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.dropout_rate = float(dropout_rate)
        self.l2_weight = float(l2_weight)
        self.preprocessing = bool(preprocessing)
        self.device = "cpu" if device is None else str(device)
        self.verbose = int(verbose)
        self.random_state = None if random_state is None else int(random_state)
        self.scaler = None
        self.model = None

        if self.objective not in {"one-class", "soft-boundary"}:
            raise ValueError("objective must be 'one-class' or 'soft-boundary'")
        if not 0.0 < self.nu <= 1.0:
            raise ValueError("nu must satisfy 0 < nu <= 1")
        if self.warm_up_epochs < 0:
            raise ValueError("warm_up_epochs must be non-negative")
        if self.radius_update_interval <= 0:
            raise ValueError("radius_update_interval must be positive")
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.l2_weight < 0:
            raise ValueError("l2_weight must be non-negative")
        if self.optimizer not in {"adam", "amsgrad"}:
            raise ValueError("optimizer must be 'adam' or 'amsgrad'")

        check_parameter(
            dropout_rate,
            low=0,
            high=1,
            include_left=True,
            include_right=False,
            param_name="dropout_rate",
        )
        check_parameter(
            self.lr,
            low=0,
            include_left=False,
            param_name="lr",
        )

    # ------------------------------------------------------------------
    @isolated_random_state_method
    def fit(self, x, y=None):
        x = check_array(x)
        self._set_n_classes(y)

        if self.n_features is None:
            self.n_features = int(x.shape[1])
        elif int(x.shape[1]) != int(self.n_features):
            raise ValueError(f"Expected n_features={self.n_features}, got {x.shape[1]}")

        if self.preprocessing:
            self.scaler = StandardScaler()
            x_norm = self.scaler.fit_transform(x)
        else:
            x_norm = x.copy()

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(x_norm.shape[0])
        x_norm = x_norm[indices]

        self.model = InnerDeepSVDD(
            n_features=self.n_features,
            use_autoencoder=self.use_autoencoder,
            hidden_neurons=self.hidden_neurons,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            dropout_rate=self.dropout_rate,
        ).to(self.device)
        self.radius_ = 0.0

        tensor_data = torch.tensor(x_norm, dtype=torch.float32)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Deep SVDD's optional autoencoder is a pretraining stage. It is not
        # optimized jointly with the hypersphere objective.
        if self.use_autoencoder:
            pretrain_optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.l2_weight,
                amsgrad=self.optimizer == "amsgrad",
            )
            for _epoch in range(self.epochs):
                self.model.train()
                for batch_x, _ in dataloader:
                    batch_x = batch_x.to(self.device)
                    pretrain_optimizer.zero_grad()
                    reconstruction = self.model.reconstruct(self.model.encode(batch_x))
                    reconstruction_loss = torch.mean(torch.square(reconstruction - batch_x))
                    reconstruction_loss.backward()
                    pretrain_optimizer.step()

        if self.center is None:
            self.model.init_center(tensor_data.to(self.device))
            self.center_ = self.model.center
        else:
            center_arr = np.asarray(self.center, dtype=np.float32).reshape(-1)
            rep_dim = int(self.hidden_neurons[-1])
            if center_arr.shape[0] != rep_dim:
                raise ValueError(f"Expected center shape ({rep_dim},), got {center_arr.shape}")
            self.center_ = torch.tensor(center_arr, dtype=torch.float32)
            self.center_ = self.center_.to(self.device)
            self.model.center = self.center_

        optimizer = torch.optim.Adam(
            self.model.encoder.parameters(),
            lr=self.lr,
            weight_decay=self.l2_weight,
            amsgrad=self.optimizer == "amsgrad",
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.model.train()

            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                rep = self.model.encode(batch_x)
                dist = torch.sum((rep - self.center_) ** 2, dim=-1)
                if self.objective == "soft-boundary":
                    radius_squared = dist.new_tensor(self.radius_**2)
                    loss = radius_squared + torch.relu(dist - radius_squared).mean() / self.nu
                else:
                    loss = dist.mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item() * batch_x.size(0)

            epoch_loss /= x_norm.shape[0]

            should_update_radius = (
                self.objective == "soft-boundary"
                and epoch + 1 >= self.warm_up_epochs
                and (epoch + 1) % self.radius_update_interval == 0
            )
            if should_update_radius:
                self.model.eval()
                with torch.no_grad():
                    representations = self.model.encode(tensor_data.to(self.device))
                    distances = torch.sum((representations - self.center_) ** 2, dim=-1)
                self.radius_ = float(np.quantile(np.sqrt(distances.cpu().numpy()), 1.0 - self.nu))

            if self.verbose:
                logger.info("Epoch %d/%d - Loss: %.6f", epoch + 1, self.epochs, epoch_loss)

        self.decision_scores_ = self.decision_function(x)
        self._process_decision_scores()
        return self

    def decision_function(self, x):
        x = check_array(x)
        if self.model is None or self.center_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if int(x.shape[1]) != int(self.n_features):
            raise ValueError(f"Expected n_features={self.n_features}, got {x.shape[1]}")

        if self.preprocessing and self.scaler is not None:
            x_norm = self.scaler.transform(x)
        else:
            x_norm = x.copy()

        tensor_data = torch.tensor(x_norm, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            rep = self.model.encode(tensor_data)
            dist = torch.sum((rep - self.center_) ** 2, dim=-1)
            if self.objective == "soft-boundary":
                dist = dist - self.radius_**2

        return dist.cpu().numpy()


@register_model(
    "vision_deep_svdd",
    tags=("vision", "deep", "torch"),
    metadata={
        "description": "基于 DeepSVDD 的视觉异常检测器",
        "paper": "Deep One-Class Classification",
        "paper_url": "https://proceedings.mlr.press/v80/ruff18a.html",
        "year": 2018,
        "supervision": "one-class",
        "implementation_status": "feature-extractor-image-adaptation",
        "paper_fidelity": "paper-adaptation",
    },
)
class VisionDeepSVDD(BaseVisionDetector):
    """视觉版 DeepSVDD：对图像提取特征后，在特征空间训练 DeepSVDD。"""

    def __init__(
        self,
        *,
        feature_extractor=None,
        n_features: int | None = None,
        center=None,
        objective: str = "one-class",
        nu: float = 0.1,
        warm_up_epochs: int = 10,
        radius_update_interval: int = 5,
        use_autoencoder: bool = False,
        hidden_neurons=None,
        hidden_activation: str = "leaky_relu",
        output_activation: str = "identity",
        optimizer: str = "adam",
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 128,
        dropout_rate: float = 0.0,
        l2_weight: float = 1e-6,
        preprocessing: bool = True,
        device: str | None = None,
        verbose: int = 1,
        random_state: int | None = None,
        contamination: float = 0.1,
        **kwargs,
    ) -> None:
        if feature_extractor is None:
            # DeepSVDD is sensitive to input dimensionality. The BaseVisionDetector
            # default (224x224 flattened pixels) can be too large for a simple MLP.
            # Use a smaller default while still supporting paths input.
            from pyimgano.utils.image_ops import ImagePreprocessor

            feature_extractor = ImagePreprocessor(
                resize=(32, 32),
                output_tensor=False,
                error_mode="zeros",
            )

        self._detector_kwargs = dict(
            n_features=n_features,
            center=center,
            objective=objective,
            nu=nu,
            warm_up_epochs=warm_up_epochs,
            radius_update_interval=radius_update_interval,
            use_autoencoder=use_autoencoder,
            hidden_neurons=hidden_neurons,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            optimizer=optimizer,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            l2_weight=l2_weight,
            preprocessing=preprocessing,
            device=device,
            verbose=verbose,
            random_state=random_state,
            contamination=contamination,
            **dict(kwargs),
        )

        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreDeepSVDD(**self._detector_kwargs)

    def fit(self, x: Iterable[str], y=None):
        return super().fit(x, y=y)

    def decision_function(self, x):
        return super().decision_function(x)
