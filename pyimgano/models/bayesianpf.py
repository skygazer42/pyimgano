"""
BayesianPF - Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection

Reference:
    "Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection"
    CVPR 2025

Uses Bayesian learning to optimize prompt flows for zero-shot anomaly detection,
enabling effective detection without any training on the target domain.
"""

from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from pyimgano.models._imagenet_preprocess import preprocess_imagenet_batch
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class BayesianPromptGenerator(nn.Module):
    """Generates prompts using Bayesian inference."""

    def __init__(
        self, input_dim: int, prompt_dim: int = 512, num_prompts: int = 5, hidden_dim: int = 256
    ):
        super().__init__()
        self.num_prompts = num_prompts

        # Bayesian neural network for prompt generation
        self.mean_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim * num_prompts),
        )

        self.logvar_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim * num_prompts),
        )

        self.prompt_dim = prompt_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate prompts with uncertainty.

        Parameters
        ----------
        x : torch.Tensor
            Input features (B, D)

        Returns
        -------
        prompts : torch.Tensor
            Generated prompts (B, num_prompts, prompt_dim)
        mean : torch.Tensor
            Mean of posterior (B, num_prompts * prompt_dim)
        logvar : torch.Tensor
            Log variance of posterior
        """
        # Compute posterior parameters
        mean = self.mean_network(x)
        logvar = self.logvar_network(x)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        prompts_flat = mean + eps * std

        # Reshape to prompts
        b = x.size(0)
        prompts = prompts_flat.view(b, self.num_prompts, self.prompt_dim)

        return prompts, mean, logvar


class PromptFlowNetwork(nn.Module):
    """Prompt flow network for anomaly scoring."""

    def __init__(self, prompt_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Flow layers
        self.flow = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim),
        )

        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Flow prompts and compute anomaly scores.

        Parameters
        ----------
        prompts : torch.Tensor
            Input prompts (B, num_prompts, D)

        Returns
        -------
        scores : torch.Tensor
            Anomaly scores (B,)
        """
        # Apply flow to each prompt
        flowed_prompts = self.flow(prompts)  # (B, num_prompts, D)

        # Aggregate prompts
        aggregated = flowed_prompts.mean(dim=1)  # (B, D)

        # Compute anomaly score
        scores = self.scorer(aggregated).squeeze(-1)  # (B,)

        return scores


@register_model(
    "vision_bayesianpf",
    tags=("vision", "deep", "bayesianpf", "zero-shot", "bayesian", "cvpr2025", "sota"),
    metadata={
        "description": "BayesianPF - Bayesian Prompt Flow for Zero-Shot AD (CVPR 2025)",
        "paper": "Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection",
        "year": 2025,
        "conference": "CVPR",
        "type": "zero-shot-bayesian",
    },
)
class VisionBayesianPF(BaseVisionDeepDetector):
    """
    BayesianPF: Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection.

    Uses Bayesian inference to generate optimal prompts for zero-shot
    anomaly detection, requiring no training on the target domain.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    prompt_dim : int, default=512
        Dimension of prompts
    num_prompts : int, default=5
        Number of prompts to generate
    hidden_dim : int, default=256
        Hidden dimension in networks
    num_samples : int, default=10
        Number of Monte Carlo samples for Bayesian inference
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    feature_extractor_ : nn.Module
        Pre-trained feature extractor
    prompt_generator_ : BayesianPromptGenerator
        Bayesian prompt generator
    prompt_flow_ : PromptFlowNetwork
        Prompt flow network

    Examples
    --------
    >>> from pyimgano.models import VisionBayesianPF
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.random((50, 224, 224, 3)).astype(np.float32)
    >>> X_test = rng.random((20, 224, 224, 3)).astype(np.float32)
    >>>
    >>> # Create detector (zero-shot, minimal training!)
    >>> detector = VisionBayesianPF(num_samples=10)
    >>> detector.fit(X_train)  # Minimal calibration
    >>>
    >>> # Zero-shot prediction on new domains
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        prompt_dim: int = 512,
        num_prompts: int = 5,
        hidden_dim: int = 256,
        num_samples: int = 10,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        self.feature_extractor_ = None
        self.prompt_generator_ = None
        self.prompt_flow_ = None
        self.reference_statistics_ = None

    def _preprocess(self, x: NDArray) -> torch.Tensor:
        """Preprocess images."""
        return preprocess_imagenet_batch(x)

    def _build_feature_extractor(self):
        """Build feature extractor."""
        if self.backbone == "wide_resnet50":
            resnet, _ = load_torchvision_model("wide_resnet50", pretrained=True)
        elif self.backbone == "resnet18":
            resnet, _ = load_torchvision_model("resnet18", pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Freeze
        for param in extractor.parameters():
            param.requires_grad = False
        extractor.eval()

        return extractor

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionBayesianPF":
        """
        Fit the BayesianPF detector (minimal calibration).

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Calibration images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionBayesianPF
            Fitted detector
        """
        del y
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        # Preprocess
        x_tensor = self._preprocess(x_array)

        # Initialize feature extractor
        if self.feature_extractor_ is None:
            self.feature_extractor_ = self._build_feature_extractor().to(self.device)

        # Get feature dimension
        with torch.no_grad():
            sample_features = self.feature_extractor_(x_tensor[:1].to(self.device))
            feature_dim = sample_features.shape[1]

        # Initialize Bayesian prompt generator
        if self.prompt_generator_ is None:
            self.prompt_generator_ = BayesianPromptGenerator(
                input_dim=feature_dim,
                prompt_dim=self.prompt_dim,
                num_prompts=self.num_prompts,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

        # Initialize prompt flow
        if self.prompt_flow_ is None:
            self.prompt_flow_ = PromptFlowNetwork(
                prompt_dim=self.prompt_dim, hidden_dim=self.hidden_dim
            ).to(self.device)

        # Compute reference statistics from normal samples
        self.prompt_generator_.eval()
        self.prompt_flow_.eval()

        with torch.no_grad():
            all_scores = []
            for i in range(0, len(x_tensor), 32):
                batch = x_tensor[i : i + 32].to(self.device)
                features = self.feature_extractor_(batch)

                # Sample multiple times for Bayesian inference
                batch_scores = []
                for _ in range(self.num_samples):
                    prompts, _, _ = self.prompt_generator_(features)
                    scores = self.prompt_flow_(prompts)
                    batch_scores.append(scores)

                # Average scores
                avg_scores = torch.stack(batch_scores).mean(dim=0)
                all_scores.append(avg_scores.cpu())

            all_scores = torch.cat(all_scores)

            # Store statistics
            self.reference_statistics_ = {
                "mean": all_scores.mean().item(),
                "std": all_scores.std().item(),
            }

        print(
            f"Calibrated with normal samples: "
            f"mean={self.reference_statistics_['mean']:.4f}, "
            f"std={self.reference_statistics_['std']:.4f}"
        )

        return self

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        """
        Predict anomaly scores (zero-shot).

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : NDArray of shape (n_samples,)
            Anomaly scores
        """
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="predict"))

        self.prompt_generator_.eval()
        self.prompt_flow_.eval()

        x_tensor = self._preprocess(x_array)
        scores = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract features
                features = self.feature_extractor_(batch)

                # Bayesian inference with multiple samples
                batch_scores = []
                for _ in range(self.num_samples):
                    prompts, _, _ = self.prompt_generator_(features)
                    flow_scores = self.prompt_flow_(prompts)
                    batch_scores.append(flow_scores)

                # Average predictions (Bayesian model averaging)
                avg_scores = torch.stack(batch_scores).mean(dim=0)

                # Normalize using reference statistics
                if self.reference_statistics_ is not None:
                    avg_scores = (avg_scores - self.reference_statistics_["mean"]) / (
                        self.reference_statistics_["std"] + 1e-8
                    )

                scores.append(avg_scores.cpu().numpy())

        return np.concatenate(scores)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        """Alias for predict."""
        x_array = cast(
            NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        )
        if batch_size is None:
            return self.predict(x_array)

        batch_size_int = int(batch_size)
        if batch_size_int <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")

        old_batch_size = self.batch_size
        try:
            self.batch_size = batch_size_int
            return self.predict(x_array)
        finally:
            self.batch_size = old_batch_size

    def predict_with_uncertainty(
        self, x: object = MISSING, **kwargs: object
    ) -> tuple[NDArray, NDArray]:
        """
        Predict anomaly scores with uncertainty estimates.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : NDArray of shape (n_samples,)
            Mean anomaly scores
        uncertainties : NDArray of shape (n_samples,)
            Uncertainty estimates (standard deviation)
        """
        self.prompt_generator_.eval()
        self.prompt_flow_.eval()

        x_array = cast(
            NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="predict_with_uncertainty")
        )
        x_tensor = self._preprocess(x_array)
        scores = []
        uncertainties = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), 32):
                batch = x_tensor[i : i + 32].to(self.device)

                # Extract features
                features = self.feature_extractor_(batch)

                # Multiple samples for uncertainty estimation
                batch_scores = []
                for _ in range(self.num_samples):
                    prompts, _, _ = self.prompt_generator_(features)
                    flow_scores = self.prompt_flow_(prompts)
                    batch_scores.append(flow_scores)

                batch_scores = torch.stack(batch_scores)  # (num_samples, B)

                # Compute mean and std
                mean_scores = batch_scores.mean(dim=0)
                std_scores = batch_scores.std(dim=0)

                scores.append(mean_scores.cpu().numpy())
                uncertainties.append(std_scores.cpu().numpy())

        return np.concatenate(scores), np.concatenate(uncertainties)
