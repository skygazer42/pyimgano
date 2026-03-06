"""
PromptAD - Learning Prompts with Only Normal Samples for Few-Shot Anomaly Detection

Reference:
    "PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection"
    CVPR 2024

Learns visual prompts from only normal samples for few-shot anomaly detection,
enabling effective detection with minimal training data.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class VisualPromptLearner(nn.Module):
    """Learns visual prompts for anomaly detection."""

    def __init__(self, num_prompts: int = 10, prompt_dim: int = 512, context_length: int = 16):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim

        # Learnable prompt embeddings
        self.prompts = nn.Parameter(torch.randn(num_prompts, context_length, prompt_dim))

        # Prompt encoder
        self.encoder = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_dim * 2, prompt_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply prompts to features.

        Parameters
        ----------
        features : torch.Tensor
            Input features (B, D)

        Returns
        -------
        prompted_features : torch.Tensor
            Features with learned prompts (B, D)
        """
        # Select relevant prompts based on input
        # Compute similarity to each prompt
        features_norm = F.normalize(features, p=2, dim=1)
        prompts_mean = self.prompts.mean(dim=1)  # (num_prompts, D)
        prompts_norm = F.normalize(prompts_mean, p=2, dim=1)

        similarity = torch.matmul(features_norm, prompts_norm.t())  # (B, num_prompts)
        weights = F.softmax(similarity, dim=1)  # (B, num_prompts)

        # Weighted combination of prompts
        selected_prompts = torch.einsum("bn,ncd->bcd", weights, self.prompts)  # (B, C, D)
        prompt_features = selected_prompts.mean(dim=1)  # (B, D)

        # Encode prompts
        prompt_encoded = self.encoder(prompt_features)

        # Combine with original features
        prompted = features + prompt_encoded

        return prompted


class FeatureAdapter(nn.Module):
    """Adapts features for few-shot learning."""

    def __init__(self, feature_dim: int):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
        )

        # Initialize as identity
        nn.init.zeros_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptation."""
        return x + self.adapter(x)


@register_model(
    "vision_promptad",
    tags=("vision", "deep", "promptad", "few-shot", "prompt", "cvpr2024", "sota"),
    metadata={
        "description": "PromptAD - Prompt learning with only normal samples (CVPR 2024)",
        "paper": "PromptAD: Learning Prompts with only Normal Samples",
        "year": 2024,
        "conference": "CVPR",
        "type": "prompt-learning",
    },
)
class VisionPromptAD(BaseVisionDeepDetector):
    """
    PromptAD: Learning Prompts with only Normal Samples for Few-Shot AD.

    Learns visual prompts from normal samples only, enabling effective
    few-shot anomaly detection without requiring anomalous samples.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    num_prompts : int, default=10
        Number of learnable prompts
    prompt_dim : int, default=512
        Dimension of prompts
    context_length : int, default=16
        Length of prompt context
    learning_rate : float, default=1e-3
        Learning rate
    batch_size : int, default=16
        Batch size for training
    epochs : int, default=30
        Number of training epochs
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    feature_extractor_ : nn.Module
        Pre-trained feature extractor
    prompt_learner_ : VisualPromptLearner
        Visual prompt learning module
    adapter_ : FeatureAdapter
        Feature adaptation module
    normal_prototypes_ : torch.Tensor
        Prototypes of normal samples

    Examples
    --------
    >>> from pyimgano.models import VisionPromptAD
    >>> import numpy as np
    >>>
    >>> # Create sample data (few-shot scenario)
    >>> X_train = np.random.rand(20, 224, 224, 3).astype(np.float32)  # Only 20 samples!
    >>> X_test = np.random.rand(50, 224, 224, 3).astype(np.float32)
    >>>
    >>> # Create and train detector with few-shot
    >>> detector = VisionPromptAD(num_prompts=10, epochs=20)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        num_prompts: int = 10,
        prompt_dim: int = 512,
        context_length: int = 16,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        epochs: int = 30,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.feature_extractor_ = None
        self.prompt_learner_ = None
        self.adapter_ = None
        self.normal_prototypes_ = None

    def _preprocess(self, X: NDArray) -> torch.Tensor:
        """Preprocess images."""
        if X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))

        X = X.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        X = (X - mean) / std

        return torch.from_numpy(X).float()

    def _build_feature_extractor(self):
        """Build pre-trained feature extractor."""
        if self.backbone == "wide_resnet50":
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            resnet = wide_resnet50_2(weights=weights)
        elif self.backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
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

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "VisionPromptAD":
        """
        Fit the PromptAD detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples only, can be few-shot)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionPromptAD
            Fitted detector
        """
        # Preprocess
        X_tensor = self._preprocess(X)

        # Initialize feature extractor
        if self.feature_extractor_ is None:
            self.feature_extractor_ = self._build_feature_extractor().to(self.device)

        # Get feature dimension
        with torch.no_grad():
            sample_features = self.feature_extractor_(X_tensor[:1].to(self.device))
            feature_dim = sample_features.shape[1]

        # Initialize modules
        if self.prompt_learner_ is None:
            self.prompt_learner_ = VisualPromptLearner(
                num_prompts=self.num_prompts,
                prompt_dim=min(feature_dim, self.prompt_dim),
                context_length=self.context_length,
            ).to(self.device)

        if self.adapter_ is None:
            self.adapter_ = FeatureAdapter(feature_dim=feature_dim).to(self.device)

        # Training
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(
            list(self.prompt_learner_.parameters()) + list(self.adapter_.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        self.prompt_learner_.train()
        self.adapter_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Extract features
                with torch.no_grad():
                    features = self.feature_extractor_(batch_images)

                # Apply prompts and adaptation
                prompted_features = self.prompt_learner_(features)
                adapted_features = self.adapter_(prompted_features)

                # Self-supervised loss: consistency between original and adapted
                # Normalize features
                features_norm = F.normalize(features, p=2, dim=1)
                adapted_norm = F.normalize(adapted_features, p=2, dim=1)

                # Cosine similarity loss (encourage consistency for normal samples)
                similarity = (features_norm * adapted_norm).sum(dim=1)
                loss = (1 - similarity).mean()

                # Compactness loss: encourage tight clustering
                center = adapted_features.mean(dim=0, keepdim=True)
                compactness = ((adapted_features - center) ** 2).sum(dim=1).mean()
                loss = loss + 0.1 * compactness

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # Build normal prototypes
        self._build_prototypes(X_tensor)

        return self

    def _build_prototypes(self, X_tensor: torch.Tensor):
        """Build prototypes from normal samples."""
        self.prompt_learner_.eval()
        self.adapter_.eval()

        all_features = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size].to(self.device)

                # Extract and process features
                features = self.feature_extractor_(batch)
                prompted = self.prompt_learner_(features)
                adapted = self.adapter_(prompted)

                all_features.append(adapted.cpu())

        all_features = torch.cat(all_features, dim=0)
        self.normal_prototypes_ = all_features

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : NDArray of shape (n_samples,)
            Anomaly scores (distance to normal prototypes)
        """
        self.prompt_learner_.eval()
        self.adapter_.eval()

        X_tensor = self._preprocess(X)
        scores = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size].to(self.device)

                # Extract and process features
                features = self.feature_extractor_(batch)
                prompted = self.prompt_learner_(features)
                adapted = self.adapter_(prompted)

                # Compute distance to normal prototypes
                adapted_cpu = adapted.cpu()
                dists = torch.cdist(adapted_cpu, self.normal_prototypes_)

                # Anomaly score = minimum distance to normal prototypes
                min_dists = dists.min(dim=1)[0]
                scores.append(min_dists.numpy())

        return np.concatenate(scores)

    def decision_function(self, X: NDArray) -> NDArray:
        """Alias for predict."""
        return self.predict(X)
