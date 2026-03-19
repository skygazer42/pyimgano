"""
PANDA - Prototypical Anomaly Network for Deep Anomaly Detection

Reference:
    "Prototypical Networks for Anomaly Detection in Industrial Images"

Uses prototypical learning to create representative prototypes of normal
patterns and detects anomalies based on distance to these prototypes.
"""

from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class PrototypicalEncoder(nn.Module):
    """Encoder network for prototypical learning."""

    def __init__(self, backbone: str = "wide_resnet50", projection_dim: int = 256):
        super().__init__()

        # Feature extractor
        if backbone == "wide_resnet50":
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            resnet = wide_resnet50_2(weights=weights)
            feature_dim = 1024
        elif backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract feature layers
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # Projection head for metric learning
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project features."""
        features = self.backbone(x)
        projected = self.projection(features)
        # L2 normalize for metric learning
        return F.normalize(projected, p=2, dim=1)


@register_model(
    "vision_panda",
    tags=("vision", "deep", "panda", "prototypical", "metric", "sota"),
    metadata={
        "description": "PANDA - Prototypical Anomaly Network with metric learning",
        "paper": "Prototypical Networks for Anomaly Detection",
        "year": 2023,
        "type": "metric-learning",
    },
)
class VisionPANDA(BaseVisionDeepDetector):
    """
    PANDA: Prototypical Anomaly Network for Deep Anomaly Detection.

    Uses prototypical learning to learn representative prototypes of normal
    patterns. Anomalies are detected based on their distance to prototypes.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        Feature extraction backbone
    projection_dim : int, default=256
        Dimension of projected embedding space
    n_prototypes : int, default=10
        Number of prototypes to learn
    learning_rate : float, default=1e-4
        Learning rate for training
    batch_size : int, default=32
        Batch size for training
    epochs : int, default=30
        Number of training epochs
    margin : float, default=0.5
        Margin for contrastive learning
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    encoder_ : PrototypicalEncoder
        Encoder network
    prototypes_ : torch.Tensor
        Learned prototypes of shape (n_prototypes, projection_dim)

    Examples
    --------
    >>> from pyimgano.models import VisionPANDA
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.random((100, 224, 224, 3)).astype(np.float32)
    >>> X_test = rng.random((20, 224, 224, 3)).astype(np.float32)
    >>>
    >>> # Create and train detector
    >>> detector = VisionPANDA(n_prototypes=10, epochs=20)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        projection_dim: int = 256,
        n_prototypes: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 30,
        margin: float = 0.5,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.projection_dim = projection_dim
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        self.encoder_ = None
        self.prototypes_ = None

    def _preprocess(self, x: NDArray) -> torch.Tensor:
        """Preprocess images."""
        # Convert to CHW format if needed
        if x.shape[-1] == 3:
            x = np.transpose(x, (0, 3, 1, 2))

        # Normalize
        x = x.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        x = (x - mean) / std

        return torch.from_numpy(x).float()

    def _prototype_loss(self, embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute prototypical loss.

        Parameters
        ----------
        embeddings : torch.Tensor
            Batch embeddings (B, D)
        prototypes : torch.Tensor
            Prototypes (K, D)

        Returns
        -------
        loss : torch.Tensor
            Prototypical loss
        """
        # Compute distances to all prototypes
        # embeddings: (B, D), prototypes: (K, D)
        # distances: (B, K)
        distances = torch.cdist(embeddings, prototypes, p=2)

        # For each sample, minimize distance to nearest prototype
        min_distances, _ = distances.min(dim=1)

        # Loss is average minimum distance
        loss = min_distances.mean()

        # Add compactness regularization
        # Encourage prototypes to be well-separated
        if len(prototypes) > 1:
            proto_distances = torch.cdist(prototypes, prototypes, p=2)
            # Mask out diagonal (self-distances)
            mask = ~torch.eye(len(prototypes), dtype=torch.bool, device=proto_distances.device)
            proto_min_dist = proto_distances[mask].view(len(prototypes), -1).min(dim=1)[0]
            # Regularization: encourage minimum distance between prototypes
            compactness_loss = -proto_min_dist.mean()
            loss = loss + 0.1 * compactness_loss

        return loss

    def _initialize_prototypes(self, x_tensor: torch.Tensor):
        """Initialize prototypes using K-means clustering."""
        # Extract embeddings for all training data
        self.encoder_.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)
                embeddings = self.encoder_(batch)
                all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.vstack(all_embeddings)

        # Use K-means to find initial prototypes
        kmeans = KMeans(n_clusters=self.n_prototypes, random_state=self.random_state, n_init=10)
        kmeans.fit(all_embeddings)

        # Initialize prototypes
        self.prototypes_ = nn.Parameter(
            torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        )

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionPANDA":
        """
        Fit the PANDA detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionPANDA
            Fitted detector
        """
        del y
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        # Preprocess
        x_tensor = self._preprocess(x_array)

        # Initialize encoder
        if self.encoder_ is None:
            self.encoder_ = PrototypicalEncoder(
                backbone=self.backbone, projection_dim=self.projection_dim
            ).to(self.device)

        # Initialize prototypes
        self._initialize_prototypes(x_tensor)

        # Training
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Optimize both encoder and prototypes
        optimizer = torch.optim.Adam(
            list(self.encoder_.parameters()) + [self.prototypes_], lr=self.learning_rate, weight_decay=0.0
        )

        self.encoder_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Extract embeddings
                embeddings = self.encoder_(batch_images)

                # Compute loss
                loss = self._prototype_loss(embeddings, self.prototypes_)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Normalize prototypes after update
                with torch.no_grad():
                    self.prototypes_.data = F.normalize(self.prototypes_.data, p=2, dim=1)

                total_loss += loss.detach().item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        return self

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : NDArray of shape (n_samples,)
            Anomaly scores (distance to nearest prototype)
        """
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )

        self.encoder_.eval()

        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        x_tensor = self._preprocess(x_array)
        scores = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract embeddings
                embeddings = self.encoder_(batch)

                # Compute distances to prototypes
                distances = torch.cdist(embeddings, self.prototypes_, p=2)

                # Anomaly score = distance to nearest prototype
                min_distances, _ = distances.min(dim=1)
                scores.append(min_distances.cpu().numpy())

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

    def get_prototype_assignments(
        self, x: object = MISSING, **kwargs: object
    ) -> NDArray:
        """
        Get prototype assignments for samples.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Input images

        Returns
        -------
        assignments : NDArray of shape (n_samples,)
            Index of nearest prototype for each sample
        """
        self.encoder_.eval()

        x_array = cast(
            NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="get_prototype_assignments")
        )
        x_tensor = self._preprocess(x_array)
        assignments = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract embeddings
                embeddings = self.encoder_(batch)

                # Find nearest prototype
                distances = torch.cdist(embeddings, self.prototypes_, p=2)
                _, indices = distances.min(dim=1)
                assignments.append(indices.cpu().numpy())

        return np.concatenate(assignments)
