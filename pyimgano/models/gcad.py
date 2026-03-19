"""
GCAD - Graph Convolutional Anomaly Detection

Reference:
    "Graph Convolutional Anomaly Detection for Visual Anomaly Detection"
    Combines graph neural networks with anomaly detection for capturing spatial relationships

This method uses graph convolutional networks to model spatial relationships
between image patches, enabling better anomaly localization.
"""

from typing import List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class GraphConvLayer(nn.Module):
    """Graph convolutional layer for feature aggregation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (batch_size, num_nodes, in_features)
        adj : torch.Tensor
            Adjacency matrix of shape (batch_size, num_nodes, num_nodes)

        Returns
        -------
        out : torch.Tensor
            Output features of shape (batch_size, num_nodes, out_features)
        """
        # x: (B, N, F_in), weight: (F_in, F_out)
        support = torch.matmul(x, self.weight)  # (B, N, F_out)

        # Aggregate neighbors: adj @ support
        output = torch.matmul(adj, support)  # (B, N, F_out)

        if self.bias is not None:
            output = output + self.bias

        return output


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.5):
        super().__init__()

        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            self.layers.append(GraphConvLayer(dims[i], dims[i + 1]))

        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN layers.

        Parameters
        ----------
        x : torch.Tensor
            Node features (B, N, F)
        adj : torch.Tensor
            Adjacency matrix (B, N, N)

        Returns
        -------
        h : torch.Tensor
            Encoded features (B, N, F_out)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)

            # Apply activation except for last layer
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


class FeatureExtractor(nn.Module):
    """Extract features from images using CNN."""

    def __init__(self, backbone: str = "wide_resnet50"):
        super().__init__()

        if backbone == "wide_resnet50":
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            resnet = wide_resnet50_2(weights=weights)
        elif backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract feature layers
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )

        # Freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        with torch.no_grad():
            return self.features(x)


@register_model(
    "vision_gcad",
    tags=("vision", "deep", "gcad", "graph", "sota"),
    metadata={
        "description": "Graph Convolutional Anomaly Detection - Uses GCN to model spatial relationships",
        "paper": "Graph Convolutional Anomaly Detection",
        "year": 2023,
        "type": "graph-based",
    },
)
class VisionGCAD(BaseVisionDeepDetector):
    """
    GCAD: Graph Convolutional Anomaly Detection.

    Uses graph convolutional networks to model spatial relationships between
    image patches for improved anomaly detection and localization.

    Parameters
    ----------
    backbone : str, default='wide_resnet50'
        CNN backbone for feature extraction
    patch_size : int, default=8
        Size of image patches for graph nodes
    hidden_dims : List[int], default=[256, 128, 64]
        Hidden dimensions for GCN layers
    k_neighbors : int, default=8
        Number of nearest neighbors for graph construction
    learning_rate : float, default=1e-4
        Learning rate
    batch_size : int, default=16
        Batch size for training
    epochs : int, default=30
        Number of training epochs
    device : str, default='cuda'
        Device for computation

    Attributes
    ----------
    feature_extractor_ : FeatureExtractor
        CNN for feature extraction
    gcn_ : GCNEncoder
        Graph convolutional encoder
    memory_bank_ : torch.Tensor
        Memory bank of normal features

    Examples
    --------
    >>> from pyimgano.models import VisionGCAD
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.random((100, 224, 224, 3)).astype(np.float32)
    >>> X_test = rng.random((20, 224, 224, 3)).astype(np.float32)
    >>>
    >>> # Create and train detector
    >>> detector = VisionGCAD(epochs=10)
    >>> detector.fit(X_train)
    >>>
    >>> # Predict anomaly scores
    >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        patch_size: int = 8,
        hidden_dims: Optional[List[int]] = None,
        k_neighbors: int = 8,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 30,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.patch_size = patch_size
        self.hidden_dims = list(hidden_dims or [256, 128, 64])
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        self.feature_extractor_ = None
        self.gcn_ = None
        self.memory_bank_ = None
        self.feature_dim_ = None

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

    def _extract_patches(self, features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Extract patch features and create graph nodes.

        Parameters
        ----------
        features : torch.Tensor
            Feature maps (B, C, H, W)

        Returns
        -------
        patches : torch.Tensor
            Patch features (B, num_patches, feature_dim)
        h_patches : int
            Number of patches in height
        w_patches : int
            Number of patches in width
        """
        batch_size, channels, height, width = features.shape
        p = self.patch_size

        # Ensure dimensions are divisible by patch_size
        height_trimmed = (height // p) * p
        width_trimmed = (width // p) * p
        features = features[:, :, :height_trimmed, :width_trimmed]

        # Extract patches
        patches = features.unfold(2, p, p).unfold(3, p, p)
        # (B, C, h_patches, w_patches, p, p)

        h_patches = height_trimmed // p
        w_patches = width_trimmed // p

        # Reshape to (B, num_patches, feature_dim)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, h_patches * w_patches, channels * p * p)

        return patches, h_patches, w_patches

    def _build_adjacency_matrix(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build k-NN graph adjacency matrix.

        Parameters
        ----------
        features : torch.Tensor
            Node features (B, N, F)
        k : int
            Number of nearest neighbors

        Returns
        -------
        adj : torch.Tensor
            Normalized adjacency matrix (B, N, N)
        """
        batch_size, num_nodes, feature_dim = features.shape

        # Compute pairwise distances
        features_norm = features.view(batch_size, num_nodes, 1, feature_dim)
        dist = torch.sum((features_norm - features.view(batch_size, 1, num_nodes, feature_dim)) ** 2, dim=-1)
        # (B, N, N)

        # Find k nearest neighbors
        _, indices = torch.topk(-dist, k=min(k + 1, num_nodes), dim=-1)
        # (B, N, k+1) - includes self

        # Build adjacency matrix
        adj = torch.zeros(batch_size, num_nodes, num_nodes, device=features.device)
        for b in range(batch_size):
            for i in range(num_nodes):
                adj[b, i, indices[b, i]] = 1.0

        # Normalize adjacency matrix (add self-loops and apply degree normalization)
        adj = adj + torch.eye(num_nodes, device=features.device).unsqueeze(0)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / degree

        return adj

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionGCAD":
        """
        Fit the GCAD detector.

        Parameters
        ----------
        X : NDArray of shape (n_samples, height, width, channels)
            Training images (normal samples)
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : VisionGCAD
            Fitted detector
        """
        del y
        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        # Preprocess
        x_tensor = self._preprocess(x_array)

        # Initialize feature extractor
        if self.feature_extractor_ is None:
            self.feature_extractor_ = FeatureExtractor(backbone=self.backbone).to(self.device)

        # Extract features to get dimensions
        with torch.no_grad():
            sample_features = self.feature_extractor_(x_tensor[:1].to(self.device))
            sample_patches, _, _ = self._extract_patches(sample_features)
            self.feature_dim_ = sample_patches.shape[-1]

        # Initialize GCN
        if self.gcn_ is None:
            self.gcn_ = GCNEncoder(
                input_dim=self.feature_dim_, hidden_dims=self.hidden_dims, dropout=0.3
            ).to(self.device)

        # Training
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.gcn_.parameters(), lr=self.learning_rate, weight_decay=0.0)

        self.gcn_.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch_images,) in dataloader:
                batch_images = batch_images.to(self.device)

                # Extract features
                with torch.no_grad():
                    features = self.feature_extractor_(batch_images)
                    patches, _, _ = self._extract_patches(features)

                # Build graph
                adj = self._build_adjacency_matrix(patches, self.k_neighbors)

                # Forward through GCN
                encoded = self.gcn_(patches, adj)

                # Self-supervised loss: reconstruction
                loss = F.mse_loss(encoded, patches)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # Build memory bank
        self._build_memory_bank(x_tensor)

        return self

    def _build_memory_bank(self, x: torch.Tensor):
        """Build memory bank from training data."""
        self.gcn_.eval()
        all_features = []

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch = x[i : i + self.batch_size].to(self.device)

                # Extract and encode
                features = self.feature_extractor_(batch)
                patches, _, _ = self._extract_patches(features)
                adj = self._build_adjacency_matrix(patches, self.k_neighbors)
                encoded = self.gcn_(patches, adj)

                all_features.append(encoded.cpu())

        self.memory_bank_ = torch.cat(all_features, dim=0)

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
            Anomaly scores
        """
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )

        self.gcn_.eval()

        x_array = cast(NDArray, resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        x_tensor = self._preprocess(x_array)
        scores = []

        with torch.no_grad():
            for i in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[i : i + self.batch_size].to(self.device)

                # Extract and encode
                features = self.feature_extractor_(batch)
                patches, _, _ = self._extract_patches(features)
                adj = self._build_adjacency_matrix(patches, self.k_neighbors)
                encoded = self.gcn_(patches, adj)

                # Compute distance to memory bank
                # For each test sample, find min distance to training samples
                for j in range(len(encoded)):
                    test_feat = encoded[j]  # (N, F)
                    memory = self.memory_bank_.to(self.device)  # (M, N, F)

                    # Compute distances
                    dists = torch.cdist(
                        test_feat.unsqueeze(0), memory.view(-1, memory.shape[-1])
                    )  # (1, M*N)

                    # Anomaly score = minimum distance
                    score = dists.min().item()
                    scores.append(score)

        return np.array(scores)

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
