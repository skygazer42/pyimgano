"""
MambaAD-style reconstruction on patch embeddings (industrial anomaly detection).

This implementation is intentionally **lightweight** and follows the core idea:
1) Extract fixed-grid patch embeddings using a frozen foundation backbone (default: DINOv2).
2) Train a small sequence model (Mamba SSM) to reconstruct the normal patch embedding patterns.
3) Use per-patch reconstruction error as an anomaly map; aggregate to an image score.

Notes
-----
- The goal is a practical industrial workflow (numpy-first + pixel maps), not a
  line-by-line reimplementation of every training detail from the original codebase.
- The Mamba dependency is optional. Install with:
    pip install "pyimgano[mamba]"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import require

from .anomalydino import PatchEmbedder, TorchHubDinoV2Embedder
from .baseCv import BaseVisionDeepDetector
from .patchknn_core import AggregationMethod, aggregate_patch_scores, reshape_patch_scores
from .registry import register_model


ImageInput = Union[str, np.ndarray]


def _require_torch():
    return require("torch", purpose="MambaAD training/inference")


def _require_mamba_ssm():
    return require("mamba_ssm", extra="mamba", purpose="MambaAD sequence model (Mamba SSM)")


@dataclass
class _Embedded:
    patches: NDArray
    grid_shape: Tuple[int, int]
    original_size: Tuple[int, int]


def _as_patch_batch(patches: NDArray) -> "Any":
    """Convert (P, D) numpy patch embeddings to torch (1, P, D)."""
    torch = _require_torch()
    arr = np.asarray(patches, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D patch embeddings (P,D), got {arr.shape}")
    return torch.from_numpy(arr).unsqueeze(0)


class _MambaReconstructor:
    """Small Mamba-based reconstruction network on patch token sequences."""

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        torch = _require_torch()
        mamba_ssm = _require_mamba_ssm()

        # Mamba class import path varies by version; support a couple of common ones.
        Mamba = getattr(mamba_ssm, "Mamba", None)
        if Mamba is None:  # pragma: no cover
            try:
                from mamba_ssm.modules.mamba_simple import Mamba as _Mamba  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "Unable to import Mamba from mamba_ssm. "
                    "Your installed mamba-ssm version may be unsupported."
                ) from exc
            Mamba = _Mamba

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = torch.nn.LayerNorm(d_model)
                self.mamba = Mamba(
                    d_model=d_model,
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                )

            def forward(self, x):
                # x: (B, L, D)
                return x + self.mamba(self.norm(x))

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block() for _ in range(int(n_layers))])
                self.norm = torch.nn.LayerNorm(d_model)
                self.head = torch.nn.Linear(d_model, d_model, bias=True)

            def forward(self, x):
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return self.head(x)

        self.net = Net()

    def to(self, device: str):
        self.net.to(device)
        return self

    def train(self, mode: bool = True):
        self.net.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        return self.net.parameters()

    def __call__(self, x):
        return self.net(x)


@register_model(
    "vision_mambaad",
    tags=("vision", "deep", "mambaad", "mamba", "ssm", "numpy", "pixel_map"),
    metadata={
        "description": "MambaAD-style patch embedding reconstruction with Mamba SSM (NeurIPS 2024)",
        "paper": "MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection",
        "year": 2024,
        "conference": "NeurIPS",
        "requires_optional_deps": ["mamba-ssm"],
    },
)
class VisionMambaAD(BaseVisionDeepDetector):
    """MambaAD-style patch reconstruction detector (unsupervised, pixel-map capable)."""

    def __init__(
        self,
        *,
        embedder: Optional[PatchEmbedder] = None,
        device: str = "cpu",
        image_size: int = 518,
        dino_model_name: str = "dinov2_vits14",
        epochs: int = 5,
        batch_size: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        noise_std: float = 0.02,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        aggregation_method: AggregationMethod = "topk_mean",
        aggregation_topk: float = 0.01,
        contamination: float = 0.1,
        random_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(contamination=float(contamination), **kwargs)

        if embedder is None:
            embedder = TorchHubDinoV2Embedder(
                model_name=str(dino_model_name),
                device=str(device),
                image_size=int(image_size),
            )

        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.embedder = embedder
        self.device = str(device)
        self.image_size = int(image_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.noise_std = float(noise_std)
        self.n_layers = int(n_layers)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.aggregation_method = aggregation_method
        self.aggregation_topk = float(aggregation_topk)
        self.random_seed = int(random_seed)

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

        self._reconstructor: Optional[_MambaReconstructor] = None
        self._d_model: Optional[int] = None

    def _embed(self, image: ImageInput) -> _Embedded:
        patches, grid_shape, original_size = self.embedder.embed(image)
        patches_np = np.asarray(patches, dtype=np.float32)
        if patches_np.ndim != 2:
            raise ValueError(f"Expected 2D patch embeddings, got {patches_np.shape}")

        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if patches_np.shape[0] != grid_h * grid_w:
            raise ValueError(
                "Patch embedding count does not match grid shape. "
                f"Got {patches_np.shape[0]} patches for grid {grid_h}x{grid_w}."
            )

        original_h, original_w = int(original_size[0]), int(original_size[1])
        if original_h <= 0 or original_w <= 0:
            raise ValueError(f"Invalid original_size: {original_size}")

        return _Embedded(
            patches=patches_np,
            grid_shape=(grid_h, grid_w),
            original_size=(original_h, original_w),
        )

    def _ensure_model(self, *, d_model: int) -> None:
        if self._reconstructor is not None:
            return
        self._d_model = int(d_model)
        self._reconstructor = _MambaReconstructor(
            d_model=int(d_model),
            n_layers=self.n_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        ).to(self.device)

    def fit(self, X: Iterable[ImageInput], y=None):
        items = list(X)
        if not items:
            raise ValueError("X must contain at least one training image.")

        torch = _require_torch()

        # Reproducibility for training noise, shuffle, init.
        rng = np.random.default_rng(self.random_seed)
        torch.manual_seed(int(self.random_seed))

        embedded_train = [self._embed(item) for item in items]
        d_model = int(embedded_train[0].patches.shape[1])
        self._ensure_model(d_model=d_model)
        if self._reconstructor is None:  # pragma: no cover
            raise RuntimeError("Internal error: reconstructor not initialized")

        # Build a simple tensor dataset of patch sequences: (N, P, D)
        sequences = [torch.from_numpy(e.patches).to(torch.float32) for e in embedded_train]
        batch = torch.stack(sequences, dim=0)  # (N, P, D)

        dataset = torch.utils.data.TensorDataset(batch)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )

        opt = torch.optim.AdamW(
            self._reconstructor.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = torch.nn.MSELoss(reduction="mean")

        self._reconstructor.train(True)
        for _epoch in range(int(self.epochs)):
            for (x,) in loader:
                x = x.to(self.device)
                if self.noise_std > 0:
                    noise = torch.randn_like(x) * float(self.noise_std)
                    x_in = x + noise
                else:
                    x_in = x

                pred = self._reconstructor(x_in)
                loss = loss_fn(pred, x)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        # Calibrate threshold on training images (PyOD-like semantics).
        self.decision_scores_ = self.decision_function(items)
        self._process_decision_scores()
        # BaseDeepLearningDetector sets threshold_ but keep our attribute in sync for tooling.
        self.threshold_ = getattr(self, "threshold_", None)
        return self

    def _patch_errors(self, embedded: _Embedded) -> NDArray:
        if self._reconstructor is None or self._d_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        torch = _require_torch()
        with torch.no_grad():
            x = _as_patch_batch(embedded.patches).to(self.device)  # (1, P, D)
            pred = self._reconstructor(x)  # (1, P, D)
            err = (pred - x).pow(2).mean(dim=-1)  # (1, P)
            out = err.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(out, dtype=np.float32)

    def decision_function(self, X: Iterable[ImageInput]) -> NDArray:
        items = list(X)
        scores = np.zeros(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            embedded = self._embed(item)
            patch_err = self._patch_errors(embedded)
            scores[i] = aggregate_patch_scores(
                patch_err,
                method=self.aggregation_method,
                topk=self.aggregation_topk,
            )
        return scores

    def predict(self, X: Iterable[ImageInput]) -> NDArray:
        if getattr(self, "threshold_", None) is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores >= float(self.threshold_)).astype(np.int64)

    def get_anomaly_map(self, image: ImageInput) -> NDArray:
        embedded = self._embed(image)
        patch_err = self._patch_errors(embedded)
        patch_grid = reshape_patch_scores(
            patch_err,
            grid_h=embedded.grid_shape[0],
            grid_w=embedded.grid_shape[1],
        )

        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required to upsample anomaly maps.\n"
                "Install it via:\n  pip install 'opencv-python'\n"
                f"Original error: {exc}"
            ) from exc

        original_h, original_w = embedded.original_size
        upsampled = cv2.resize(
            np.asarray(patch_grid, dtype=np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.asarray(upsampled, dtype=np.float32)

    def predict_anomaly_map(self, X: Iterable[ImageInput]) -> NDArray:
        items = list(X)
        maps = [self.get_anomaly_map(item) for item in items]
        if not maps:
            raise ValueError("X must be non-empty")

        first_shape = maps[0].shape
        for m in maps[1:]:
            if m.shape != first_shape:
                raise ValueError(
                    "Inconsistent anomaly map shapes; cannot stack. "
                    f"Expected {first_shape}, got {m.shape}."
                )
        return np.stack(maps, axis=0)

