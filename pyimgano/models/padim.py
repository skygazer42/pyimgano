"""
PaDiM: Patch Distribution Modeling for industrial anomaly detection.

PaDiM models per-location (patch) feature distributions from a pretrained
backbone and scores anomalies via Mahalanobis distance.

Notes for this implementation:
- Fits on `list[str]` image paths (unified pyimgano interface).
- Supports pixel-level anomaly maps via `get_anomaly_map()` and
  `predict_anomaly_map()`.
- Keeps the historical registry name `padim` for compatibility and adds the
  canonical name `vision_padim` to match docs/CLI expectations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import require
from pyimgano.utils.torchvision_safe import load_torchvision_model

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseCv import BaseVisionDeepDetector
from .deep_io import safe_torch_load
from .registry import register_model

ImageInput = Union[str, np.ndarray]

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _build_torchvision_backbone(name: str, *, pretrained: bool) -> torch.nn.Module:
    if name not in {"resnet18", "resnet50", "wide_resnet50_2"}:
        raise ValueError(
            f"Unsupported backbone: {name!r}. Choose from: " "resnet18, wide_resnet50_2, resnet50"
        )
    model, _ = load_torchvision_model(name, pretrained=bool(pretrained))
    return model


@register_model(
    "vision_padim",
    tags=("vision", "deep", "patch", "distribution", "numpy", "pixel_map"),
    metadata={
        "description": "PaDiM core algorithm with fixed channel subsampling and per-location Gaussians",
        "paper": "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization",
        "paper_url": "https://arxiv.org/abs/2011.08785",
        "year": 2020,
        "supervision": "one-class",
        "implementation_status": "paper-resnet-paths-and-statistics-aligned",
        "paper_fidelity": "core-aligned",
    },
)
@register_model(
    "padim",
    tags=("vision", "deep", "patch", "distribution", "numpy", "pixel_map"),
    metadata={
        "description": "PaDiM (legacy alias) - patch distribution modeling",
        "paper": "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization",
        "paper_url": "https://arxiv.org/abs/2011.08785",
        "year": 2020,
        "supervision": "one-class",
        "implementation_status": "paper-resnet-paths-and-statistics-aligned",
        "paper_fidelity": "core-aligned",
    },
)
class VisionPaDiM(BaseVisionDeepDetector):
    """PaDiM-style anomaly detector (feature distribution per patch location)."""

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = "resnet18",
        d_reduced: Optional[int] = None,
        image_size: int = 224,
        resize_size: Optional[int] = None,
        pretrained: bool = False,
        device: str = "cpu",
        covariance_eps: float = 0.01,
        gaussian_sigma: float = 4.0,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        # Keep module import light: only require deep deps when instantiating.
        require("torch", extra="torch", purpose="VisionPaDiM")
        transforms = require(
            "torchvision.transforms",
            extra="torch",
            purpose="VisionPaDiM torchvision transforms",
        )

        super().__init__(contamination=contamination, **kwargs)

        backbone_name = str(backbone).strip()
        if backbone_name == "wide_resnet50":
            backbone_name = "wide_resnet50_2"
        if d_reduced is None:
            d_reduced = 100 if backbone_name == "resnet18" else 550
        if d_reduced < 1:
            raise ValueError(f"d_reduced must be >= 1, got {d_reduced}")
        if image_size < 32:
            raise ValueError(f"image_size must be >= 32, got {image_size}")
        if resize_size is not None and int(resize_size) < image_size:
            raise ValueError(f"resize_size must be >= image_size ({image_size}), got {resize_size}")
        if covariance_eps <= 0:
            raise ValueError(f"covariance_eps must be > 0, got {covariance_eps}")
        if gaussian_sigma < 0:
            raise ValueError(f"gaussian_sigma must be >= 0, got {gaussian_sigma}")

        self.backbone_name = backbone_name
        self.d_reduced = int(d_reduced)
        self.image_size = int(image_size)
        self.resize_size = (
            int(resize_size)
            if resize_size is not None
            else max(self.image_size, int(round(self.image_size * 256 / 224)))
        )
        self.pretrained = bool(pretrained)
        self.device = str(device)
        self.covariance_eps = float(covariance_eps)
        self.gaussian_sigma = float(gaussian_sigma)
        self.random_state = int(random_state)

        self.model = _build_torchvision_backbone(self.backbone_name, pretrained=self.pretrained)
        self.model.eval()
        self.model.to(self.device)

        self.feature_maps: dict[str, torch.Tensor] = {}
        self._register_hooks()

        self.feature_indices_: Optional[NDArray] = None

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (self.resize_size, self.resize_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.means: Optional[NDArray] = None
        self.inv_covs: Optional[NDArray] = None
        self.patch_shape: Optional[Tuple[int, int]] = None

    def save_checkpoint(self, path: str | Path) -> Path:
        self._check_fitted()

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="VisionPaDiM checkpoint saving")

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
                "schema_version": 3,
                "model_state_dict": model_state_dict,
                "feature_indices": torch.as_tensor(
                    np.asarray(self.feature_indices_, dtype=np.int64), dtype=torch.int64
                ),
                "means": torch.as_tensor(
                    np.asarray(self.means, dtype=np.float32), dtype=torch.float32
                ),
                "inv_covs": torch.as_tensor(
                    np.asarray(self.inv_covs, dtype=np.float32), dtype=torch.float32
                ),
                "patch_shape": [int(v) for v in cast(tuple[int, int], self.patch_shape)],
                "decision_scores_": torch.as_tensor(
                    np.asarray(self.decision_scores_, dtype=np.float64), dtype=torch.float64
                ),
                "threshold_": float(self.threshold_),
                "gaussian_sigma": float(self.gaussian_sigma),
            },
            out_path,
        )
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        state = safe_torch_load(Path(path), map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Invalid VisionPaDiM checkpoint payload.")
        if int(state.get("schema_version", 0)) != 3:
            raise ValueError(
                "Unsupported legacy PaDiM checkpoint: older checkpoints used a different "
                "channel-selection or cross-level feature-alignment contract. Refit and save "
                "it again."
            )

        model_state_dict = state.get("model_state_dict", None)
        if not isinstance(model_state_dict, dict):
            raise ValueError("VisionPaDiM checkpoint is missing model_state_dict.")
        self.model.load_state_dict(dict(model_state_dict), strict=False)
        self.model.to(self.device)
        self.model.eval()

        feature_indices = np.asarray(state.get("feature_indices", []), dtype=np.int64).reshape(-1)
        if feature_indices.size != self.d_reduced:
            raise ValueError(
                "VisionPaDiM checkpoint feature subset does not match d_reduced. "
                f"Expected {self.d_reduced}, got {feature_indices.size}."
            )
        self.feature_indices_ = feature_indices

        self.means = np.asarray(state["means"], dtype=np.float32)
        self.inv_covs = np.asarray(state["inv_covs"], dtype=np.float32)
        patch_shape = state.get("patch_shape", None)
        if not isinstance(patch_shape, (list, tuple)) or len(patch_shape) != 2:
            raise ValueError("VisionPaDiM checkpoint is missing patch_shape.")
        self.patch_shape = (int(patch_shape[0]), int(patch_shape[1]))
        self.decision_scores_ = np.asarray(state["decision_scores_"], dtype=np.float64)
        self.threshold_ = float(state["threshold_"])
        self.gaussian_sigma = float(state.get("gaussian_sigma", self.gaussian_sigma))

    def _register_hooks(self) -> None:
        def get_activation(name: str):
            def hook(_module, _input, output):
                self.feature_maps[name] = output.detach()

            return hook

        for layer in ("layer1", "layer2", "layer3"):
            if not hasattr(self.model, layer):
                raise ValueError(f"Backbone {self.backbone_name!r} has no layer {layer!r}")
            getattr(self.model, layer).register_forward_hook(get_activation(layer))

    def _load_image_rgb(self, image_path: ImageInput) -> NDArray:
        cv2 = require("cv2", purpose="VisionPaDiM image loading")
        if isinstance(image_path, np.ndarray):
            if image_path.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB image, got dtype={image_path.dtype}")
            if image_path.ndim != 3 or image_path.shape[2] != 3:
                raise ValueError(f"Expected shape (H,W,3), got {image_path.shape}")
            return np.ascontiguousarray(image_path)

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _extract_patch_features(self, image_path: ImageInput) -> NDArray:
        torch = require("torch", extra="torch", purpose="VisionPaDiM feature extraction")
        functional = require(
            "torch.nn.functional", extra="torch", purpose="VisionPaDiM feature extraction"
        )

        img = self._load_image_rgb(image_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model(img_tensor)

        layer1_feat = self.feature_maps["layer1"]  # (1, C1, H1, W1)
        layer2_feat = self.feature_maps["layer2"]  # (1, C2, H2, W2)
        layer3_feat = self.feature_maps["layer3"]  # (1, C3, H3, W3)

        layer2_feat = functional.interpolate(
            layer2_feat,
            size=layer1_feat.shape[-2:],
            mode="nearest",
        )
        layer3_feat = functional.interpolate(
            layer3_feat,
            size=layer1_feat.shape[-2:],
            mode="nearest",
        )

        features = torch.cat([layer1_feat, layer2_feat, layer3_feat], dim=1)
        _b, c, h, w = features.shape
        self.patch_shape = (int(h), int(w))

        # (H*W, C)
        return features.permute(0, 2, 3, 1).reshape(h * w, c).cpu().numpy()

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionPaDiM":
        del y
        x_iter = cast(Iterable[ImageInput], resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        x_list = list(x_iter)
        if not x_list:
            raise ValueError("Training set cannot be empty")

        self.feature_indices_ = None
        reduced_features: list[NDArray] = []
        for image in x_list:
            feat = self._extract_patch_features(image)
            if self.feature_indices_ is None:
                feature_dim = int(feat.shape[1])
                if self.d_reduced > feature_dim:
                    raise ValueError(
                        f"d_reduced={self.d_reduced} exceeds feature dimension {feature_dim}."
                    )
                rng = np.random.default_rng(self.random_state)
                self.feature_indices_ = np.sort(
                    rng.choice(feature_dim, size=self.d_reduced, replace=False)
                ).astype(np.int64, copy=False)
            reduced_features.append(feat[:, self.feature_indices_])

        all_features = np.stack(reduced_features, axis=0)  # (N, P, D)
        means = np.mean(all_features, axis=0).astype(np.float32, copy=False)  # (P, D)

        n_images, n_patches, d = all_features.shape
        if d != self.d_reduced:
            raise RuntimeError(f"Expected reduced dim={self.d_reduced}, got {d}")

        inv_covs = np.empty((n_patches, d, d), dtype=np.float32)
        eye = np.eye(d, dtype=np.float32)

        # Per-location covariance. For small N, fall back to diagonal eps.
        for i in range(n_patches):
            patch_feats = all_features[:, i, :]
            if n_images < 2:
                cov = eye * self.covariance_eps
            else:
                cov = np.cov(patch_feats, rowvar=False).astype(np.float32, copy=False)
                cov = cov + eye * self.covariance_eps
            inv_covs[i] = np.linalg.inv(cov).astype(np.float32, copy=False)

        self.means = means
        self.inv_covs = inv_covs

        # Calibrate threshold for `predict()`.
        self.decision_scores_ = self.decision_function(x_list)
        self._process_decision_scores()
        return self

    def _check_fitted(self) -> None:
        if (
            self.means is None
            or self.inv_covs is None
            or self.patch_shape is None
            or self.feature_indices_ is None
        ):
            raise RuntimeError("Model not fitted. Call fit() first.")

    def _compute_patch_distances(self, image_path: ImageInput) -> NDArray:
        self._check_fitted()
        feat = self._extract_patch_features(image_path)
        reduced = feat[:, self.feature_indices_].astype(np.float32, copy=False)

        diff = reduced - self.means  # (P, D)
        q = np.einsum("pd,pde,pe->p", diff, self.inv_covs, diff)
        q = np.maximum(q, 0.0)
        return np.sqrt(q, dtype=np.float32)

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        # This detector scores one image at a time. Keep `batch_size` for
        # interface compatibility with BaseDeepLearningDetector.
        if batch_size is not None:
            batch_size_int = int(batch_size)
            if batch_size_int <= 0:
                raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")

        self._check_fitted()
        x_iter = cast(
            Iterable[ImageInput],
            resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
        )
        x_list = list(x_iter)
        return np.asarray(
            [float(self._compute_anomaly_map(image).max()) for image in x_list],
            dtype=np.float32,
        )

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )

        if not hasattr(self, "threshold_"):
            raise RuntimeError("Model not fitted. Call fit() first.")
        x_iter = cast(
            Iterable[ImageInput], resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        )
        scores = self.decision_function(x_iter)
        return (scores >= self.threshold_).astype(int)

    def _compute_anomaly_map(self, image_path: ImageInput) -> NDArray:
        cv2 = require("cv2", purpose="VisionPaDiM anomaly map upsampling")
        distances = self._compute_patch_distances(image_path)
        h, w = self.patch_shape or (0, 0)
        if h * w != distances.shape[0]:
            raise RuntimeError(
                f"Patch shape mismatch: {self.patch_shape} -> {h*w} != {distances.shape[0]}"
            )

        low_res = distances.reshape(h, w)
        anomaly_map = cv2.resize(
            low_res,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        if self.gaussian_sigma > 0:
            anomaly_map = cv2.GaussianBlur(
                anomaly_map,
                (0, 0),
                sigmaX=self.gaussian_sigma,
                sigmaY=self.gaussian_sigma,
            )
        return anomaly_map.astype(np.float32, copy=False)

    def get_anomaly_map(self, image_path: ImageInput) -> NDArray:
        return self._compute_anomaly_map(image_path)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        x_iter = cast(
            Iterable[ImageInput],
            resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map"),
        )
        maps = [self.get_anomaly_map(item) for item in x_iter]
        return np.stack(maps, axis=0)
