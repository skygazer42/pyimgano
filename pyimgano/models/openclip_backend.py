from __future__ import annotations

"""OpenCLIP backend model skeletons.

This module intentionally has **no hard dependency** on `open_clip_torch`. The
OpenCLIP Python module (`open_clip`) is only imported at runtime when a model is
constructed, via `pyimgano.utils.optional_deps.require`.

The goal is to make `import pyimgano.models` safe even when optional deps are
not installed, while still registering model names so they can be discovered.
"""

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .anomalydino import (
    PatchEmbedder,
    VisionAnomalyDINO,
)

from .patchknn_core import aggregate_patch_scores, reshape_patch_scores

from pyimgano.utils.optional_deps import require

from .registry import register_model


def _require_open_clip(open_clip_module=None):
    if open_clip_module is not None:
        return open_clip_module
    return require("open_clip", extra="clip", purpose="OpenCLIP detectors")


def _l2_normalize(x: NDArray, *, axis: int, eps: float = 1e-12) -> NDArray:
    x_np = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x_np, axis=axis, keepdims=True)
    denom = np.maximum(denom, float(eps))
    return x_np / denom


_PromptScoreMode = Literal["diff", "ratio"]


def _prompt_patch_scores(
    patch_embeddings: NDArray,
    *,
    text_features_normal: NDArray,
    text_features_anomaly: NDArray,
    mode: _PromptScoreMode = "diff",
    eps: float = 1e-6,
) -> NDArray:
    """Compute per-patch anomaly scores from OpenCLIP-style embeddings.

    This helper is intentionally NumPy-only so it can be unit tested without
    `torch`/`open_clip` installed.
    """

    patches = np.asarray(patch_embeddings, dtype=np.float32)
    if patches.ndim != 2:
        raise ValueError(f"patch_embeddings must be 2D (num_patches, dim). Got {patches.shape}.")

    normal = np.asarray(text_features_normal, dtype=np.float32).reshape(-1)
    anomaly = np.asarray(text_features_anomaly, dtype=np.float32).reshape(-1)
    if normal.ndim != 1 or anomaly.ndim != 1:
        raise ValueError("text_features_normal/anomaly must be 1D feature vectors.")
    if normal.shape != anomaly.shape:
        raise ValueError(
            "text_features_normal and text_features_anomaly must have the same shape. "
            f"Got {normal.shape} vs {anomaly.shape}."
        )
    if patches.shape[1] != normal.shape[0]:
        raise ValueError(
            "Embedding dimension mismatch between patches and text features. "
            f"Got patches dim={patches.shape[1]} vs text dim={normal.shape[0]}."
        )

    patches_norm = _l2_normalize(patches, axis=1, eps=float(eps))
    normal_norm = _l2_normalize(normal, axis=0, eps=float(eps)).reshape(-1)
    anomaly_norm = _l2_normalize(anomaly, axis=0, eps=float(eps)).reshape(-1)

    sim_normal = patches_norm @ normal_norm
    sim_anomaly = patches_norm @ anomaly_norm

    mode_lower = str(mode).lower()
    if mode_lower == "diff":
        return np.asarray(sim_anomaly - sim_normal, dtype=np.float32)
    if mode_lower == "ratio":
        eps_float = float(eps)
        return np.asarray((sim_anomaly + eps_float) / (sim_normal + eps_float), dtype=np.float32)

    raise ValueError(f"Unknown mode: {mode}. Choose from: diff, ratio")


def _load_openclip_model_and_preprocess(
    *,
    open_clip_module=None,
    model_name: str,
    pretrained: Optional[str],
    device: str,
    force_image_size: Optional[int] = None,
):
    """Load an OpenCLIP model + preprocess lazily (best-effort API compatibility)."""

    open_clip = _require_open_clip(open_clip_module)

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required to use OpenCLIP detectors.\n"
            "Install it via:\n  pip install 'torch'\n"
            f"Original error: {exc}"
        ) from exc

    device_t = torch.device(str(device))
    kwargs: dict[str, Any] = {}
    if force_image_size is not None:
        kwargs["force_image_size"] = int(force_image_size)

    result = open_clip.create_model_and_transforms(
        str(model_name),
        pretrained=pretrained,
        **kwargs,
    )
    if not isinstance(result, tuple):
        raise RuntimeError(
            "Unexpected return value from open_clip.create_model_and_transforms: "
            f"{type(result)}"
        )

    if len(result) == 3:
        model, _preprocess_train, preprocess_val = result
        preprocess = preprocess_val
    elif len(result) == 2:
        model, preprocess = result
    else:
        raise RuntimeError(
            "Unexpected return arity from open_clip.create_model_and_transforms: "
            f"{len(result)}"
        )

    model.eval()
    model = model.to(device_t)
    return model, preprocess, device_t


def _run_openclip_transformer(transformer: Any, x: Any) -> Any:
    """Run an OpenCLIP transformer in either (B, N, C) or (N, B, C) mode."""

    # Strategy 1: some implementations accept (B, N, C).
    try:
        out = transformer(x)
        if isinstance(out, tuple):
            out = out[0]
        return out
    except Exception:
        pass

    # Strategy 2: OpenAI CLIP style expects (N, B, C).
    try:
        x_t = x.permute(1, 0, 2)
        out = transformer(x_t)
        if isinstance(out, tuple):
            out = out[0]
        return out.permute(1, 0, 2)
    except Exception as exc:
        raise RuntimeError(
            "Failed to run OpenCLIP visual transformer. This may be an unsupported "
            "OpenCLIP version / model architecture."
        ) from exc


class OpenCLIPViTPatchEmbedder:
    """Patch-token embedder for OpenCLIP ViT models.

    Notes
    -----
    - This class is intentionally **lazy**: it does not import torch/open_clip or
      load weights until the first call to :meth:`embed`.
    - It only supports OpenCLIP ViT visual backbones (e.g. ``ViT-B-32``).
    """

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: Optional[str] = "laion2b_s34b_b79k",
        device: str = "cpu",
        force_image_size: Optional[int] = None,
        normalize: bool = True,
        open_clip_module=None,
        model: Any = None,
        preprocess: Any = None,
    ) -> None:
        self.model_name = str(model_name)
        self.pretrained = pretrained
        self.device = str(device)
        self.force_image_size = force_image_size
        self.normalize = bool(normalize)
        self._open_clip = open_clip_module

        self._model: Any = model
        self._preprocess: Any = preprocess
        self._device_t: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._preprocess is not None:
            return
        model, preprocess, device_t = _load_openclip_model_and_preprocess(
            open_clip_module=self._open_clip,
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self.device,
            force_image_size=self.force_image_size,
        )
        self._model = model
        self._preprocess = preprocess
        self._device_t = device_t

    def get_model_and_preprocess(self):
        self._ensure_loaded()
        return self._model, self._preprocess, self._device_t

    def _extract_vit_patch_tokens(self, image_tensor):
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use OpenCLIP detectors.\n"
                "Install it via:\n  pip install 'torch'\n"
                f"Original error: {exc}"
            ) from exc

        if self._model is None:  # pragma: no cover - guarded by _ensure_loaded
            raise RuntimeError("OpenCLIP model not loaded")

        visual = getattr(self._model, "visual", None)
        if visual is None:
            raise RuntimeError("OpenCLIP model has no `.visual` attribute")

        required_attrs = ("conv1", "class_embedding", "positional_embedding", "transformer")
        if not all(hasattr(visual, name) for name in required_attrs):
            raise ValueError(
                "OpenCLIPViTPatchEmbedder only supports OpenCLIP ViT visual backbones. "
                f"Missing one of: {required_attrs}"
            )

        x = visual.conv1(image_tensor)  # (B, C, Gh, Gw)
        grid_h, grid_w = int(x.shape[-2]), int(x.shape[-1])
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, N, C)

        class_embedding = visual.class_embedding
        if getattr(class_embedding, "ndim", None) == 1:
            class_token = (
                class_embedding.to(dtype=x.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(x.shape[0], 1, -1)
            )
        else:  # pragma: no cover - rare variants
            class_token = class_embedding.to(dtype=x.dtype)
            if class_token.ndim == 2:
                class_token = class_token.unsqueeze(0).expand(x.shape[0], -1, -1)

        x = torch.cat([class_token, x], dim=1)  # (B, 1+N, C)

        pos = visual.positional_embedding
        if getattr(pos, "ndim", None) == 2:
            pos = pos.unsqueeze(0)
        pos = pos.to(dtype=x.dtype)
        if pos.shape[1] != x.shape[1]:  # pragma: no cover - unexpected
            raise RuntimeError(
                "OpenCLIP position embedding length does not match token length. "
                f"Got pos={tuple(pos.shape)} vs tokens={tuple(x.shape)}."
            )
        x = x + pos

        patch_dropout = getattr(visual, "patch_dropout", None)
        if patch_dropout is not None:
            try:
                x = patch_dropout(x)
            except Exception:
                pass

        ln_pre = getattr(visual, "ln_pre", None)
        if ln_pre is not None:
            x = ln_pre(x)

        x = _run_openclip_transformer(visual.transformer, x)

        ln_post = getattr(visual, "ln_post", None)
        if ln_post is not None:
            x = ln_post(x)

        proj = getattr(visual, "proj", None)
        if proj is not None:
            try:
                x = x @ proj
            except Exception:
                pass

        patch_tokens = x[:, 1:, :]  # drop CLS token
        if self.normalize:
            denom = torch.linalg.norm(patch_tokens, dim=-1, keepdim=True)
            denom = torch.clamp(denom, min=1e-12)
            patch_tokens = patch_tokens / denom

        return patch_tokens, (grid_h, grid_w)

    def embed(self, image_path: str):
        self._ensure_loaded()

        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Pillow is required to load images for OpenCLIP.\n"
                "Install it via:\n  pip install 'pillow'\n"
                f"Original error: {exc}"
            ) from exc

        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use OpenCLIP detectors.\n"
                "Install it via:\n  pip install 'torch'\n"
                f"Original error: {exc}"
            ) from exc

        if self._preprocess is None or self._device_t is None:  # pragma: no cover
            raise RuntimeError("OpenCLIP preprocess not loaded")

        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size

        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device_t)
        with torch.no_grad():
            patch_tokens, grid_shape = self._extract_vit_patch_tokens(image_tensor)

        patch_embeddings = patch_tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return patch_embeddings, grid_shape, (int(original_h), int(original_w))


def _encode_openclip_text_features(
    *,
    open_clip: Any,
    model: Any,
    device_t: Any,
    prompts: list[str],
) -> NDArray:
    if not prompts:
        raise ValueError("prompts must be non-empty")

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required to use OpenCLIP detectors.\n"
            "Install it via:\n  pip install 'torch'\n"
            f"Original error: {exc}"
        ) from exc

    tokens = open_clip.tokenize(prompts).to(device_t)
    with torch.no_grad():
        features = model.encode_text(tokens)
        denom = torch.linalg.norm(features, dim=-1, keepdim=True)
        denom = torch.clamp(denom, min=1e-12)
        features = features / denom

    mean_feature = features.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return mean_feature


@register_model(
    "vision_openclip_promptscore",
    tags=("vision", "deep", "clip", "openclip", "backend", "prompt"),
    metadata={
        "description": "OpenCLIP prompt scoring detector (requires pyimgano[clip])",
        "backend": "openclip",
    },
)
class VisionOpenCLIPPromptScore:
    """OpenCLIP prompt-score based vision detector.

    Parameters
    ----------
    open_clip_module : optional
        Dependency injection hook. When provided, avoids importing `open_clip`.
        Intended for unit tests and advanced callers.
    """

    _DEFAULT_TEXT_PROMPTS = {
        "normal": [
            "a photo of a normal {}",
            "a high-quality photo of a {}",
            "a photo of a perfect {}",
        ],
        "anomaly": [
            "a photo of a damaged {}",
            "a photo of a defective {}",
            "a photo of an anomalous {}",
        ],
    }

    def __init__(
        self,
        *,
        embedder: Optional[PatchEmbedder] = None,
        text_features_normal: Optional[NDArray] = None,
        text_features_anomaly: Optional[NDArray] = None,
        class_name: str = "object",
        text_prompts: Optional[dict[str, list[str]]] = None,
        openclip_model_name: str = "ViT-B-32",
        openclip_pretrained: Optional[str] = "laion2b_s34b_b79k",
        device: str = "cpu",
        force_image_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        contamination: float = 0.1,
        aggregation_method: str = "topk_mean",
        aggregation_topk: float = 0.01,
        mode: _PromptScoreMode = "diff",
        open_clip_module=None,
        **kwargs: Any,
    ) -> None:
        self.embedder = embedder
        self.text_features_normal = text_features_normal
        self.text_features_anomaly = text_features_anomaly
        self.class_name = str(class_name)
        self.text_prompts = dict(text_prompts) if text_prompts is not None else dict(self._DEFAULT_TEXT_PROMPTS)
        self.openclip_model_name = str(openclip_model_name)
        self.openclip_pretrained = openclip_pretrained
        self.device = str(device)
        self.force_image_size = force_image_size
        self.normalize_embeddings = bool(normalize_embeddings)
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5). Got {self.contamination}."
            )

        self.aggregation_method = str(aggregation_method)
        self.aggregation_topk = float(aggregation_topk)
        self.mode = mode
        self._kwargs = dict(kwargs)
        self._open_clip = open_clip_module
        self._text_cache_key: Optional[tuple[Any, ...]] = None
        self._manual_text_features = (
            self.text_features_normal is not None and self.text_features_anomaly is not None
        )

        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

        if self.embedder is None:
            open_clip = _require_open_clip(open_clip_module)
            self._open_clip = open_clip
            self.embedder = OpenCLIPViTPatchEmbedder(
                open_clip_module=open_clip,
                model_name=self.openclip_model_name,
                pretrained=self.openclip_pretrained,
                device=self.device,
                force_image_size=self.force_image_size,
                normalize=self.normalize_embeddings,
            )
        else:
            # If callers provide a custom embedder, require explicit text features
            # so patch embeddings and text embeddings are in the same space.
            if self.text_features_normal is None or self.text_features_anomaly is None:
                raise ValueError(
                    "text_features_normal and text_features_anomaly are required when providing a custom embedder."
                )

    def set_class_name(self, class_name: str):
        class_name_str = str(class_name)
        if self.class_name == class_name_str:
            return self

        self.class_name = class_name_str
        if not self._manual_text_features:
            self.text_features_normal = None
            self.text_features_anomaly = None
            self._text_cache_key = None
        return self

    def _format_prompts(self, templates: list[str]) -> list[str]:
        formatted: list[str] = []
        for template in templates:
            try:
                formatted.append(str(template).format(self.class_name))
            except Exception:
                formatted.append(str(template).format(class_name=self.class_name))
        return formatted

    def _ensure_text_features(self) -> None:
        if self.text_features_normal is not None and self.text_features_anomaly is not None:
            return

        if self.embedder is None:
            raise RuntimeError("embedder is required")

        if not isinstance(self.embedder, OpenCLIPViTPatchEmbedder):
            raise RuntimeError(
                "Automatic text feature encoding requires the default OpenCLIP embedder. "
                "Provide text_features_normal/text_features_anomaly explicitly."
            )

        open_clip = _require_open_clip(self._open_clip)
        model, _preprocess, device_t = self.embedder.get_model_and_preprocess()

        normal_templates = list(self.text_prompts.get("normal", []))
        anomaly_templates = list(self.text_prompts.get("anomaly", []))
        key = (
            self.class_name,
            tuple(normal_templates),
            tuple(anomaly_templates),
            self.openclip_model_name,
            self.openclip_pretrained,
        )
        if self._text_cache_key == key:
            return

        normal_prompts = self._format_prompts(normal_templates)
        anomaly_prompts = self._format_prompts(anomaly_templates)
        self.text_features_normal = _encode_openclip_text_features(
            open_clip=open_clip,
            model=model,
            device_t=device_t,
            prompts=normal_prompts,
        )
        self.text_features_anomaly = _encode_openclip_text_features(
            open_clip=open_clip,
            model=model,
            device_t=device_t,
            prompts=anomaly_prompts,
        )
        self._text_cache_key = key

    def _embed(self, image_path: str) -> tuple[NDArray, tuple[int, int], tuple[int, int]]:
        if self.embedder is None:  # pragma: no cover - guarded by __init__
            raise RuntimeError("embedder is required")

        patch_embeddings, grid_shape, original_size = self.embedder.embed(image_path)
        patch_embeddings_np = np.asarray(patch_embeddings, dtype=np.float32)
        if patch_embeddings_np.ndim != 2:
            raise ValueError(
                f"Expected 2D patch embeddings, got shape {patch_embeddings_np.shape}"
            )

        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if patch_embeddings_np.shape[0] != grid_h * grid_w:
            raise ValueError(
                "Patch embedding count does not match grid shape. "
                f"Got {patch_embeddings_np.shape[0]} patches for grid {grid_h}x{grid_w}."
            )

        original_h, original_w = int(original_size[0]), int(original_size[1])
        if original_h <= 0 or original_w <= 0:
            raise ValueError(f"Invalid original_size: {original_size}")

        return patch_embeddings_np, (grid_h, grid_w), (original_h, original_w)

    def fit(self, X, y=None):
        paths = list(X)
        if not paths:
            raise ValueError("X must contain at least one training image path.")

        self._ensure_text_features()
        self.decision_scores_ = np.asarray(self.decision_function(paths), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def decision_function(self, X):
        self._ensure_text_features()
        if self.text_features_normal is None or self.text_features_anomaly is None:  # pragma: no cover
            raise RuntimeError("text_features_normal/text_features_anomaly are required")

        paths = list(X)
        scores = np.zeros(len(paths), dtype=np.float64)
        for i, path in enumerate(paths):
            patch_embeddings, _grid_shape, _original_size = self._embed(path)
            patch_scores = _prompt_patch_scores(
                patch_embeddings,
                text_features_normal=self.text_features_normal,
                text_features_anomaly=self.text_features_anomaly,
                mode=self.mode,
            )
            scores[i] = aggregate_patch_scores(
                patch_scores,
                method=self.aggregation_method,
                topk=self.aggregation_topk,
            )
        return scores

    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(np.int64)

    def get_anomaly_map(self, image_path: str) -> NDArray:
        self._ensure_text_features()
        if self.text_features_normal is None or self.text_features_anomaly is None:  # pragma: no cover
            raise RuntimeError("text_features_normal/text_features_anomaly are required")

        patch_embeddings, grid_shape, original_size = self._embed(image_path)
        patch_scores = _prompt_patch_scores(
            patch_embeddings,
            text_features_normal=self.text_features_normal,
            text_features_anomaly=self.text_features_anomaly,
            mode=self.mode,
        )
        patch_grid = reshape_patch_scores(patch_scores, grid_h=grid_shape[0], grid_w=grid_shape[1])

        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required to upsample anomaly maps.\n"
                "Install it via:\n  pip install 'opencv-python'\n"
                f"Original error: {exc}"
            ) from exc

        original_h, original_w = original_size
        upsampled = cv2.resize(
            np.asarray(patch_grid, dtype=np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.asarray(upsampled, dtype=np.float32)

    def predict_anomaly_map(self, X):
        paths = list(X)
        maps = [self.get_anomaly_map(path) for path in paths]
        if not maps:
            raise ValueError("X must be non-empty")

        first_shape = maps[0].shape
        for m in maps[1:]:
            if m.shape != first_shape:
                raise ValueError(
                    "Inconsistent anomaly map shapes. "
                    f"Expected {first_shape}, got {m.shape}."
                )
        return np.stack(maps)


@register_model(
    "vision_openclip_patchknn",
    tags=("vision", "deep", "clip", "openclip", "backend", "knn"),
    metadata={
        "description": "OpenCLIP patch embedding + kNN detector (requires pyimgano[clip])",
        "backend": "openclip",
    },
)
class VisionOpenCLIPPatchKNN:
    """Skeleton for an OpenCLIP patch embedding + kNN detector.

    Parameters
    ----------
    open_clip_module : optional
        Dependency injection hook. When provided, avoids importing `open_clip`.
    """

    def __init__(
        self,
        *,
        open_clip_module=None,
        embedder: Optional[PatchEmbedder] = None,
        knn_index: Optional[Any] = None,
        openclip_model_name: str = "ViT-B-32",
        openclip_pretrained: Optional[str] = "laion2b_s34b_b79k",
        device: str = "cpu",
        force_image_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        # This detector is implemented as an AnomalyDINO-style patch-kNN model.
        # To keep tests lightweight, callers can inject a pure-numpy embedder.
        if embedder is None:
            open_clip = _require_open_clip(open_clip_module)
            embedder = OpenCLIPViTPatchEmbedder(
                open_clip_module=open_clip,
                model_name=str(openclip_model_name),
                pretrained=openclip_pretrained,
                device=str(device),
                force_image_size=force_image_size,
                normalize=bool(normalize_embeddings),
            )

        self._open_clip = open_clip_module
        self._knn_index = knn_index
        self._kwargs = dict(kwargs)
        self._core = VisionAnomalyDINO(embedder=embedder, device=str(device), **kwargs)

    def fit(self, X, y=None):  # pragma: no cover - skeleton API
        self._core.fit(X, y=y)
        return self

    def decision_function(self, X):  # pragma: no cover - skeleton API
        return self._core.decision_function(X)

    def predict(self, X):
        return self._core.predict(X)

    def get_anomaly_map(self, image_path: str):
        return self._core.get_anomaly_map(image_path)

    def predict_anomaly_map(self, X):
        return self._core.predict_anomaly_map(X)

    @property
    def decision_scores_(self):
        return self._core.decision_scores_

    @property
    def threshold_(self):
        return self._core.threshold_
