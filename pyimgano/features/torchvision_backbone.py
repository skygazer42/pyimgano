from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor


_InputColor = Literal["rgb", "bgr"]


def _as_pil_rgb(item: Any, *, input_color: _InputColor):  # noqa: ANN001, ANN201
    # Import lazily: keep registry discovery light.
    from PIL import Image

    if isinstance(item, (str, Path)):
        img = Image.open(str(item)).convert("RGB")
        return img

    if isinstance(item, np.ndarray):
        arr = np.asarray(item)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (1, 3):
            raise ValueError(f"Unsupported numpy image shape: {arr.shape}")

        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32, copy=False)
            if float(np.nanmax(arr_f)) <= 1.0:
                arr_f = arr_f * 255.0
            arr_f = np.clip(arr_f, 0.0, 255.0)
            arr = arr_f.astype(np.uint8)

        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        if input_color == "bgr":
            arr = arr[..., ::-1]  # BGR -> RGB

        return Image.fromarray(arr, mode="RGB")

    raise TypeError(
        "TorchvisionBackboneExtractor expects inputs of type str|Path|np.ndarray, "
        f"got {type(item)}"
    )


def _make_device(device: str):  # noqa: ANN001, ANN201
    import torch

    dev = str(device).strip().lower()
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(dev)


def _load_torchvision_backbone(backbone: str, *, pretrained: bool):
    import torch.nn as nn
    import torchvision.models as models

    name = str(backbone).strip()

    if hasattr(models, "get_model") and hasattr(models, "get_model_weights"):
        weights = None
        if pretrained:
            weights_enum = models.get_model_weights(name)
            weights = weights_enum.DEFAULT
        model = models.get_model(name, weights=weights)
        transform = weights.transforms() if weights is not None else None
    else:  # pragma: no cover - fallback for older torchvision
        ctor = getattr(models, name)
        model = ctor(pretrained=bool(pretrained))
        transform = None

    # Remove classification head to expose embeddings.
    if hasattr(model, "fc"):
        model.fc = nn.Identity()  # type: ignore[attr-defined]
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()  # type: ignore[attr-defined]
    elif hasattr(model, "head"):
        model.head = nn.Identity()  # type: ignore[attr-defined]

    return model, transform


@dataclass
@register_feature_extractor(
    "torchvision_backbone",
    tags=("embeddings", "torch", "torchvision", "deep-features"),
    metadata={
        "description": "Torchvision backbone embeddings (global pooled; classifier head removed)",
    },
)
class TorchvisionBackboneExtractor(BaseFeatureExtractor):
    """Extract global embeddings from a torchvision backbone.

    Notes
    -----
    - By default, `pretrained=False` to avoid implicit weight downloads.
      Set `pretrained=True` explicitly if you want ImageNet weights and your
      environment can download them (or already has the weights cached).
    """

    backbone: str = "resnet18"
    pretrained: bool = False
    device: str = "cpu"
    batch_size: int = 16
    input_color: _InputColor = "rgb"

    # When not using weights-provided transforms, fall back to this size+norm.
    image_size: int = 224

    def __post_init__(self) -> None:
        # Lazy init: don't load weights / models during registry import.
        self._model = None
        self._transform = None
        self._device = None

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        import torch
        import torchvision.transforms as T

        model, weight_transform = _load_torchvision_backbone(
            str(self.backbone), pretrained=bool(self.pretrained)
        )
        dev = _make_device(str(self.device))
        model.to(dev)
        model.eval()

        if weight_transform is not None:
            transform = weight_transform
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            transform = T.Compose(
                [
                    T.Resize((int(self.image_size), int(self.image_size))),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]
            )

        self._model = model
        self._transform = transform
        self._device = dev
        self._torch = torch

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        items = list(inputs)
        if not items:
            return np.zeros((0, 1), dtype=np.float64)

        self._ensure_ready()
        assert self._model is not None
        assert self._transform is not None
        assert self._device is not None

        torch = self._torch
        bs = max(1, int(self.batch_size))

        rows: list[np.ndarray] = []
        from pyimgano.utils.torch_infer import torch_inference

        with torch_inference(self._model):
            for i in range(0, len(items), bs):
                batch_items = items[i : i + bs]
                pil_imgs = [_as_pil_rgb(it, input_color=self.input_color) for it in batch_items]
                x = torch.stack([self._transform(im) for im in pil_imgs], dim=0)
                x = x.to(self._device)
                out = self._model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                out_t = torch.as_tensor(out)
                if out_t.ndim > 2:
                    out_t = torch.flatten(out_t, start_dim=1)
                out_np = out_t.detach().cpu().numpy().astype(np.float64, copy=False)
                rows.append(out_np)

        feats = np.concatenate(rows, axis=0)
        return np.asarray(feats, dtype=np.float64)
