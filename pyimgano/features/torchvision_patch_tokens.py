from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor

from .torchvision_backbone import _as_pil_rgb, _load_torchvision_backbone, _make_device


_InputColor = Literal["rgb", "bgr"]


@dataclass
@register_feature_extractor(
    "torchvision_patch_tokens",
    tags=("embeddings", "torch", "torchvision", "patch", "deep-features"),
    metadata={
        "description": "Torchvision backbone patch tokens from a conv feature map (flattened per image)",
    },
)
class TorchvisionPatchTokensExtractor(BaseFeatureExtractor):
    """Extract patch tokens (conv feature map) and flatten them into a single vector per image.

    Notes
    -----
    - This extractor is intended for research / patch-based classical cores.
    - Output is one row per input image to preserve the FeatureExtractor contract.
    - Dimension can be large: (C * H * W) for the chosen node and image_size.
    """

    backbone: str = "resnet18"
    node: str = "layer4"
    pretrained: bool = False
    device: str = "cpu"
    batch_size: int = 16
    input_color: _InputColor = "rgb"
    image_size: int = 224

    def __post_init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = None
        self._torch = None

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        import torch
        import torchvision.transforms as T
        from torchvision.models.feature_extraction import create_feature_extractor

        model, weight_transform = _load_torchvision_backbone(
            str(self.backbone), pretrained=bool(self.pretrained)
        )
        feat_model = create_feature_extractor(model, return_nodes={str(self.node): "feat"})

        dev = _make_device(str(self.device))
        feat_model.to(dev)
        feat_model.eval()

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

        self._model = feat_model
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
        assert self._torch is not None

        torch = self._torch
        bs = max(1, int(self.batch_size))

        rows: list[np.ndarray] = []
        from pyimgano.utils.torch_infer import torch_inference

        with torch_inference(self._model):
            for i in range(0, len(items), bs):
                batch_items = items[i : i + bs]
                pil_imgs = [_as_pil_rgb(it, input_color=self.input_color) for it in batch_items]
                x = torch.stack([self._transform(im) for im in pil_imgs], dim=0).to(self._device)

                out = self._model(x)["feat"]
                out_t = torch.as_tensor(out)
                if out_t.ndim == 4:
                    n, c, h, w = out_t.shape
                    flat = out_t.reshape(n, c * h * w)
                else:
                    flat = torch.flatten(out_t, start_dim=1)
                rows.append(flat.detach().cpu().numpy().astype(np.float64, copy=False))

        return np.concatenate(rows, axis=0)


__all__ = ["TorchvisionPatchTokensExtractor"]

