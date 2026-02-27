from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor

from .torchvision_backbone import _as_pil_rgb, _load_torchvision_backbone, _make_device


_InputColor = Literal["rgb", "bgr"]


@dataclass
@register_feature_extractor(
    "torchvision_multilayer",
    tags=("embeddings", "torch", "torchvision", "deep-features"),
    metadata={
        "description": "Torchvision multi-layer embeddings (concat pooled intermediate feature maps)",
    },
)
class TorchvisionMultiLayerExtractor(BaseFeatureExtractor):
    """Extract embeddings from intermediate layers of a torchvision model.

    By default this is configured for ResNet backbones (`layer1..layer4`).
    """

    backbone: str = "resnet18"
    return_nodes: Optional[Sequence[str]] = None
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
        import torch.nn.functional as F
        import torchvision.transforms as T
        from torchvision.models.feature_extraction import create_feature_extractor

        model, weight_transform = _load_torchvision_backbone(
            str(self.backbone), pretrained=bool(self.pretrained)
        )

        nodes = list(self.return_nodes) if self.return_nodes is not None else ["layer1", "layer2", "layer3", "layer4"]
        feat_model = create_feature_extractor(model, return_nodes={n: n for n in nodes})

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
        self._F = F
        self._nodes = nodes

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
        F = self._F
        nodes = self._nodes

        bs = max(1, int(self.batch_size))
        rows: list[np.ndarray] = []

        from pyimgano.utils.torch_infer import torch_inference

        with torch_inference(self._model):
            for i in range(0, len(items), bs):
                batch_items = items[i : i + bs]
                pil_imgs = [_as_pil_rgb(it, input_color=self.input_color) for it in batch_items]
                x = torch.stack([self._transform(im) for im in pil_imgs], dim=0).to(self._device)

                out = self._model(x)
                # `out` is a dict node->tensor feature map
                pooled: list[Any] = []
                for n in nodes:
                    feat = out[n]
                    if feat.ndim == 4:
                        feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1))
                        feat = torch.flatten(feat, start_dim=1)
                    elif feat.ndim > 2:
                        feat = torch.flatten(feat, start_dim=1)
                    pooled.append(feat)

                emb = torch.cat(pooled, dim=1)
                rows.append(emb.detach().cpu().numpy().astype(np.float64, copy=False))

        return np.concatenate(rows, axis=0)
