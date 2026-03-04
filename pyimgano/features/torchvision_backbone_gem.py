from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.pooling import gem_pool2d
from pyimgano.features.registry import register_feature_extractor

from .torchvision_backbone import _as_pil_rgb, _load_torchvision_backbone, _make_device


_InputColor = Literal["rgb", "bgr"]


@dataclass
@register_feature_extractor(
    "torchvision_backbone_gem",
    tags=("embeddings", "torch", "torchvision", "gem", "deep-features"),
    metadata={
        "description": "Torchvision backbone embeddings via GeM pooling over a conv feature map",
    },
)
class TorchvisionBackboneGeMExtractor(BaseFeatureExtractor):
    """Extract GeM-pooled embeddings from a torchvision backbone.

    Notes
    -----
    - By default, `pretrained=False` to avoid implicit weight downloads.
    - This extractor pulls a convolutional feature map (default: `layer4` for
      ResNet backbones) and applies GeM pooling across HxW to produce (N,C).
    """

    backbone: str = "resnet18"
    node: str = "layer4"
    pretrained: bool = False
    device: str = "cpu"
    batch_size: int = 16
    input_color: _InputColor = "rgb"
    image_size: int = 224

    gem_p: float = 3.0
    gem_eps: float = 1e-6

    def __post_init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = None
        self._torch = None

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="TorchvisionBackboneGeMExtractor")
        T = require("torchvision.transforms", extra="torch", purpose="TorchvisionBackboneGeMExtractor")
        fe = require(
            "torchvision.models.feature_extraction",
            extra="torch",
            purpose="TorchvisionBackboneGeMExtractor",
        )
        create_feature_extractor = fe.create_feature_extractor

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

        # Validate pooling params early to fail fast when user passes bad values.
        _ = float(self.gem_p)
        _ = float(self.gem_eps)

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
                    emb = gem_pool2d(out_t, p=float(self.gem_p), eps=float(self.gem_eps))
                else:
                    emb = torch.flatten(out_t, start_dim=1)

                rows.append(emb.detach().cpu().numpy().astype(np.float64, copy=False))

        feats = np.concatenate(rows, axis=0)
        return np.asarray(feats, dtype=np.float64)


__all__ = ["TorchvisionBackboneGeMExtractor"]
