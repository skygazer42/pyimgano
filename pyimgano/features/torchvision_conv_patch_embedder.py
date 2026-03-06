from __future__ import annotations

"""Torchvision conv-feature patch embedder (offline-safe by default).

This is a small utility used by patch-level "classical core" detectors:
  image -> conv feature map -> flatten spatial grid into patch embeddings

The returned patch embeddings are suitable for memory-bank kNN scoring (e.g.
PatchCore-lite-map / SoftPatch-style pipelines).

Key constraints
---------------
- No implicit weight downloads by default (`pretrained=False`).
- Lazy initialization: torch/torchvision are imported only when needed.
"""

from dataclasses import dataclass
from typing import Any, Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .torchvision_backbone import _as_pil_rgb, _load_torchvision_backbone, _make_device

_InputColor = Literal["rgb", "bgr"]


@dataclass
class TorchvisionConvPatchEmbedder:
    """Extract patch embeddings from a torchvision conv backbone node.

    Parameters
    ----------
    backbone:
        A torchvision backbone name supported by `pyimgano.utils.torchvision_safe`
        (e.g. 'resnet18', 'resnet50', ...).
    node:
        Feature node name for `torchvision.models.feature_extraction.create_feature_extractor`.
        Typical ResNet nodes: 'layer1'|'layer2'|'layer3'|'layer4'.
    pretrained:
        When True, uses torchvision weights (may download). Default is False.
    normalize:
        If True, L2-normalize each patch embedding row.
    """

    backbone: str = "resnet18"
    node: str = "layer3"
    pretrained: bool = False

    device: str = "cpu"
    input_color: _InputColor = "rgb"
    image_size: int = 224

    normalize: bool = True
    eps: float = 1e-12

    _model: Any = None
    _transform: Any = None
    _device: Any = None
    _torch: Any = None

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="TorchvisionConvPatchEmbedder")
        T = require("torchvision.transforms", extra="torch", purpose="TorchvisionConvPatchEmbedder")
        fe = require(
            "torchvision.models.feature_extraction",
            extra="torch",
            purpose="TorchvisionConvPatchEmbedder",
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

        self._model = feat_model
        self._transform = transform
        self._device = dev
        self._torch = torch

    def _l2_normalize_rows(self, x: NDArray) -> NDArray:
        if not self.normalize:
            return np.asarray(x, dtype=np.float32)
        x32 = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x32, axis=1, keepdims=True)
        norms = np.maximum(norms, float(self.eps))
        return np.asarray(x32 / norms, dtype=np.float32)

    def embed(
        self, image: Union[str, np.ndarray]
    ) -> Tuple[NDArray[np.float32], Tuple[int, int], Tuple[int, int]]:
        """Return (patch_embeddings, grid_shape, original_size).

        - patch_embeddings: (H*W, C) float32
        - grid_shape: (H, W) of the conv feature map
        - original_size: (orig_h, orig_w) of the input image
        """

        self._ensure_ready()
        assert self._model is not None
        assert self._transform is not None
        assert self._device is not None
        assert self._torch is not None

        pil_img = _as_pil_rgb(image, input_color=str(self.input_color))
        orig_w, orig_h = pil_img.size

        torch = self._torch
        from pyimgano.utils.torch_infer import torch_inference

        with torch_inference(self._model):
            x = self._transform(pil_img).unsqueeze(0).to(self._device)
            out = self._model(x)["feat"]
            ft = torch.as_tensor(out)

        if ft.ndim != 4:
            raise ValueError(
                f"Expected conv feature map with shape (B,C,H,W), got {tuple(ft.shape)}"
            )

        # (1, C, H, W) -> (H*W, C)
        _b, c, h, w = ft.shape
        patches = ft[0].permute(1, 2, 0).reshape(int(h) * int(w), int(c))
        patch_embeddings = patches.detach().cpu().numpy().astype(np.float32, copy=False)
        patch_embeddings = self._l2_normalize_rows(patch_embeddings)

        return (
            np.asarray(patch_embeddings, dtype=np.float32),
            (int(h), int(w)),
            (int(orig_h), int(orig_w)),
        )


__all__ = ["TorchvisionConvPatchEmbedder"]
