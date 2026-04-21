from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor
from pyimgano.utils.optional_deps import require

from .torchvision_backbone import _as_pil_rgb, _make_device, _require_initialized

_InputColor = Literal["rgb", "bgr"]


@dataclass
@register_feature_extractor(
    "openclip_embed",
    tags=("embeddings", "torch", "openclip", "deep-features"),
    metadata={
        "description": "OpenCLIP image embeddings (optional; requires open_clip_torch)",
    },
)
class OpenCLIPExtractor(BaseFeatureExtractor):
    """Extract global image embeddings from OpenCLIP.

    Notes
    -----
    - This extractor is **optional**: it requires `open_clip_torch` (`import open_clip`).
    - By default `pretrained=None` to avoid implicit weight downloads.
    """

    model_name: str = "ViT-B-32"
    pretrained: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 16
    input_color: _InputColor = "rgb"
    normalize: bool = True

    def __post_init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._device = None
        self._torch = None

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        open_clip = require("open_clip", extra="clip", purpose="OpenCLIPExtractor")
        torch = require("torch", extra="torch", purpose="OpenCLIPExtractor")

        dev = _make_device(str(self.device))

        result = open_clip.create_model_and_transforms(
            str(self.model_name),
            pretrained=self.pretrained,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("open_clip.create_model_and_transforms returned an unexpected value")

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

        model.to(dev)
        model.eval()

        self._model = model
        self._preprocess = preprocess
        self._device = dev
        self._torch = torch

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        items = list(inputs)
        if not items:
            return np.zeros((0, 1), dtype=np.float64)

        self._ensure_ready()
        model = _require_initialized(self._model, owner="OpenCLIPExtractor", attribute="_model")
        preprocess = _require_initialized(
            self._preprocess, owner="OpenCLIPExtractor", attribute="_preprocess"
        )
        device = _require_initialized(self._device, owner="OpenCLIPExtractor", attribute="_device")
        torch = _require_initialized(self._torch, owner="OpenCLIPExtractor", attribute="_torch")
        bs = max(1, int(self.batch_size))

        rows: list[np.ndarray] = []
        from pyimgano.utils.torch_infer import torch_inference

        with torch_inference(model):
            for i in range(0, len(items), bs):
                batch_items = items[i : i + bs]
                pil_imgs = [_as_pil_rgb(it, input_color=self.input_color) for it in batch_items]
                x = torch.stack([preprocess(im) for im in pil_imgs], dim=0).to(device)

                emb = model.encode_image(x)
                emb = torch.as_tensor(emb)
                if self.normalize:
                    emb = emb / torch.clamp(emb.norm(dim=1, keepdim=True), min=1e-12)

                rows.append(emb.detach().cpu().numpy().astype(np.float64, copy=False))

        return np.concatenate(rows, axis=0)
