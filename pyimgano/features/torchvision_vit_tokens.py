from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor

from .torchvision_backbone import _as_pil_rgb, _make_device

_InputColor = Literal["rgb", "bgr"]
_Pool = Literal["cls", "mean"]


def _load_torchvision_vit(backbone: str, *, pretrained: bool):
    # Use the shared safe loader (no downloads by default).
    from pyimgano.utils.torchvision_safe import load_torchvision_model

    return load_torchvision_model(str(backbone), pretrained=bool(pretrained))


@dataclass
@register_feature_extractor(
    "torchvision_vit_tokens",
    tags=("embeddings", "torch", "torchvision", "vit", "deep-features"),
    metadata={
        "description": "Torchvision ViT token embeddings (CLS token or mean-pooled patch tokens)",
    },
)
class TorchvisionViTTokensExtractor(BaseFeatureExtractor):
    """Extract ViT token embeddings from a torchvision VisionTransformer.

    Notes
    -----
    - By default, `pretrained=False` to avoid implicit weight downloads.
    - `pool='cls'` returns the CLS token embedding (B, D).
    - `pool='mean'` returns mean-pooled patch tokens (B, D).
    """

    backbone: str = "vit_b_16"
    pretrained: bool = False
    pool: _Pool = "cls"
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

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="TorchvisionViTTokensExtractor")
        t = require(
            "torchvision.transforms", extra="torch", purpose="TorchvisionViTTokensExtractor"
        )

        model, weight_transform = _load_torchvision_vit(
            str(self.backbone), pretrained=bool(self.pretrained)
        )

        # Best-effort: drop the classification head if present. We do not rely
        # on it, but stripping avoids accidentally returning logits if someone
        # calls `model(x)` directly in the future.
        try:  # pragma: no cover - depends on torchvision internals
            nn = require("torch.nn", extra="torch", purpose="TorchvisionViTTokensExtractor")

            if hasattr(model, "heads"):
                model.heads = nn.Identity()  # type: ignore[attr-defined]
        except Exception:
            pass

        dev = _make_device(str(self.device))
        model.to(dev)
        model.eval()

        if weight_transform is not None:
            transform = weight_transform
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            transform = t.Compose(
                [
                    t.Resize((int(self.image_size), int(self.image_size))),
                    t.ToTensor(),
                    t.Normalize(mean=mean, std=std),
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
        assert self._torch is not None

        torch = self._torch
        bs = max(1, int(self.batch_size))
        pool = str(self.pool).strip().lower()
        if pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")

        rows: list[np.ndarray] = []
        from pyimgano.utils.torch_infer import torch_inference

        with torch_inference(self._model):
            for i in range(0, len(items), bs):
                batch_items = items[i : i + bs]
                pil_imgs = [_as_pil_rgb(it, input_color=self.input_color) for it in batch_items]
                x = torch.stack([self._transform(im) for im in pil_imgs], dim=0).to(self._device)

                # Token extraction (cls + patch tokens).
                if not hasattr(self._model, "_process_input"):
                    raise TypeError(
                        "Backbone does not support ViT token extraction (missing _process_input)."
                    )
                if not hasattr(self._model, "encoder") or not hasattr(self._model, "class_token"):
                    raise TypeError("Backbone does not look like a torchvision VisionTransformer.")

                xt = self._model._process_input(x)  # type: ignore[attr-defined]
                n = int(xt.shape[0])
                cls = self._model.class_token.expand(n, -1, -1)  # type: ignore[attr-defined]
                tokens = torch.cat([cls, xt], dim=1)
                tokens = self._model.encoder(tokens)  # type: ignore[attr-defined]

                if pool == "cls":
                    emb = tokens[:, 0, :]
                else:
                    emb = tokens[:, 1:, :].mean(dim=1)

                rows.append(emb.detach().cpu().numpy().astype(np.float64, copy=False))

        return np.concatenate(rows, axis=0)
