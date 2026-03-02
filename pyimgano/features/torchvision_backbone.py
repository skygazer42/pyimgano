from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor


_InputColor = Literal["rgb", "bgr"]
_Pool = Literal["avg", "max", "gem", "cls"]


def _as_pil_rgb(item: Any, *, input_color: _InputColor):  # noqa: ANN001, ANN201
    # Import lazily: keep registry discovery light.
    from PIL import Image

    if isinstance(item, (str, Path)):
        img = Image.open(str(item)).convert("RGB")
        return img

    if isinstance(item, Image.Image):
        # PIL images are already RGB-ordered conceptually; treat them as RGB inputs.
        return item.convert("RGB")

    # Common industrial pipeline: images already decoded as torch tensors.
    try:  # pragma: no cover - depends on torch being installed
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(item, torch.Tensor):
        t = item.detach().cpu()
        if t.ndim == 2:
            return _as_pil_rgb(t.numpy(), input_color=input_color)
        if t.ndim == 3:
            # Accept both CHW and HWC (best-effort).
            if int(t.shape[0]) in (1, 3):  # CHW
                arr = t.numpy()
                arr_hwc = np.transpose(arr, (1, 2, 0))
                return _as_pil_rgb(arr_hwc, input_color=input_color)
            if int(t.shape[2]) in (1, 3):  # HWC
                return _as_pil_rgb(t.numpy(), input_color=input_color)
            raise ValueError(f"Unsupported torch image shape: {tuple(t.shape)} (expected CHW or HWC)")

        raise ValueError(f"Unsupported torch image ndim={int(t.ndim)} (expected 2 or 3)")

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
        "TorchvisionBackboneExtractor expects inputs of type str|Path|np.ndarray|PIL.Image|torch.Tensor, "
        f"got {type(item)}"
    )


def _make_device(device: str):  # noqa: ANN001, ANN201
    import torch

    dev = str(device).strip().lower()
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(dev)


def _load_torchvision_backbone(backbone: str, *, pretrained: bool):
    from pyimgano.utils.torchvision_safe import load_torchvision_backbone

    return load_torchvision_backbone(str(backbone), pretrained=bool(pretrained))


@contextmanager
def _maybe_autocast(torch, *, device, enabled: bool):  # noqa: ANN001 - torch is dynamic
    if not enabled:
        yield
        return

    # Best-effort: only enable autocast on CUDA by default. CPU autocast can
    # be surprising (bf16 availability, different numerics) and is rarely the
    # industrial default for inference.
    dev_type = str(getattr(device, "type", "cpu"))
    if dev_type != "cuda":
        yield
        return

    try:  # pragma: no cover - depends on torch version/backend
        from torch.cuda.amp import autocast

        with autocast(dtype=torch.float16):
            yield
    except Exception:  # noqa: BLE001 - best-effort
        try:  # pragma: no cover
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        except Exception:
            yield


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
    pool: _Pool = "avg"
    pool_node: str = "layer4"
    gem_p: float = 3.0
    gem_eps: float = 1e-6

    device: str = "cpu"
    batch_size: int = 16
    input_color: _InputColor = "rgb"

    # When not using weights-provided transforms, fall back to this size+norm.
    image_size: int = 224
    channels_last: bool = False
    amp: bool = False
    compile: bool = False

    # Optional disk cache for path inputs (embedding rows keyed by path+mtime+extractor config).
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        # Lazy init: don't load weights / models during registry import.
        self._model = None
        self._transform = None
        self._device = None
        self._cache = None
        self.last_cache_stats_ = {"hits": 0, "misses": 0, "enabled": False}

        pool = str(self.pool).strip().lower()
        if pool not in ("avg", "max", "gem", "cls"):
            raise ValueError("pool must be one of: avg|max|gem|cls")
        self.pool = pool  # type: ignore[assignment]

        if self.cache_dir is not None:
            from pyimgano.cache.embeddings import EmbeddingCache, fingerprint_payload

            payload = {
                "type": "torchvision_backbone",
                "backbone": str(self.backbone),
                "pretrained": bool(self.pretrained),
                "pool": str(self.pool),
                "pool_node": str(self.pool_node),
                "gem_p": float(self.gem_p),
                "gem_eps": float(self.gem_eps),
                "image_size": int(self.image_size),
                "input_color": str(self.input_color),
            }
            fp = fingerprint_payload(payload)
            self._cache = EmbeddingCache(cache_dir=Path(str(self.cache_dir)), extractor_fingerprint=fp)

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        import torch
        import torch.nn.functional as F
        import torchvision.transforms as T

        pool = str(self.pool).strip().lower()
        model, weight_transform = _load_torchvision_backbone(
            str(self.backbone), pretrained=bool(self.pretrained)
        )
        dev = _make_device(str(self.device))

        if pool == "gem":
            from torchvision.models.feature_extraction import create_feature_extractor

            node = str(self.pool_node)
            model = create_feature_extractor(model, return_nodes={node: "feat"})

        model.to(dev)
        model.eval()

        if bool(self.channels_last):
            try:  # pragma: no cover - best-effort
                model.to(memory_format=torch.channels_last)
            except Exception:
                pass

        if bool(self.compile) and str(getattr(dev, "type", "cpu")) == "cuda" and hasattr(torch, "compile"):
            try:  # pragma: no cover - compilation is backend dependent
                model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
            except Exception:
                # Best-effort: keep uncompiled model.
                pass

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
        self._F = F

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        items = list(inputs)
        if not items:
            return np.zeros((0, 1), dtype=np.float64)

        if self._cache is not None and all(isinstance(it, (str, Path)) for it in items):
            paths = [str(p) for p in items]
            cached_rows: dict[str, np.ndarray] = {}
            missing: list[str] = []

            for p in paths:
                row = self._cache.load(p)
                if row is None:
                    missing.append(p)
                else:
                    cached_rows[p] = np.asarray(row, dtype=np.float64).reshape(-1)

            if missing:
                feats_missing = np.asarray(self._extract_uncached(missing), dtype=np.float64)
                if feats_missing.ndim == 1:
                    feats_missing = feats_missing.reshape(-1, 1)
                if feats_missing.shape[0] != len(missing):
                    raise ValueError(
                        "torchvision_backbone extractor must return one row per path. "
                        f"Got shape {feats_missing.shape} for {len(missing)} inputs."
                    )
                for i, p in enumerate(missing):
                    row = np.asarray(feats_missing[i], dtype=np.float32).reshape(-1)
                    self._cache.save(p, row)
                    cached_rows[p] = row.astype(np.float64, copy=False)

            self.last_cache_stats_ = {
                "enabled": True,
                "hits": int(len(paths) - len(missing)),
                "misses": int(len(missing)),
            }
            rows = [cached_rows[p] for p in paths]
            return np.stack(rows, axis=0).astype(np.float64, copy=False)

        self.last_cache_stats_ = {"enabled": False, "hits": 0, "misses": int(len(items))}
        return self._extract_uncached(items)

    def _extract_uncached(self, items: list[Any]) -> np.ndarray:
        self._ensure_ready()
        assert self._model is not None
        assert self._transform is not None
        assert self._device is not None

        torch = self._torch
        F = self._F
        bs = max(1, int(self.batch_size))
        pool = str(self.pool).strip().lower()

        rows: list[np.ndarray] = []
        from pyimgano.utils.torch_infer import torch_inference

        with torch_inference(self._model):
            for i in range(0, len(items), bs):
                batch_items = items[i : i + bs]
                pil_imgs = [_as_pil_rgb(it, input_color=self.input_color) for it in batch_items]
                x = torch.stack([self._transform(im) for im in pil_imgs], dim=0)
                x = x.to(self._device)

                if bool(self.channels_last):
                    try:  # pragma: no cover - best-effort
                        x = x.contiguous(memory_format=torch.channels_last)
                    except Exception:
                        pass

                with _maybe_autocast(torch, device=self._device, enabled=bool(self.amp)):
                    if pool == "gem":
                        out = self._model(x)["feat"]
                        out_t = torch.as_tensor(out)
                        if out_t.ndim == 4:
                            from pyimgano.features.pooling import gem_pool2d

                            emb = gem_pool2d(out_t, p=float(self.gem_p), eps=float(self.gem_eps))
                        else:
                            emb = torch.flatten(out_t, start_dim=1)
                    elif pool == "cls":
                        emb = self._vit_cls_embedding(x)
                    else:
                        out = self._model(x)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                        out_t = torch.as_tensor(out)
                        if out_t.ndim == 4:
                            if pool == "max":
                                out_t = F.adaptive_max_pool2d(out_t, output_size=(1, 1))
                            else:
                                out_t = F.adaptive_avg_pool2d(out_t, output_size=(1, 1))
                            emb = torch.flatten(out_t, start_dim=1)
                        elif out_t.ndim > 2:
                            emb = torch.flatten(out_t, start_dim=1)
                        else:
                            emb = out_t

                out_np = emb.detach().cpu().numpy().astype(np.float64, copy=False)
                rows.append(out_np)

        feats = np.concatenate(rows, axis=0)
        return np.asarray(feats, dtype=np.float64)

    def _vit_cls_embedding(self, x):  # noqa: ANN001, ANN201 - torch tensor
        torch = self._torch
        if torch is None:  # pragma: no cover - should be initialized in _ensure_ready
            raise RuntimeError("Internal error: torch not initialized")

        model = self._model
        if model is None:
            raise RuntimeError("Internal error: model not initialized")

        if not hasattr(model, "_process_input"):
            raise TypeError("Backbone does not support ViT token extraction (missing _process_input).")
        if not hasattr(model, "encoder") or not hasattr(model, "class_token"):
            raise TypeError("pool='cls' requires a torchvision VisionTransformer backbone.")

        xt = model._process_input(x)  # type: ignore[attr-defined]
        n = int(xt.shape[0])
        cls = model.class_token.expand(n, -1, -1)  # type: ignore[attr-defined]
        tokens = torch.cat([cls, xt], dim=1)
        tokens = model.encoder(tokens)  # type: ignore[attr-defined]
        return tokens[:, 0, :]
