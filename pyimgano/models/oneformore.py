"""Official-source checkpoint inference adapter for One-for-More (CVPR 2025).

The paper's CDAD training is a continual, dataset-level protocol.  This module
does not replace it with a small local proxy: it loads the authors' model from
their repository and a task checkpoint produced by their training scripts.
"""

from __future__ import annotations

import importlib
import io
import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import gaussian_filter

from pyimgano.utils.random_state import isolated_random_state

from ._batch_size import call_with_temporary_attr, validate_batch_size
from ._image_batch import _coerce_single_rgb_image
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .deep_io import safe_torch_load
from .registry import register_model

OFFICIAL_REPOSITORY = "https://github.com/FuNz-0/One-for-More"
OFFICIAL_COMMIT = "f4eb78841dbfa5612e008570b690072b19a3d9b3"
PAPER_BASE_MODEL = "Stable Diffusion v1.5"
PAPER_IMAGE_SIZE = 256
PAPER_TEXT_CONDITION = ""
PAPER_LATENT_CHANNELS = 4
PAPER_MODEL_CHANNELS = 320
PAPER_CHANNEL_MULT = (1, 2, 4, 4)
PAPER_ATTENTION_RESOLUTIONS = (4, 2, 1)
PAPER_NUM_RES_BLOCKS = 2
PAPER_NUM_HEADS = 8
PAPER_AMN_DEPTH = 8
PAPER_AMN_NEIGHBOR_SIZE = (7, 7)
PAPER_DDIM_STEPS = 10
PAPER_DDIM_ETA = 0.0
PAPER_GUIDANCE_SCALE = 9.0
PAPER_GAUSSIAN_SIGMA = 5.0
PAPER_FEATURE_LAYERS = {"mvtec": (1, 2, 3), "visa": (0, 2, 4)}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _dataset_key(dataset: str) -> str:
    key = "".join(character for character in str(dataset).lower() if character.isalnum())
    aliases = {"mvtecad": "mvtec", "mvtec": "mvtec", "visa": "visa"}
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("One-for-More supports only MVTec AD and VisA checkpoints.") from exc


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 4:
            items = [value[index] for index in range(len(value))]
        elif value.ndim in (2, 3):
            items = [value]
        else:
            raise ValueError(f"Expected image input with 2-4 dimensions, got {value.shape}.")
    elif isinstance(value, (str, Path)):
        items = [value]
    else:
        items = list(value)
    if not items:
        raise ValueError("One-for-More requires at least one image.")
    return items


def _paper_image_tensor(item: Any) -> torch.Tensor:
    """Apply the authors' 256px bilinear resize and ImageNet normalization."""

    array = np.asarray(_coerce_single_rgb_image(item))
    if not np.isfinite(array).all():
        raise ValueError("One-for-More inputs must contain only finite values.")
    if np.issubdtype(array.dtype, np.floating) and array.size and float(array.max()) <= 1.0:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(array, mode="RGB").resize(
        (PAPER_IMAGE_SIZE, PAPER_IMAGE_SIZE), resample=resampling.BILINEAR
    )
    tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(_IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _paper_anomaly_maps(
    input_features: Sequence[torch.Tensor],
    output_features: Sequence[torch.Tensor],
    *,
    output_size: int = PAPER_IMAGE_SIZE,
) -> NDArray[np.float32]:
    """Released ResNet feature-distance sum followed by sigma-5 smoothing."""

    if not input_features or len(input_features) != len(output_features):
        raise ValueError("One-for-More feature pyramids must be non-empty and aligned.")
    batch = int(input_features[0].shape[0])
    maps = np.zeros((batch, output_size, output_size), dtype=np.float64)
    for source, reconstruction in zip(input_features, output_features):
        if source.shape != reconstruction.shape or int(source.shape[0]) != batch:
            raise ValueError("One-for-More feature tensors must have matching batch shapes.")
        layer_map = torch.norm(source - reconstruction, dim=1, keepdim=True)
        layer_map = F.interpolate(
            layer_map,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=True,
        )[:, 0]
        maps += layer_map.detach().cpu().numpy()
    for index in range(batch):
        maps[index] = gaussian_filter(maps[index], sigma=PAPER_GAUSSIAN_SIGMA)
    return maps.astype(np.float32, copy=False)


@contextmanager
def _author_import_path(repository_path: Path, *, allow_download: bool) -> Iterator[None]:
    path = str(repository_path)
    previous_offline = {
        name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, path)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        if sys.path and sys.path[0] == path:
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        for name, value in previous_offline.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_author_model(
    *,
    repository_path: Path,
    checkpoint_path: Path,
    dataset: str,
    device: torch.device,
    allow_download: bool,
) -> Any:
    config_path = repository_path / "models" / f"cdad_{dataset}.yaml"
    model_source = repository_path / "cdm" / "model.py"
    if not config_path.is_file() or not model_source.is_file():
        raise FileNotFoundError(
            f"Not an official One-for-More source tree: {repository_path}. "
            f"Clone {OFFICIAL_REPOSITORY}."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"One-for-More checkpoint not found: {checkpoint_path}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The released One-for-More runtime requires a CUDA device.")

    payload = safe_torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("One-for-More checkpoint must contain a state dictionary.")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("One-for-More checkpoint has no usable state dictionary.")

    try:
        with (
            _author_import_path(repository_path, allow_download=allow_download),
            redirect_stdout(io.StringIO()),
        ):
            importlib.import_module("share")
            model_module = importlib.import_module("cdm.model")
            loaded_source = Path(model_module.__file__).resolve()
            if not loaded_source.is_relative_to(repository_path):
                raise RuntimeError(
                    "A different top-level 'cdm' package is already imported; start a clean "
                    "process before loading the One-for-More author repository."
                )
            model = model_module.create_model(str(config_path)).cpu()
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The One-for-More author runtime dependencies are unavailable. Install the exact "
            "versions from its install.sh (notably pytorch-lightning==1.9.0)."
        ) from exc

    model.load_state_dict(dict(state), strict=True)
    return model.to(device).eval()


class AuthorOneForMoreBackend:
    """Execute the authors' CDAD checkpoint and released scoring path."""

    def __init__(
        self,
        *,
        repository_path: str | Path | None,
        checkpoint_path: str | Path | None,
        dataset: str,
        device: str,
        batch_size: int,
        allow_download: bool = False,
        model: Any = None,
    ) -> None:
        self.repository_path = None if repository_path is None else Path(repository_path).resolve()
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path).resolve()
        self.dataset = _dataset_key(dataset)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.allow_download = bool(allow_download)
        self.model = model
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.model is None:
            if self.repository_path is None or self.checkpoint_path is None:
                raise ValueError(
                    "vision_oneformore requires repository_path and an author-trained "
                    "checkpoint_path; the authors did not release task checkpoints."
                )
            self.model = _load_author_model(
                repository_path=self.repository_path,
                checkpoint_path=self.checkpoint_path,
                dataset=self.dataset,
                device=self.device,
                allow_download=self.allow_download,
            )
        else:
            self.model = self.model.to(self.device).eval()

        layers = tuple(int(layer) for layer in getattr(self.model, "layers_", ()))
        if layers != PAPER_FEATURE_LAYERS[self.dataset]:
            raise ValueError(
                f"One-for-More {self.dataset} requires ResNet layers "
                f"{PAPER_FEATURE_LAYERS[self.dataset]}, got {layers}."
            )
        if str(getattr(self.model, "distance", "")) != "eucl":
            raise ValueError("The released One-for-More checkpoint requires Euclidean scoring.")
        if not callable(getattr(self.model, "log_images_test", None)) or not callable(
            getattr(self.model, "pretrained_resnet50", None)
        ):
            raise TypeError("Expected an author CDAD model with log_images_test() and ResNet50.")
        self._loaded = True

    @torch.inference_mode()
    def score_items(
        self, items: Sequence[Any], *, seed: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        self._ensure_loaded()
        scores: list[NDArray[np.float32]] = []
        maps: list[NDArray[np.float32]] = []
        with isolated_random_state(int(seed)):
            for start in range(0, len(items), self.batch_size):
                batch_items = items[start : start + self.batch_size]
                images = torch.stack([_paper_image_tensor(item) for item in batch_items]).to(
                    self.device
                )
                batch = {
                    "jpg": images,
                    "hint": images,
                    "txt": [PAPER_TEXT_CONDITION] * len(batch_items),
                }
                output = self.model.log_images_test(
                    batch,
                    ddim_steps=PAPER_DDIM_STEPS,
                    ddim_eta=PAPER_DDIM_ETA,
                    unconditional_guidance_scale=PAPER_GUIDANCE_SCALE,
                )
                if not isinstance(output, Mapping) or not {
                    "reconstruction",
                    "samples",
                }.issubset(output):
                    raise TypeError("Author CDAD log_images_test() returned an invalid payload.")
                feature_extractor = self.model.pretrained_resnet50
                feature_extractor.eval()
                input_features = feature_extractor(output["reconstruction"])
                output_features = feature_extractor(output["samples"])
                layers = PAPER_FEATURE_LAYERS[self.dataset]
                batch_maps = _paper_anomaly_maps(
                    [input_features[index] for index in layers],
                    [output_features[index] for index in layers],
                )
                maps.append(batch_maps)
                scores.append(batch_maps.reshape(len(batch_items), -1).max(axis=1))
        return (
            np.concatenate(scores).astype(np.float32, copy=False),
            np.concatenate(maps).astype(np.float32, copy=False),
        )


@register_model(
    "vision_oneformore",
    tags=(
        "vision",
        "deep",
        "oneformore",
        "continual",
        "diffusion",
        "reconstruction",
        "cvpr2025",
        "external-backend",
    ),
    metadata={
        "description": "Official-source One-for-More CDAD checkpoint inference adapter",
        "paper": "One-for-More: Continual Diffusion Model for Anomaly Detection",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2025/html/Li_One-for-More_Continual_Diffusion_Model_for_Anomaly_Detection_CVPR_2025_paper.html",
        "official_repository": "https://github.com/FuNz-0/One-for-More",
        "year": 2025,
        "conference": "CVPR",
        "implementation_status": "official-source-checkpoint-inference-adapter",
        "paper_fidelity": "external-backend",
        "backend": "official-one-for-more",
        "type": "continual-diffusion",
        "supervision": "one-class",
        "requires_checkpoint": True,
        "supports_pixel_map": True,
        "weights_source": "author-trained One-for-More task checkpoint (not released upstream)",
    },
)
class VisionOneForMore(BaseDetector):
    """One-for-More author-checkpoint inference; ``fit`` only calibrates scores."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        repository_path: str | Path | None = None,
        dataset: str = "mvtec",
        backend: Any = None,
        batch_size: int = 12,
        device: str = "cuda",
        random_state: int = 1,
        allow_download: bool = False,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination=contamination)
        self._set_n_classes(None)
        self.checkpoint_path = checkpoint_path
        self.repository_path = repository_path
        self.dataset = _dataset_key(dataset)
        batch_size_int = validate_batch_size(batch_size)
        assert batch_size_int is not None
        self.batch_size = batch_size_int
        self.device = str(device)
        self.random_state = int(random_state)
        self.allow_download = bool(allow_download)
        self.backend = (
            backend
            if backend is not None
            else AuthorOneForMoreBackend(
                repository_path=repository_path,
                checkpoint_path=checkpoint_path,
                dataset=self.dataset,
                device=self.device,
                batch_size=self.batch_size,
                allow_download=self.allow_download,
            )
        )

    def _score_items(self, items: Sequence[Any]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not callable(getattr(self.backend, "score_items", None)):
            raise TypeError("One-for-More backend must implement score_items(items, seed=...).")
        if hasattr(self.backend, "batch_size"):
            result = call_with_temporary_attr(
                self.backend,
                "batch_size",
                self.batch_size,
                lambda: self.backend.score_items(items, seed=self.random_state),
            )
        else:
            result = self.backend.score_items(items, seed=self.random_state)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("One-for-More backend must return (scores, anomaly_maps).")
        scores = np.asarray(result[0], dtype=np.float32).reshape(-1)
        maps = np.asarray(result[1], dtype=np.float32)
        if scores.shape != (len(items),) or maps.ndim != 3 or maps.shape[0] != len(items):
            raise ValueError("One-for-More backend returned shapes inconsistent with the inputs.")
        if not np.isfinite(scores).all() or not np.isfinite(maps).all():
            raise ValueError("One-for-More backend returned non-finite scores or maps.")
        return scores, maps

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray[Any]] = None,
        **kwargs: object,
    ) -> "VisionOneForMore":
        del y
        values = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        self.decision_scores_ = self._score_items(_as_items(values))[0]
        self._process_decision_scores()
        self.is_fitted_ = True
        return self

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        if return_confidence:
            raise NotImplementedError(
                f"return_confidence is not implemented for {self.__class__.__name__}"
            )
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        return self._score_items(_as_items(values))[0]

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        return self._score_items(_as_items(values))[1]

    def get_anomaly_map(self, image: Any) -> NDArray[np.float32]:
        return self.predict_anomaly_map([image])[0]

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray[np.float32]:
        values = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        batch_size_int = validate_batch_size(batch_size)
        if batch_size_int is None:
            return self.predict(values)
        return call_with_temporary_attr(
            self,
            "batch_size",
            batch_size_int,
            lambda: self.predict(values),
        )


__all__ = [
    "AuthorOneForMoreBackend",
    "OFFICIAL_COMMIT",
    "PAPER_AMN_DEPTH",
    "PAPER_AMN_NEIGHBOR_SIZE",
    "PAPER_CHANNEL_MULT",
    "PAPER_DDIM_STEPS",
    "PAPER_FEATURE_LAYERS",
    "PAPER_IMAGE_SIZE",
    "PAPER_MODEL_CHANNELS",
    "VisionOneForMore",
]
