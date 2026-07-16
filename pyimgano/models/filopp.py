"""Official-source FiLo inference adapter (ACM MM 2024).

The historical registry key is ``vision_filopp``, but the repository named by
the FiLo++ paper still contains only the released FiLo implementation.  This
module therefore exposes that verifiable FiLo graph instead of inventing the
unreleased FiLo++ MDCI/few-shot path.
"""

from __future__ import annotations

import importlib
import io
import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .deep_io import safe_torch_load
from .registry import register_model

OFFICIAL_REPOSITORY = "https://github.com/CASIA-IVA-Lab/FiLo"
OFFICIAL_COMMIT = "36ff29ca09ba8ba3af24d7654582aea856031400"
PAPER_CLIP_BACKBONE = "ViT-L-14-336"
PAPER_CLIP_PRETRAINED = "openai"
PAPER_IMAGE_SIZE = 518
PAPER_FEATURE_LAYERS = (6, 12, 18, 24)
PAPER_CONTEXT_TOKENS = 12
PAPER_PATCH_DIM = 1024
PAPER_TEXT_DIM = 768
PAPER_MMCI_BRANCHES = 3
PAPER_LINEAR_BRANCHES = 4
PAPER_BOX_THRESHOLD = 0.25
PAPER_TEXT_THRESHOLD = 0.25
PAPER_AREA_THRESHOLD = 0.7
PAPER_BACKGROUND_WEIGHT = 0.7
PAPER_GAUSSIAN_KERNEL = 3
PAPER_GAUSSIAN_SIGMA = 4.0

SUPPORTED_CATEGORIES = {
    "mvtec": (
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ),
    "visa": (
        "candle",
        "cashew",
        "chewinggum",
        "fryum",
        "pipe fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "capsules",
    ),
}

_LOCATION_BINS = (
    ("top left", 0, 172, 0, 172),
    ("top", 173, 344, 0, 172),
    ("top right", 345, 517, 0, 172),
    ("left", 0, 172, 173, 344),
    ("center", 173, 344, 173, 344),
    ("right", 345, 517, 173, 344),
    ("bottom left", 0, 172, 345, 517),
    ("bottom", 173, 344, 345, 517),
    ("bottom right", 345, 517, 345, 517),
)


def _dataset_key(dataset: str) -> str:
    key = "".join(character for character in str(dataset).lower() if character.isalnum())
    aliases = {"mvtec": "mvtec", "mvtecad": "mvtec", "visa": "visa"}
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("FiLo supports only MVTec AD and VisA checkpoints.") from exc


def _class_key(dataset: str, class_name: str | None) -> str:
    value = "" if class_name is None else " ".join(str(class_name).lower().split())
    value = value.replace("_", " ")
    if value not in SUPPORTED_CATEGORIES[dataset]:
        raise ValueError(
            f"FiLo {dataset} supports categories {SUPPORTED_CATEGORIES[dataset]}; got {value!r}."
        )
    return value


def _as_image_paths(value: Any) -> list[Path]:
    raw = [value] if isinstance(value, (str, Path)) else list(value)
    if not raw:
        raise ValueError("FiLo requires at least one image path.")
    paths: list[Path] = []
    for item in raw:
        if not isinstance(item, (str, Path)):
            raise TypeError(
                "The official FiLo runtime requires image paths for Grounding-DINO localization."
            )
        path = Path(item).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FiLo image not found: {path}")
        paths.append(path)
    return paths


@contextmanager
def _author_runtime(repository_path: Path) -> Iterator[None]:
    paths = [str(repository_path), str(repository_path / "models" / "GroundingDINO")]
    previous_cwd = Path.cwd()
    previous_backend = os.environ.get("MPLBACKEND")
    os.environ.setdefault("MPLBACKEND", "Agg")
    for path in reversed(paths):
        sys.path.insert(0, path)
    # ponytail: upstream uses top-level packages and cwd-relative files; isolate concurrent use.
    os.chdir(repository_path)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        os.chdir(previous_cwd)
        for path in paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        if previous_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous_backend


def _validate_author_model(model: Any) -> None:
    args = getattr(model, "args", None)
    expected = {
        "clip_model": PAPER_CLIP_BACKBONE,
        "clip_pretrained": PAPER_CLIP_PRETRAINED,
        "image_size": PAPER_IMAGE_SIZE,
        "features_list": PAPER_FEATURE_LAYERS,
        "n_ctx": PAPER_CONTEXT_TOKENS,
    }
    for name, value in expected.items():
        actual = getattr(args, name, None)
        matches = tuple(actual or ()) == value if isinstance(value, tuple) else actual == value
        if not matches:
            raise ValueError(
                f"FiLo author parameter mismatch: {name}={actual!r}, expected {value!r}."
            )

    normal_context = getattr(getattr(model, "normal_prompt_learner", None), "ctx", None)
    abnormal_context = getattr(getattr(model, "abnormal_prompt_learner", None), "ctx", None)
    if tuple(getattr(normal_context, "shape", ())) != (PAPER_CONTEXT_TOKENS, PAPER_TEXT_DIM):
        raise ValueError("FiLo normal prompt context must have shape (12, 768).")
    if tuple(getattr(abnormal_context, "shape", ())) != (PAPER_CONTEXT_TOKENS, PAPER_TEXT_DIM):
        raise ValueError("FiLo abnormal prompt context must have shape (12, 768).")

    linear_layers = tuple(getattr(getattr(model, "decoder_linear", None), "fc", ()))
    if len(linear_layers) != PAPER_LINEAR_BRANCHES or any(
        int(getattr(layer, "in_features", -1)) != PAPER_PATCH_DIM
        or int(getattr(layer, "out_features", -1)) != PAPER_TEXT_DIM
        for layer in linear_layers
    ):
        raise ValueError("FiLo QKV projection must contain four 1024-to-768 linear layers.")

    decoder = getattr(model, "decoder_cov", None)
    for name in ("fc_11", "fc_33", "fc_55", "fc_77", "fc_15", "fc_51"):
        layers = tuple(getattr(decoder, name, ()))
        if len(layers) != PAPER_MMCI_BRANCHES or any(
            int(getattr(layer, "in_channels", -1)) != PAPER_PATCH_DIM
            or int(getattr(layer, "out_channels", -1)) != PAPER_TEXT_DIM
            for layer in layers
        ):
            raise ValueError(
                f"FiLo MMCI branch {name} must contain three 1024-to-768 convolutions."
            )

    adapter_layers = [
        layer
        for layer in getattr(getattr(model, "adapter", None), "fc", ())
        if isinstance(layer, torch.nn.Linear)
    ]
    adapter_dims = [(int(layer.in_features), int(layer.out_features)) for layer in adapter_layers]
    if adapter_dims != [
        (PAPER_TEXT_DIM, PAPER_TEXT_DIM // 2),
        (PAPER_TEXT_DIM // 2, PAPER_TEXT_DIM),
    ]:
        raise ValueError("FiLo adapter must use the released 768-to-384-to-768 bottleneck.")


def _load_author_models(
    *,
    repository_path: Path,
    filo_checkpoint_path: Path,
    grounding_checkpoint_path: Path,
    dataset: str,
    device: torch.device,
    grounding_device: torch.device,
    allow_download: bool,
) -> tuple[Any, Any, Any]:
    source = repository_path / "test.py"
    filo_source = repository_path / "models" / "FiLo.py"
    grounding_config = (
        repository_path
        / "models"
        / "GroundingDINO"
        / "groundingdino"
        / "config"
        / "GroundingDINO_SwinT_OGC.py"
    )
    required = (
        source,
        filo_source,
        grounding_config,
        filo_checkpoint_path,
        grounding_checkpoint_path,
    )
    missing = next((path for path in required if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"Incomplete official FiLo runtime; missing: {missing}.")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The released FiLo inference path requires a CUDA device.")
    if grounding_device.type not in {"cpu", "cuda"} or (
        grounding_device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise RuntimeError(f"Unsupported FiLo Grounding-DINO device: {grounding_device}.")
    if not allow_download:
        raise ValueError(
            "FiLo loads the public OpenAI CLIP-L/14 weights on first use; pass "
            "allow_download=True after reviewing the upstream terms."
        )

    checkpoint = safe_torch_load(filo_checkpoint_path, map_location="cpu")
    state = checkpoint.get("filo") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping) or not state:
        raise ValueError("The official FiLo checkpoint must contain a non-empty 'filo' state dict.")

    args = SimpleNamespace(
        clip_model=PAPER_CLIP_BACKBONE,
        clip_pretrained=PAPER_CLIP_PRETRAINED,
        image_size=PAPER_IMAGE_SIZE,
        features_list=list(PAPER_FEATURE_LAYERS),
        n_ctx=PAPER_CONTEXT_TOKENS,
        device=str(device),
    )
    try:
        with _author_runtime(repository_path), redirect_stdout(io.StringIO()):
            module = importlib.import_module("test")
            if Path(module.__file__).resolve() != source.resolve():
                raise RuntimeError(
                    "A different top-level 'test' module is already imported; start a clean process."
                )
            loaded_filo_source = Path(importlib.import_module("models.FiLo").__file__).resolve()
            if loaded_filo_source != filo_source.resolve():
                raise RuntimeError(
                    "A different top-level 'models' package is already imported; start a clean process."
                )
            module.image_size = PAPER_IMAGE_SIZE
            object_names = list(
                module.mvtec_obj_list if dataset == "mvtec" else module.visa_obj_list
            )
            filo_model = module.FiLo(object_names, args, str(device)).to(device)
            filo_model.load_state_dict(dict(state), strict=False)
            grounding_model = module.load_model(
                str(grounding_config), str(grounding_checkpoint_path), str(grounding_device)
            )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The FiLo author dependencies are unavailable. Install its pinned requirements and "
            "GroundingDINO extension in an isolated environment."
        ) from exc

    _validate_author_model(filo_model)
    return filo_model.eval(), grounding_model.eval(), module


def _paper_boxes_and_position(
    boxes: torch.Tensor,
    phrases: Sequence[str],
    allowed_terms: Sequence[str],
) -> tuple[torch.Tensor, list[str]]:
    boxes = torch.as_tensor(boxes).detach().float().cpu().clone()
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(phrases) != len(boxes):
        raise ValueError("FiLo Grounding-DINO output must contain aligned Nx4 boxes and phrases.")

    valid: list[int] = []
    scale = boxes.new_tensor([PAPER_IMAGE_SIZE] * 4)
    for index, phrase in enumerate(phrases):
        if not any(term in phrase for term in allowed_terms):
            continue
        boxes[index] *= scale
        boxes[index, :2] -= boxes[index, 2:] / 2
        boxes[index, 2:] += boxes[index, :2]
        valid.append(index)

    if valid:

        def confidence(index: int) -> float:
            try:
                return float(phrases[index].rsplit("(", 1)[1].rstrip(")"))
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Invalid FiLo Grounding-DINO phrase: {phrases[index]!r}."
                ) from exc

        best = max(valid, key=confidence)
        center_x = float((boxes[best, 0] + boxes[best, 2]) / 2)
        center_y = float((boxes[best, 1] + boxes[best, 3]) / 2)
    else:
        center_x = center_y = 259.0

    positions = [
        name
        for name, x1, x2, y1, y2 in _LOCATION_BINS
        if x1 <= center_x <= x2 and y1 <= center_y <= y2
    ]
    return boxes, positions[:1]


def _paper_score_and_map(
    text_probabilities: Any,
    anomaly_maps: Sequence[Any],
    boxes: torch.Tensor,
    *,
    blur: Any,
) -> tuple[float, torch.Tensor]:
    maps = [torch.as_tensor(value).float() for value in anomaly_maps]
    if not maps or any(
        value.ndim != 4
        or value.shape[0] != 1
        or value.shape[1] != 2
        or tuple(value.shape[-2:]) != (PAPER_IMAGE_SIZE, PAPER_IMAGE_SIZE)
        for value in maps
    ):
        raise ValueError("FiLo must return 1x2x518x518 anomaly maps from its feature branches.")

    smoothed = [blur((value[:, 1] - value[:, 0] + 1) / 2) for value in maps]
    mean_map = torch.stack(smoothed).mean(dim=0).unsqueeze(1)
    text = torch.as_tensor(text_probabilities).flatten()
    if text.numel() != 2:
        raise ValueError("FiLo must return exactly normal and anomalous image probabilities.")
    score = float(((text[1] + mean_map.max()) / 2).item())

    foreground = torch.zeros_like(mean_map, dtype=torch.bool)
    for rectangle in torch.as_tensor(boxes).reshape(-1, 4):
        left, top, right, bottom = (int(value.item()) for value in rectangle)
        foreground[:, :, top:bottom, left:right] = True
    weighted = torch.where(foreground, mean_map, mean_map * PAPER_BACKGROUND_WEIGHT)
    return score, weighted[0, 0]


class AuthorFiLoBackend:
    """Execute the authors' CLIP/Grounding-DINO/FG-Des/HQ-Loc graph."""

    def __init__(
        self,
        *,
        repository_path: str | Path | None,
        filo_checkpoint_path: str | Path | None,
        grounding_checkpoint_path: str | Path | None,
        dataset: str,
        class_name: str | None,
        device: str,
        grounding_device: str | None = None,
        allow_download: bool = False,
    ) -> None:
        self.repository_path = None if repository_path is None else Path(repository_path).resolve()
        self.filo_checkpoint_path = (
            None if filo_checkpoint_path is None else Path(filo_checkpoint_path).resolve()
        )
        self.grounding_checkpoint_path = (
            None if grounding_checkpoint_path is None else Path(grounding_checkpoint_path).resolve()
        )
        self.dataset = _dataset_key(dataset)
        self.class_name = class_name
        self.device = torch.device(device)
        self.grounding_device = torch.device(grounding_device or device)
        self.allow_download = bool(allow_download)
        self.filo_model: Any = None
        self.grounding_model: Any = None
        self.author: Any = None
        self.blur: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if (
            self.repository_path is None
            or self.filo_checkpoint_path is None
            or self.grounding_checkpoint_path is None
        ):
            raise ValueError(
                "vision_filopp requires repository_path, filo_checkpoint_path, and "
                "grounding_checkpoint_path from the official FiLo release."
            )
        self.filo_model, self.grounding_model, self.author = _load_author_models(
            repository_path=self.repository_path,
            filo_checkpoint_path=self.filo_checkpoint_path,
            grounding_checkpoint_path=self.grounding_checkpoint_path,
            dataset=self.dataset,
            device=self.device,
            grounding_device=self.grounding_device,
            allow_download=self.allow_download,
        )
        from torchvision.transforms import GaussianBlur

        self.blur = GaussianBlur(PAPER_GAUSSIAN_KERNEL, PAPER_GAUSSIAN_SIGMA)
        self._loaded = True

    @torch.inference_mode()
    def score_paths(self, paths: Sequence[Path]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        self._ensure_loaded()
        class_name = _class_key(self.dataset, self.class_name)
        details = (
            self.author.mvtec_anomaly_detail_gpt[class_name]
            if self.dataset == "mvtec"
            else self.author.visa_anomaly_detail_gpt[class_name]
        )
        allowed_terms = list(details) + list(self.author.anomaly_status_general)
        prompt = " . ".join(list(self.author.anomaly_status_general) + list(details))
        scores: list[float] = []
        maps: list[NDArray[np.float32]] = []

        with _author_runtime(self.repository_path):
            for path in paths:
                with Image.open(path) as source:
                    image = self.filo_model.preprocess(source.convert("RGB"))
                items = {"img": image.unsqueeze(0).to(self.device), "cls_name": [class_name]}
                _source_image, dino_image = self.author.load_image(str(path))
                boxes, phrases = self.author.get_grounding_output(
                    self.grounding_model,
                    dino_image,
                    prompt,
                    PAPER_BOX_THRESHOLD,
                    PAPER_TEXT_THRESHOLD,
                    category=class_name,
                    device=str(self.grounding_device),
                    area_thr=PAPER_AREA_THRESHOLD,
                )
                boxes, positions = _paper_boxes_and_position(boxes, phrases, allowed_terms)
                text_probabilities, branch_maps = self.filo_model(
                    items, with_adapter=True, positions=positions
                )
                score, anomaly_map = _paper_score_and_map(
                    text_probabilities,
                    branch_maps,
                    boxes,
                    blur=self.blur,
                )
                scores.append(score)
                maps.append(anomaly_map.detach().cpu().numpy().astype(np.float32, copy=False))

        score_array = np.asarray(scores, dtype=np.float32)
        map_array = np.stack(maps).astype(np.float32, copy=False)
        if not np.isfinite(score_array).all() or not np.isfinite(map_array).all():
            raise ValueError("Author FiLo returned non-finite scores or maps.")
        return score_array, map_array


@register_model(
    "vision_filopp",
    tags=(
        "vision",
        "deep",
        "clip",
        "zero-shot",
        "pixel_map",
        "filo",
        "filopp",
        "acmmm2024",
        "external-backend",
    ),
    metadata={
        "description": "Official-source FiLo inference adapter (legacy vision_filopp key)",
        "paper": "FiLo: Zero-Shot Anomaly Detection by Fine-Grained Description and High-Quality Localization",
        "paper_url": "https://arxiv.org/abs/2404.13671",
        "official_repository": "https://github.com/CASIA-IVA-Lab/FiLo",
        "official_commit": "36ff29ca09ba8ba3af24d7654582aea856031400",
        "year": 2024,
        "conference": "ACM MM",
        "implementation_status": "official-source-checkpoint-inference-adapter",
        "paper_fidelity": "external-backend",
        "backend": "official-filo",
        "type": "cross-dataset-zero-shot-vlm",
        "supervision": "zero-shot",
        "training_protocol": "supervised on the opposite benchmark dataset",
        "requires_checkpoint": True,
        "supports_pixel_map": True,
        "weights_source": "official cross-dataset FiLo and Grounding-DINO checkpoints plus OpenAI CLIP",
        "upstream_license": "Apache-2.0",
        "compatibility_note": "FiLo++ source is not present at the official repository commit; this legacy key runs FiLo",
    },
)
class VisionFiLoPP(BaseDetector):
    """Official FiLo checkpoint inference under the historical registry name."""

    input_mode = "images"

    def __init__(
        self,
        *,
        repository_path: str | Path | None = None,
        filo_checkpoint_path: str | Path | None = None,
        grounding_checkpoint_path: str | Path | None = None,
        dataset: str = "mvtec",
        class_name: str | None = None,
        device: str = "cuda",
        grounding_device: str | None = None,
        allow_download: bool = False,
        backend: Any = None,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination=contamination)
        self._set_n_classes(None)
        self.repository_path = repository_path
        self.filo_checkpoint_path = filo_checkpoint_path
        self.grounding_checkpoint_path = grounding_checkpoint_path
        self.dataset = _dataset_key(dataset)
        self.class_name = class_name
        self.device = str(device)
        self.grounding_device = None if grounding_device is None else str(grounding_device)
        self.allow_download = bool(allow_download)
        self.backend = (
            backend
            if backend is not None
            else AuthorFiLoBackend(
                repository_path=repository_path,
                filo_checkpoint_path=filo_checkpoint_path,
                grounding_checkpoint_path=grounding_checkpoint_path,
                dataset=self.dataset,
                class_name=class_name,
                device=self.device,
                grounding_device=self.grounding_device,
                allow_download=self.allow_download,
            )
        )
        self.is_fitted_ = False

    def _score_paths(
        self, paths: Sequence[Path]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not callable(getattr(self.backend, "score_paths", None)):
            raise TypeError("FiLo backend must implement score_paths(paths).")
        result = self.backend.score_paths(paths)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("FiLo backend must return (scores, anomaly_maps).")
        scores = np.asarray(result[0], dtype=np.float32).reshape(-1)
        maps = np.asarray(result[1], dtype=np.float32)
        if scores.shape != (len(paths),) or maps.shape != (
            len(paths),
            PAPER_IMAGE_SIZE,
            PAPER_IMAGE_SIZE,
        ):
            raise ValueError("FiLo backend returned shapes inconsistent with the paper output.")
        if not np.isfinite(scores).all() or not np.isfinite(maps).all():
            raise ValueError("FiLo backend returned non-finite scores or maps.")
        return scores, maps

    def fit(
        self,
        x: object = MISSING,
        _y: Optional[NDArray[Any]] = None,
        **kwargs: object,
    ) -> "VisionFiLoPP":
        del _y
        paths = _as_image_paths(resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        _class_key(self.dataset, self.class_name)
        self.decision_scores_ = self._score_paths(paths)[0]
        self._process_decision_scores()
        self.is_fitted_ = True
        return self

    def decision_function(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() with normal calibration paths first.")
        values = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        return self._score_paths(_as_image_paths(values))[0]

    def predict(self, x: object = MISSING, **kwargs: object) -> NDArray[np.int64]:
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        return (self.decision_function(values) > float(self.threshold_)).astype(np.int64)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() with normal calibration paths first.")
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        return self._score_paths(_as_image_paths(values))[1]

    def get_anomaly_map(self, image: str | Path) -> NDArray[np.float32]:
        return self.predict_anomaly_map([image])[0]


__all__ = [
    "AuthorFiLoBackend",
    "OFFICIAL_COMMIT",
    "OFFICIAL_REPOSITORY",
    "PAPER_CLIP_BACKBONE",
    "PAPER_CONTEXT_TOKENS",
    "PAPER_FEATURE_LAYERS",
    "PAPER_IMAGE_SIZE",
    "SUPPORTED_CATEGORIES",
    "VisionFiLoPP",
]
