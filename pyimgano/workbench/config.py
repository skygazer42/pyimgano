from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pyimgano.workbench.adaptation import AdaptationConfig, MapPostprocessConfig, TilingConfig


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a dict/object, got {type(value).__name__}")
    return value


def _optional_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"{name} must be int or null, got {value!r}") from exc


def _optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"{name} must be float or null, got {value!r}") from exc


def _parse_resize(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return (int(default[0]), int(default[1]))

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"resize must be a list/tuple of length 2, got {value!r}")
    try:
        h = int(value[0])
        w = int(value[1])
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"resize must contain ints, got {value!r}") from exc
    if h <= 0 or w <= 0:
        raise ValueError(f"resize must be positive, got {(h, w)}")
    return (h, w)


def _parse_percentile_range(
    value: Any,
    *,
    default: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
    if value is None:
        return (float(default[0]), float(default[1]))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"percentile_range must be a list/tuple of length 2, got {value!r}"
        )
    try:
        low = float(value[0])
        high = float(value[1])
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"percentile_range must contain floats, got {value!r}") from exc
    return (low, high)


def _parse_checkpoint_name(value: Any, *, default: str = "model.pt") -> str:
    if value is None:
        return str(default)
    name = str(value).strip()
    if not name:
        raise ValueError("training.checkpoint_name must be a non-empty filename")
    if name in (".", ".."):
        raise ValueError("training.checkpoint_name must be a filename, got '.'/'..'")
    if "/" in name or "\\" in name:
        raise ValueError("training.checkpoint_name must be a filename, not a path")
    p = Path(name)
    if p.is_absolute() or p.name != name:
        raise ValueError("training.checkpoint_name must be a filename, not a path")
    return name


def _parse_roi_xyxy_norm(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("defects.roi_xyxy_norm must be a list/tuple of length 4 or null")

    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"defects.roi_xyxy_norm must contain floats, got {value!r}") from exc

    def _clamp01(v: float) -> float:
        return float(min(max(v, 0.0), 1.0))

    x1c, y1c, x2c, y2c = (_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2))
    return (min(x1c, x2c), min(y1c, y2c), max(x1c, x2c), max(y1c, y2c))


@dataclass(frozen=True)
class SplitPolicyConfig:
    """Controls auto-splitting for datasets that support it (e.g. manifest JSONL)."""

    mode: str = "benchmark"
    scope: str = "category"
    seed: int | None = None
    test_normal_fraction: float = 0.2


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    manifest_path: str | None = None
    category: str = "all"
    resize: tuple[int, int] = (256, 256)
    input_mode: str = "paths"
    limit_train: int | None = None
    limit_test: int | None = None
    split_policy: SplitPolicyConfig = field(default_factory=SplitPolicyConfig)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str = "cpu"
    preset: str | None = None
    pretrained: bool = True
    contamination: float = 0.1
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str | None = None
    save_run: bool = True
    per_image_jsonl: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = False
    epochs: int | None = None
    lr: float | None = None
    checkpoint_name: str = "model.pt"


@dataclass(frozen=True)
class DefectsConfig:
    enabled: bool = False
    pixel_threshold: float | None = None
    pixel_threshold_strategy: str = "normal_pixel_quantile"
    pixel_normal_quantile: float = 0.999
    mask_format: str = "png"
    roi_xyxy_norm: tuple[float, float, float, float] | None = None
    min_area: int = 0
    open_ksize: int = 0
    close_ksize: int = 0
    fill_holes: bool = False
    max_regions: int | None = None


@dataclass(frozen=True)
class WorkbenchConfig:
    dataset: DatasetConfig
    model: ModelConfig
    recipe: str = "industrial-adapt"
    seed: int | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    defects: DefectsConfig = field(default_factory=DefectsConfig)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WorkbenchConfig":
        top = _require_mapping(raw, name="config")

        recipe = str(top.get("recipe", "industrial-adapt"))
        seed = _optional_int(top.get("seed", None), name="seed")

        ds_raw = _require_mapping(top.get("dataset", {}), name="dataset")
        ds_name = ds_raw.get("name", None)
        ds_root = ds_raw.get("root", None)
        if ds_name is None:
            raise ValueError("dataset.name is required")
        if ds_root is None:
            raise ValueError("dataset.root is required")

        ds_manifest_path = (
            str(ds_raw["manifest_path"]).strip()
            if ds_raw.get("manifest_path", None) is not None
            else None
        )
        if ds_manifest_path is not None and not ds_manifest_path:
            ds_manifest_path = None

        split_raw = ds_raw.get("split_policy", None)
        if split_raw is None:
            split_policy = SplitPolicyConfig(seed=seed)
        else:
            sp_map = _require_mapping(split_raw, name="dataset.split_policy")
            sp_seed = _optional_int(sp_map.get("seed", seed), name="dataset.split_policy.seed")
            tnf = _optional_float(
                sp_map.get("test_normal_fraction", 0.2),
                name="dataset.split_policy.test_normal_fraction",
            )
            split_policy = SplitPolicyConfig(
                mode=str(sp_map.get("mode", "benchmark")),
                scope=str(sp_map.get("scope", "category")),
                seed=sp_seed,
                test_normal_fraction=float(tnf if tnf is not None else 0.2),
            )

        dataset = DatasetConfig(
            name=str(ds_name),
            root=str(ds_root),
            manifest_path=ds_manifest_path,
            category=str(ds_raw.get("category", "all")),
            resize=_parse_resize(ds_raw.get("resize", None), default=(256, 256)),
            input_mode=str(ds_raw.get("input_mode", "paths")),
            limit_train=_optional_int(ds_raw.get("limit_train", None), name="dataset.limit_train"),
            limit_test=_optional_int(ds_raw.get("limit_test", None), name="dataset.limit_test"),
            split_policy=split_policy,
        )
        if str(dataset.name).lower() == "manifest" and dataset.manifest_path is None:
            raise ValueError("dataset.manifest_path is required when dataset.name='manifest'")

        model_raw = _require_mapping(top.get("model", {}), name="model")
        model_name = model_raw.get("name", None)
        if model_name is None:
            raise ValueError("model.name is required")

        mk_raw = model_raw.get("model_kwargs", None)
        if mk_raw is None:
            model_kwargs = {}
        else:
            model_kwargs = dict(_require_mapping(mk_raw, name="model.model_kwargs"))

        contamination = _optional_float(model_raw.get("contamination", 0.1), name="model.contamination")
        model = ModelConfig(
            name=str(model_name),
            device=str(model_raw.get("device", "cpu")),
            preset=(str(model_raw["preset"]) if model_raw.get("preset", None) is not None else None),
            pretrained=bool(model_raw.get("pretrained", True)),
            contamination=float(contamination if contamination is not None else 0.1),
            model_kwargs=model_kwargs,
            checkpoint_path=(
                str(model_raw["checkpoint_path"])
                if model_raw.get("checkpoint_path", None) is not None
                else None
            ),
        )

        out_raw = top.get("output", None)
        if out_raw is None:
            output = OutputConfig()
        else:
            out_map = _require_mapping(out_raw, name="output")
            output = OutputConfig(
                output_dir=(
                    str(out_map["output_dir"]) if out_map.get("output_dir", None) is not None else None
                ),
                save_run=bool(out_map.get("save_run", True)),
                per_image_jsonl=bool(out_map.get("per_image_jsonl", True)),
            )

        adaptation_raw = top.get("adaptation", None)
        if adaptation_raw is None:
            adaptation = AdaptationConfig()
        else:
            a_map = _require_mapping(adaptation_raw, name="adaptation")

            tiling_raw = a_map.get("tiling", None)
            if tiling_raw is None:
                tiling = TilingConfig()
            else:
                t_map = _require_mapping(tiling_raw, name="adaptation.tiling")
                tile_size = _optional_int(t_map.get("tile_size", None), name="adaptation.tiling.tile_size")
                stride = _optional_int(t_map.get("stride", None), name="adaptation.tiling.stride")
                if tile_size is not None and tile_size <= 0:
                    raise ValueError("adaptation.tiling.tile_size must be positive or null")
                if stride is not None and stride <= 0:
                    raise ValueError("adaptation.tiling.stride must be positive or null")
                score_topk = _optional_float(t_map.get("score_topk", 0.1), name="adaptation.tiling.score_topk")
                tiling = TilingConfig(
                    tile_size=tile_size,
                    stride=stride,
                    score_reduce=str(t_map.get("score_reduce", "max")),
                    score_topk=float(score_topk if score_topk is not None else 0.1),
                    map_reduce=str(t_map.get("map_reduce", "max")),
                )

            post_raw = a_map.get("postprocess", None)
            if post_raw is None:
                postprocess = None
            else:
                p_map = _require_mapping(post_raw, name="adaptation.postprocess")
                component_threshold = _optional_float(
                    p_map.get("component_threshold", None),
                    name="adaptation.postprocess.component_threshold",
                )
                postprocess = MapPostprocessConfig(
                    normalize=bool(p_map.get("normalize", True)),
                    normalize_method=str(p_map.get("normalize_method", "minmax")),
                    percentile_range=_parse_percentile_range(p_map.get("percentile_range", None)),
                    gaussian_sigma=float(
                        _optional_float(
                            p_map.get("gaussian_sigma", 0.0),
                            name="adaptation.postprocess.gaussian_sigma",
                        )
                        or 0.0
                    ),
                    morph_open_ksize=int(
                        _optional_int(
                            p_map.get("morph_open_ksize", 0),
                            name="adaptation.postprocess.morph_open_ksize",
                        )
                        or 0
                    ),
                    morph_close_ksize=int(
                        _optional_int(
                            p_map.get("morph_close_ksize", 0),
                            name="adaptation.postprocess.morph_close_ksize",
                        )
                        or 0
                    ),
                    component_threshold=component_threshold,
                    min_component_area=int(
                        _optional_int(
                            p_map.get("min_component_area", 0),
                            name="adaptation.postprocess.min_component_area",
                        )
                        or 0
                    ),
                )

            adaptation = AdaptationConfig(
                tiling=tiling,
                postprocess=postprocess,
                save_maps=bool(a_map.get("save_maps", False)),
            )

        training_raw = top.get("training", None)
        if training_raw is None:
            training = TrainingConfig()
        else:
            t_map = _require_mapping(training_raw, name="training")
            epochs = _optional_int(t_map.get("epochs", None), name="training.epochs")
            if epochs is not None and epochs <= 0:
                raise ValueError("training.epochs must be positive or null")
            lr = _optional_float(t_map.get("lr", None), name="training.lr")
            if lr is not None and lr <= 0:
                raise ValueError("training.lr must be positive or null")
            training = TrainingConfig(
                enabled=bool(t_map.get("enabled", False)),
                epochs=epochs,
                lr=lr,
                checkpoint_name=_parse_checkpoint_name(t_map.get("checkpoint_name", None)),
            )

        defects_raw = top.get("defects", None)
        if defects_raw is None:
            defects = DefectsConfig()
        else:
            d_map = _require_mapping(defects_raw, name="defects")
            pixel_threshold = _optional_float(d_map.get("pixel_threshold", None), name="defects.pixel_threshold")
            max_regions = _optional_int(d_map.get("max_regions", None), name="defects.max_regions")
            min_area = int(_optional_int(d_map.get("min_area", 0), name="defects.min_area") or 0)
            open_ksize = int(_optional_int(d_map.get("open_ksize", 0), name="defects.open_ksize") or 0)
            close_ksize = int(_optional_int(d_map.get("close_ksize", 0), name="defects.close_ksize") or 0)

            if min_area < 0:
                raise ValueError("defects.min_area must be >= 0")
            if open_ksize < 0:
                raise ValueError("defects.open_ksize must be >= 0")
            if close_ksize < 0:
                raise ValueError("defects.close_ksize must be >= 0")
            if max_regions is not None and max_regions <= 0:
                raise ValueError("defects.max_regions must be positive or null")

            q = _optional_float(d_map.get("pixel_normal_quantile", 0.999), name="defects.pixel_normal_quantile")
            qv = float(q if q is not None else 0.999)
            if not (0.0 < qv <= 1.0):
                raise ValueError("defects.pixel_normal_quantile must be in (0,1]")

            mask_format = str(d_map.get("mask_format", "png"))
            if mask_format not in ("png", "npy"):
                raise ValueError("defects.mask_format must be 'png' or 'npy'")

            defects = DefectsConfig(
                enabled=bool(d_map.get("enabled", False)),
                pixel_threshold=(float(pixel_threshold) if pixel_threshold is not None else None),
                pixel_threshold_strategy=str(d_map.get("pixel_threshold_strategy", "normal_pixel_quantile")),
                pixel_normal_quantile=qv,
                mask_format=mask_format,
                roi_xyxy_norm=_parse_roi_xyxy_norm(d_map.get("roi_xyxy_norm", None)),
                min_area=min_area,
                open_ksize=open_ksize,
                close_ksize=close_ksize,
                fill_holes=bool(d_map.get("fill_holes", False)),
                max_regions=max_regions,
            )

        return cls(
            dataset=dataset,
            model=model,
            recipe=recipe,
            seed=seed,
            output=output,
            adaptation=adaptation,
            training=training,
            defects=defects,
        )
