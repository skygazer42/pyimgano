from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pyimgano.workbench.adaptation import AdaptationConfig, MapPostprocessConfig, TilingConfig
from pyimgano.preprocessing.industrial_presets import IlluminationContrastKnobs


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


def _parse_int_pair(
    value: Any,
    *,
    name: str,
    default: tuple[int, int],
) -> tuple[int, int]:
    if value is None:
        return (int(default[0]), int(default[1]))

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a list/tuple of length 2, got {value!r}")
    try:
        a = int(value[0])
        b = int(value[1])
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"{name} must contain ints, got {value!r}") from exc
    if a <= 0 or b <= 0:
        raise ValueError(f"{name} must be positive ints, got {(a, b)}")
    return (a, b)


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
class MapSmoothingConfig:
    method: str = "none"
    ksize: int = 0
    sigma: float = 0.0


@dataclass(frozen=True)
class HysteresisConfig:
    enabled: bool = False
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class ShapeFiltersConfig:
    min_fill_ratio: float | None = None
    max_aspect_ratio: float | None = None
    min_solidity: float | None = None


@dataclass(frozen=True)
class MergeNearbyConfig:
    enabled: bool = False
    max_gap_px: int = 0


@dataclass(frozen=True)
class DefectsConfig:
    enabled: bool = False
    pixel_threshold: float | None = None
    pixel_threshold_strategy: str = "normal_pixel_quantile"
    pixel_normal_quantile: float = 0.999
    mask_format: str = "png"
    roi_xyxy_norm: tuple[float, float, float, float] | None = None
    border_ignore_px: int = 0
    map_smoothing: MapSmoothingConfig = field(default_factory=MapSmoothingConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    shape_filters: ShapeFiltersConfig = field(default_factory=ShapeFiltersConfig)
    merge_nearby: MergeNearbyConfig = field(default_factory=MergeNearbyConfig)
    min_area: int = 0
    min_score_max: float | None = None
    min_score_mean: float | None = None
    open_ksize: int = 0
    close_ksize: int = 0
    fill_holes: bool = False
    max_regions: int | None = None
    max_regions_sort_by: str = "score_max"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Optional preprocessing configuration for industrial runs."""

    illumination_contrast: IlluminationContrastKnobs | None = None


@dataclass(frozen=True)
class WorkbenchConfig:
    dataset: DatasetConfig
    model: ModelConfig
    recipe: str = "industrial-adapt"
    seed: int | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
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

        preprocessing_raw = top.get("preprocessing", None)
        if preprocessing_raw is None:
            preprocessing = PreprocessingConfig()
        else:
            p_map = _require_mapping(preprocessing_raw, name="preprocessing")
            ic_raw = p_map.get("illumination_contrast", None)
            if ic_raw is None:
                illumination_contrast = None
            else:
                ic_map = _require_mapping(ic_raw, name="preprocessing.illumination_contrast")

                wb_raw = str(ic_map.get("white_balance", "none")).strip().lower()
                if wb_raw in ("", "none"):
                    white_balance = "none"
                elif wb_raw in ("gray_world", "gray-world", "grayworld"):
                    white_balance = "gray_world"
                elif wb_raw in ("max_rgb", "max-rgb", "maxrgb"):
                    white_balance = "max_rgb"
                else:
                    raise ValueError(
                        "preprocessing.illumination_contrast.white_balance must be one of: "
                        "none|gray_world|max_rgb"
                    )

                cutoff = _optional_float(
                    ic_map.get("homomorphic_cutoff", 0.5),
                    name="preprocessing.illumination_contrast.homomorphic_cutoff",
                )
                cutoff_v = float(cutoff if cutoff is not None else 0.5)
                if not (0.0 < cutoff_v <= 1.0):
                    raise ValueError(
                        "preprocessing.illumination_contrast.homomorphic_cutoff must be in (0,1]"
                    )

                gamma_low = _optional_float(
                    ic_map.get("homomorphic_gamma_low", 0.7),
                    name="preprocessing.illumination_contrast.homomorphic_gamma_low",
                )
                gamma_high = _optional_float(
                    ic_map.get("homomorphic_gamma_high", 1.5),
                    name="preprocessing.illumination_contrast.homomorphic_gamma_high",
                )
                c = _optional_float(
                    ic_map.get("homomorphic_c", 1.0),
                    name="preprocessing.illumination_contrast.homomorphic_c",
                )

                clahe_tile_grid_size = _parse_int_pair(
                    ic_map.get("clahe_tile_grid_size", None),
                    name="preprocessing.illumination_contrast.clahe_tile_grid_size",
                    default=(8, 8),
                )
                clahe_clip_limit = _optional_float(
                    ic_map.get("clahe_clip_limit", 2.0),
                    name="preprocessing.illumination_contrast.clahe_clip_limit",
                )
                clahe_clip_limit_v = float(clahe_clip_limit if clahe_clip_limit is not None else 2.0)
                if clahe_clip_limit_v <= 0.0:
                    raise ValueError(
                        "preprocessing.illumination_contrast.clahe_clip_limit must be > 0"
                    )

                gamma = _optional_float(
                    ic_map.get("gamma", None),
                    name="preprocessing.illumination_contrast.gamma",
                )
                if gamma is not None and float(gamma) <= 0.0:
                    raise ValueError("preprocessing.illumination_contrast.gamma must be > 0 or null")

                lp = _optional_float(
                    ic_map.get("contrast_lower_percentile", 2.0),
                    name="preprocessing.illumination_contrast.contrast_lower_percentile",
                )
                up = _optional_float(
                    ic_map.get("contrast_upper_percentile", 98.0),
                    name="preprocessing.illumination_contrast.contrast_upper_percentile",
                )
                lp_v = float(lp if lp is not None else 2.0)
                up_v = float(up if up is not None else 98.0)
                if not (0.0 <= lp_v <= 100.0 and 0.0 <= up_v <= 100.0 and lp_v < up_v):
                    raise ValueError(
                        "preprocessing.illumination_contrast contrast percentiles must satisfy "
                        "0<=lower<upper<=100"
                    )

                illumination_contrast = IlluminationContrastKnobs(
                    white_balance=white_balance,
                    homomorphic=bool(ic_map.get("homomorphic", False)),
                    homomorphic_cutoff=cutoff_v,
                    homomorphic_gamma_low=float(gamma_low if gamma_low is not None else 0.7),
                    homomorphic_gamma_high=float(gamma_high if gamma_high is not None else 1.5),
                    homomorphic_c=float(c if c is not None else 1.0),
                    homomorphic_per_channel=bool(ic_map.get("homomorphic_per_channel", True)),
                    clahe=bool(ic_map.get("clahe", False)),
                    clahe_clip_limit=clahe_clip_limit_v,
                    clahe_tile_grid_size=clahe_tile_grid_size,
                    gamma=(float(gamma) if gamma is not None else None),
                    contrast_stretch=bool(ic_map.get("contrast_stretch", False)),
                    contrast_lower_percentile=lp_v,
                    contrast_upper_percentile=up_v,
                )

            preprocessing = PreprocessingConfig(illumination_contrast=illumination_contrast)

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
            max_regions_sort_by = str(d_map.get("max_regions_sort_by", "score_max")).lower().strip()
            border_ignore_px = int(
                _optional_int(d_map.get("border_ignore_px", 0), name="defects.border_ignore_px") or 0
            )
            min_area = int(_optional_int(d_map.get("min_area", 0), name="defects.min_area") or 0)
            min_score_max = _optional_float(d_map.get("min_score_max", None), name="defects.min_score_max")
            min_score_mean = _optional_float(d_map.get("min_score_mean", None), name="defects.min_score_mean")
            open_ksize = int(_optional_int(d_map.get("open_ksize", 0), name="defects.open_ksize") or 0)
            close_ksize = int(_optional_int(d_map.get("close_ksize", 0), name="defects.close_ksize") or 0)

            if border_ignore_px < 0:
                raise ValueError("defects.border_ignore_px must be >= 0")
            if min_area < 0:
                raise ValueError("defects.min_area must be >= 0")
            if open_ksize < 0:
                raise ValueError("defects.open_ksize must be >= 0")
            if close_ksize < 0:
                raise ValueError("defects.close_ksize must be >= 0")
            if max_regions is not None and max_regions <= 0:
                raise ValueError("defects.max_regions must be positive or null")
            if max_regions_sort_by not in ("score_max", "score_mean", "area"):
                raise ValueError("defects.max_regions_sort_by must be one of: score_max|score_mean|area")
            if min_score_max is not None and float(min_score_max) < 0.0:
                raise ValueError("defects.min_score_max must be >= 0 or null")
            if min_score_mean is not None and float(min_score_mean) < 0.0:
                raise ValueError("defects.min_score_mean must be >= 0 or null")

            q = _optional_float(d_map.get("pixel_normal_quantile", 0.999), name="defects.pixel_normal_quantile")
            qv = float(q if q is not None else 0.999)
            if not (0.0 < qv <= 1.0):
                raise ValueError("defects.pixel_normal_quantile must be in (0,1]")

            mask_format = str(d_map.get("mask_format", "png"))
            if mask_format not in ("png", "npy"):
                raise ValueError("defects.mask_format must be 'png' or 'npy'")

            ms_raw = d_map.get("map_smoothing", None)
            if ms_raw is None:
                map_smoothing = MapSmoothingConfig()
            else:
                ms_map = _require_mapping(ms_raw, name="defects.map_smoothing")
                ms_method = str(ms_map.get("method", "none")).lower().strip()
                if ms_method not in ("none", "median", "gaussian", "box"):
                    raise ValueError(
                        "defects.map_smoothing.method must be one of: none|median|gaussian|box"
                    )
                ms_ksize = int(
                    _optional_int(ms_map.get("ksize", 0), name="defects.map_smoothing.ksize") or 0
                )
                ms_sigma = _optional_float(ms_map.get("sigma", 0.0), name="defects.map_smoothing.sigma")
                ms_sigma_v = float(ms_sigma if ms_sigma is not None else 0.0)
                if ms_ksize < 0:
                    raise ValueError("defects.map_smoothing.ksize must be >= 0")
                if ms_sigma_v < 0.0:
                    raise ValueError("defects.map_smoothing.sigma must be >= 0")

                if ms_method in ("median", "box") and ms_ksize not in (0, 1) and ms_ksize < 3:
                    raise ValueError("defects.map_smoothing.ksize must be >= 3 for median/box smoothing")

                map_smoothing = MapSmoothingConfig(
                    method=ms_method,
                    ksize=ms_ksize,
                    sigma=ms_sigma_v,
                )

            hyst_raw = d_map.get("hysteresis", None)
            if hyst_raw is None:
                hysteresis = HysteresisConfig()
            else:
                hyst_map = _require_mapping(hyst_raw, name="defects.hysteresis")
                hyst_enabled = bool(hyst_map.get("enabled", False))
                hyst_low = _optional_float(hyst_map.get("low", None), name="defects.hysteresis.low")
                hyst_high = _optional_float(hyst_map.get("high", None), name="defects.hysteresis.high")
                if hyst_low is not None and float(hyst_low) < 0.0:
                    raise ValueError("defects.hysteresis.low must be >= 0 or null")
                if hyst_high is not None and float(hyst_high) < 0.0:
                    raise ValueError("defects.hysteresis.high must be >= 0 or null")
                hysteresis = HysteresisConfig(
                    enabled=hyst_enabled,
                    low=(float(hyst_low) if hyst_low is not None else None),
                    high=(float(hyst_high) if hyst_high is not None else None),
                )

            sf_raw = d_map.get("shape_filters", None)
            if sf_raw is None:
                shape_filters = ShapeFiltersConfig()
            else:
                sf_map = _require_mapping(sf_raw, name="defects.shape_filters")
                sf_min_fill_ratio = _optional_float(
                    sf_map.get("min_fill_ratio", None),
                    name="defects.shape_filters.min_fill_ratio",
                )
                sf_max_aspect_ratio = _optional_float(
                    sf_map.get("max_aspect_ratio", None),
                    name="defects.shape_filters.max_aspect_ratio",
                )
                sf_min_solidity = _optional_float(
                    sf_map.get("min_solidity", None),
                    name="defects.shape_filters.min_solidity",
                )

                if sf_min_fill_ratio is not None and not (0.0 <= float(sf_min_fill_ratio) <= 1.0):
                    raise ValueError("defects.shape_filters.min_fill_ratio must be in [0,1] or null")
                if sf_max_aspect_ratio is not None and float(sf_max_aspect_ratio) < 1.0:
                    raise ValueError("defects.shape_filters.max_aspect_ratio must be >= 1.0 or null")
                if sf_min_solidity is not None and not (0.0 <= float(sf_min_solidity) <= 1.0):
                    raise ValueError("defects.shape_filters.min_solidity must be in [0,1] or null")

                shape_filters = ShapeFiltersConfig(
                    min_fill_ratio=(float(sf_min_fill_ratio) if sf_min_fill_ratio is not None else None),
                    max_aspect_ratio=(
                        float(sf_max_aspect_ratio) if sf_max_aspect_ratio is not None else None
                    ),
                    min_solidity=(float(sf_min_solidity) if sf_min_solidity is not None else None),
                )

            merge_raw = d_map.get("merge_nearby", None)
            if merge_raw is None:
                merge_nearby = MergeNearbyConfig()
            else:
                merge_map = _require_mapping(merge_raw, name="defects.merge_nearby")
                merge_enabled = bool(merge_map.get("enabled", False))
                merge_gap = int(
                    _optional_int(
                        merge_map.get("max_gap_px", 0),
                        name="defects.merge_nearby.max_gap_px",
                    )
                    or 0
                )
                if merge_gap < 0:
                    raise ValueError("defects.merge_nearby.max_gap_px must be >= 0")
                merge_nearby = MergeNearbyConfig(enabled=merge_enabled, max_gap_px=merge_gap)

            defects = DefectsConfig(
                enabled=bool(d_map.get("enabled", False)),
                pixel_threshold=(float(pixel_threshold) if pixel_threshold is not None else None),
                pixel_threshold_strategy=str(d_map.get("pixel_threshold_strategy", "normal_pixel_quantile")),
                pixel_normal_quantile=qv,
                mask_format=mask_format,
                roi_xyxy_norm=_parse_roi_xyxy_norm(d_map.get("roi_xyxy_norm", None)),
                border_ignore_px=border_ignore_px,
                map_smoothing=map_smoothing,
                hysteresis=hysteresis,
                shape_filters=shape_filters,
                merge_nearby=merge_nearby,
                min_area=min_area,
                min_score_max=(float(min_score_max) if min_score_max is not None else None),
                min_score_mean=(float(min_score_mean) if min_score_mean is not None else None),
                open_ksize=open_ksize,
                close_ksize=close_ksize,
                fill_holes=bool(d_map.get("fill_holes", False)),
                max_regions=max_regions,
                max_regions_sort_by=max_regions_sort_by,
            )

        return cls(
            dataset=dataset,
            model=model,
            recipe=recipe,
            seed=seed,
            output=output,
            adaptation=adaptation,
            preprocessing=preprocessing,
            training=training,
            defects=defects,
        )
