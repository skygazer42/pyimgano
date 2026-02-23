from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    category: str = "all"
    resize: tuple[int, int] = (256, 256)
    input_mode: str = "paths"
    limit_train: int | None = None
    limit_test: int | None = None


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
class WorkbenchConfig:
    dataset: DatasetConfig
    model: ModelConfig
    recipe: str = "industrial-adapt"
    seed: int | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)

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

        dataset = DatasetConfig(
            name=str(ds_name),
            root=str(ds_root),
            category=str(ds_raw.get("category", "all")),
            resize=_parse_resize(ds_raw.get("resize", None), default=(256, 256)),
            input_mode=str(ds_raw.get("input_mode", "paths")),
            limit_train=_optional_int(ds_raw.get("limit_train", None), name="dataset.limit_train"),
            limit_test=_optional_int(ds_raw.get("limit_test", None), name="dataset.limit_test"),
        )

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

        return cls(
            dataset=dataset,
            model=model,
            recipe=recipe,
            seed=seed,
            output=output,
            adaptation=adaptation,
        )
