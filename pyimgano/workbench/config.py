from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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

        return cls(
            dataset=dataset,
            model=model,
            recipe=recipe,
            seed=seed,
            output=output,
        )

