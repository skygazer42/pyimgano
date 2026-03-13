from __future__ import annotations

from typing import Any, Mapping

from pyimgano.workbench.config_parse_primitives import (
    _optional_float,
    _optional_int,
    _parse_resize,
    _require_mapping,
)
from pyimgano.workbench.config_types import DatasetConfig, SplitPolicyConfig


def _parse_split_policy_config(value: Any, *, seed: int | None) -> SplitPolicyConfig:
    if value is None:
        return SplitPolicyConfig(seed=seed)

    sp_map = _require_mapping(value, name="dataset.split_policy")
    sp_seed = _optional_int(sp_map.get("seed", seed), name="dataset.split_policy.seed")
    tnf = _optional_float(
        sp_map.get("test_normal_fraction", 0.2),
        name="dataset.split_policy.test_normal_fraction",
    )
    return SplitPolicyConfig(
        mode=str(sp_map.get("mode", "benchmark")),
        scope=str(sp_map.get("scope", "category")),
        seed=sp_seed,
        test_normal_fraction=float(tnf if tnf is not None else 0.2),
    )


def _parse_dataset_config(top: Mapping[str, Any], *, seed: int | None) -> DatasetConfig:
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

    dataset = DatasetConfig(
        name=str(ds_name),
        root=str(ds_root),
        manifest_path=ds_manifest_path,
        category=str(ds_raw.get("category", "all")),
        resize=_parse_resize(ds_raw.get("resize", None), default=(256, 256)),
        input_mode=str(ds_raw.get("input_mode", "paths")),
        limit_train=_optional_int(ds_raw.get("limit_train", None), name="dataset.limit_train"),
        limit_test=_optional_int(ds_raw.get("limit_test", None), name="dataset.limit_test"),
        split_policy=_parse_split_policy_config(ds_raw.get("split_policy", None), seed=seed),
    )
    if str(dataset.name).lower() == "manifest" and dataset.manifest_path is None:
        raise ValueError("dataset.manifest_path is required when dataset.name='manifest'")
    return dataset


__all__ = ["_parse_split_policy_config", "_parse_dataset_config"]
