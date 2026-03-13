from __future__ import annotations

import importlib

import pytest

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.config_parse_primitives import _parse_resize, _require_mapping
from pyimgano.workbench.config_parser import build_workbench_config_from_dict
from pyimgano.workbench.config_section_parsers import (
    _parse_adaptation_config,
    _parse_dataset_config,
    _parse_defects_config,
    _parse_model_config,
    _parse_output_config,
    _parse_preprocessing_config,
    _parse_training_config,
)
from pyimgano.workbench.config_types import WorkbenchConfig as WorkbenchConfigType


def test_workbench_config_parser_builds_compatible_workbench_config() -> None:
    raw = {
        "recipe": "industrial-adapt",
        "seed": 123,
        "dataset": {
            "name": "custom",
            "root": "/tmp/data",
            "category": "all",
            "resize": [128, 256],
            "split_policy": {"test_normal_fraction": 0.3},
        },
        "model": {
            "name": "vision_patchcore",
            "device": "cpu",
            "contamination": 0.2,
        },
        "training": {"enabled": True, "epochs": 2, "lr": 0.001},
    }

    cfg = build_workbench_config_from_dict(raw)

    assert isinstance(cfg, WorkbenchConfigType)
    assert isinstance(cfg, WorkbenchConfig)
    assert cfg == WorkbenchConfig.from_dict(raw)
    assert cfg.dataset.resize == (128, 256)
    assert cfg.dataset.split_policy.seed == 123
    assert cfg.training.enabled is True
    assert cfg.training.epochs == 2
    assert cfg.training.lr == pytest.approx(0.001)


def test_workbench_config_parser_preserves_validation_messages() -> None:
    raw = {
        "dataset": {"name": "custom", "root": "/tmp/data", "resize": [0, 16]},
        "model": {"name": "vision_patchcore"},
    }

    with pytest.raises(ValueError, match=r"resize must be positive"):
        build_workbench_config_from_dict(raw)


def test_config_parse_primitives_preserve_validation_behavior() -> None:
    with pytest.raises(ValueError, match=r"config must be a dict/object"):
        _require_mapping([], name="config")

    with pytest.raises(ValueError, match=r"resize must be positive"):
        _parse_resize([0, 16], default=(256, 256))


def test_config_section_parsers_build_compatible_sections() -> None:
    top = {
        "seed": 5,
        "dataset": {
            "name": "manifest",
            "root": "/tmp/data",
            "manifest_path": "/tmp/manifest.jsonl",
            "resize": [32, 64],
        },
        "training": {"enabled": True, "epochs": "2", "lr": "0.001"},
    }

    dataset = _parse_dataset_config(top, seed=5)
    training = _parse_training_config(top)

    assert dataset.name == "manifest"
    assert dataset.resize == (32, 64)
    assert dataset.split_policy.seed == 5
    assert training.enabled is True
    assert training.epochs == 2
    assert training.lr == pytest.approx(0.001)


def test_config_training_section_parser_builds_compatible_section() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.config_training_section_parser")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing config training section helper: {exc}")

    top = {
        "training": {
            "enabled": True,
            "epochs": "3",
            "lr": "0.002",
            "checkpoint_name": "weights.pt",
        }
    }

    training = helper_module._parse_training_config(top)

    assert training.enabled is True
    assert training.epochs == 3
    assert training.lr == pytest.approx(0.002)
    assert training.checkpoint_name == "weights.pt"


def test_config_adaptation_section_parser_builds_compatible_section() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.config_adaptation_section_parser")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing config adaptation section helper: {exc}")

    top = {
        "adaptation": {
            "save_maps": True,
            "tiling": {
                "tile_size": "256",
                "stride": "128",
                "score_reduce": "topk_mean",
                "score_topk": "0.2",
                "map_reduce": "hann",
            },
            "postprocess": {
                "normalize": True,
                "normalize_method": "percentile",
                "percentile_range": [5, 95],
                "gaussian_sigma": "1.0",
                "morph_open_ksize": "3",
                "morph_close_ksize": "5",
                "component_threshold": "0.5",
                "min_component_area": "10",
            },
        }
    }

    adaptation = helper_module._parse_adaptation_config(top)

    assert adaptation.save_maps is True
    assert adaptation.tiling.tile_size == 256
    assert adaptation.tiling.stride == 128
    assert adaptation.tiling.score_reduce == "topk_mean"
    assert adaptation.tiling.score_topk == pytest.approx(0.2)
    assert adaptation.tiling.map_reduce == "hann"
    assert adaptation.postprocess is not None
    assert adaptation.postprocess.normalize_method == "percentile"
    assert adaptation.postprocess.percentile_range == (5.0, 95.0)
    assert adaptation.postprocess.gaussian_sigma == pytest.approx(1.0)
    assert adaptation.postprocess.morph_open_ksize == 3
    assert adaptation.postprocess.morph_close_ksize == 5
    assert adaptation.postprocess.component_threshold == pytest.approx(0.5)
    assert adaptation.postprocess.min_component_area == 10


def test_config_dataset_section_parser_builds_compatible_dataset_section() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.config_dataset_section_parser")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing config dataset section helper: {exc}")

    top = {
        "seed": 7,
        "dataset": {
            "name": "manifest",
            "root": "/tmp/data",
            "manifest_path": "/tmp/manifest.jsonl",
            "resize": [48, 96],
            "split_policy": {"test_normal_fraction": 0.3},
        },
    }

    dataset = helper_module._parse_dataset_config(top, seed=7)
    split_policy = helper_module._parse_split_policy_config(
        {"mode": "grouped", "scope": "category", "seed": 9, "test_normal_fraction": 0.4},
        seed=7,
    )

    assert dataset.name == "manifest"
    assert dataset.resize == (48, 96)
    assert dataset.split_policy.seed == 7
    assert dataset.split_policy.test_normal_fraction == pytest.approx(0.3)
    assert split_policy.mode == "grouped"
    assert split_policy.seed == 9
    assert split_policy.test_normal_fraction == pytest.approx(0.4)


def test_config_model_output_section_parser_builds_compatible_sections() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.config_model_output_section_parser")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing config model/output section helper: {exc}")

    top = {
        "model": {
            "name": "vision_patchcore",
            "device": "cpu",
            "contamination": 0.2,
            "model_kwargs": {"layers": ["layer2"]},
        },
        "output": {
            "output_dir": "/tmp/runs",
            "save_run": False,
            "per_image_jsonl": False,
        },
    }

    model = helper_module._parse_model_config(top)
    output = helper_module._parse_output_config(top)

    assert model.name == "vision_patchcore"
    assert model.contamination == pytest.approx(0.2)
    assert model.model_kwargs == {"layers": ["layer2"]}
    assert output.output_dir == "/tmp/runs"
    assert output.save_run is False
    assert output.per_image_jsonl is False


def test_config_section_parsers_build_compatible_model_output_sections() -> None:
    top = {
        "model": {
            "name": "vision_patchcore",
            "device": "cpu",
            "contamination": "0.25",
        },
        "output": {
            "output_dir": "/tmp/results",
            "save_run": True,
            "per_image_jsonl": False,
        },
    }

    model = _parse_model_config(top)
    output = _parse_output_config(top)

    assert model.name == "vision_patchcore"
    assert model.contamination == pytest.approx(0.25)
    assert output.output_dir == "/tmp/results"
    assert output.save_run is True
    assert output.per_image_jsonl is False


def test_config_section_parsers_build_compatible_adaptation_section() -> None:
    top = {
        "adaptation": {
            "save_maps": False,
            "tiling": {
                "tile_size": 512,
                "stride": 256,
                "score_topk": "0.1",
            },
            "postprocess": {
                "gaussian_sigma": "1.25",
                "component_threshold": 0.6,
                "min_component_area": "11",
            },
        }
    }

    adaptation = _parse_adaptation_config(top)

    assert adaptation.save_maps is False
    assert adaptation.tiling.tile_size == 512
    assert adaptation.tiling.stride == 256
    assert adaptation.tiling.score_topk == pytest.approx(0.1)
    assert adaptation.postprocess is not None
    assert adaptation.postprocess.gaussian_sigma == pytest.approx(1.25)
    assert adaptation.postprocess.component_threshold == pytest.approx(0.6)
    assert adaptation.postprocess.min_component_area == 11


def test_config_defects_section_parser_builds_compatible_section() -> None:
    try:
        helper_module = importlib.import_module("pyimgano.workbench.config_defects_section_parser")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing config defects section helper: {exc}")

    top = {
        "defects": {
            "enabled": True,
            "pixel_threshold": "0.5",
            "pixel_threshold_strategy": "fixed",
            "pixel_normal_quantile": "0.99",
            "mask_format": "npy",
            "roi_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            "border_ignore_px": "2",
            "map_smoothing": {"method": "gaussian", "ksize": "5", "sigma": "1.25"},
            "hysteresis": {"enabled": True, "low": "0.2", "high": 0.8},
            "shape_filters": {
                "min_fill_ratio": "0.2",
                "max_aspect_ratio": 3.0,
                "min_solidity": "0.8",
            },
            "merge_nearby": {"enabled": True, "max_gap_px": "2"},
            "min_area": "10",
            "min_score_max": "0.9",
            "min_score_mean": 0.5,
            "open_ksize": "3",
            "close_ksize": 5,
            "fill_holes": True,
            "max_regions": "7",
            "max_regions_sort_by": "area",
        }
    }

    defects = helper_module._parse_defects_config(top)

    assert defects.enabled is True
    assert defects.pixel_threshold == pytest.approx(0.5)
    assert defects.pixel_threshold_strategy == "fixed"
    assert defects.pixel_normal_quantile == pytest.approx(0.99)
    assert defects.mask_format == "npy"
    assert defects.roi_xyxy_norm == pytest.approx((0.1, 0.2, 0.8, 0.9))
    assert defects.border_ignore_px == 2
    assert defects.map_smoothing.method == "gaussian"
    assert defects.map_smoothing.ksize == 5
    assert defects.map_smoothing.sigma == pytest.approx(1.25)
    assert defects.hysteresis.enabled is True
    assert defects.hysteresis.low == pytest.approx(0.2)
    assert defects.hysteresis.high == pytest.approx(0.8)
    assert defects.shape_filters.min_fill_ratio == pytest.approx(0.2)
    assert defects.shape_filters.max_aspect_ratio == pytest.approx(3.0)
    assert defects.shape_filters.min_solidity == pytest.approx(0.8)
    assert defects.merge_nearby.enabled is True
    assert defects.merge_nearby.max_gap_px == 2
    assert defects.min_area == 10
    assert defects.min_score_max == pytest.approx(0.9)
    assert defects.min_score_mean == pytest.approx(0.5)
    assert defects.open_ksize == 3
    assert defects.close_ksize == 5
    assert defects.fill_holes is True
    assert defects.max_regions == 7
    assert defects.max_regions_sort_by == "area"


def test_config_preprocessing_section_parser_builds_compatible_section() -> None:
    try:
        helper_module = importlib.import_module(
            "pyimgano.workbench.config_preprocessing_section_parser"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing config preprocessing section helper: {exc}")

    top = {
        "preprocessing": {
            "illumination_contrast": {
                "white_balance": "gray-world",
                "homomorphic": True,
                "homomorphic_cutoff": "0.6",
                "clahe": True,
                "clahe_clip_limit": "1.5",
                "clahe_tile_grid_size": [4, 6],
                "gamma": "0.9",
                "contrast_stretch": True,
                "contrast_lower_percentile": 1.0,
                "contrast_upper_percentile": 99.0,
            }
        }
    }

    preprocessing = helper_module._parse_preprocessing_config(top)

    assert preprocessing.illumination_contrast is not None
    assert preprocessing.illumination_contrast.white_balance == "gray_world"
    assert preprocessing.illumination_contrast.homomorphic is True
    assert preprocessing.illumination_contrast.homomorphic_cutoff == pytest.approx(0.6)
    assert preprocessing.illumination_contrast.clahe is True
    assert preprocessing.illumination_contrast.clahe_clip_limit == pytest.approx(1.5)
    assert preprocessing.illumination_contrast.clahe_tile_grid_size == (4, 6)
    assert preprocessing.illumination_contrast.gamma == pytest.approx(0.9)
    assert preprocessing.illumination_contrast.contrast_stretch is True
    assert preprocessing.illumination_contrast.contrast_lower_percentile == pytest.approx(1.0)
    assert preprocessing.illumination_contrast.contrast_upper_percentile == pytest.approx(99.0)


def test_config_section_parsers_build_compatible_preprocessing_section() -> None:
    top = {
        "preprocessing": {
            "illumination_contrast": {
                "white_balance": "maxrgb",
                "clahe_clip_limit": "2.5",
                "gamma": 1.2,
            }
        }
    }

    preprocessing = _parse_preprocessing_config(top)

    assert preprocessing.illumination_contrast is not None
    assert preprocessing.illumination_contrast.white_balance == "max_rgb"
    assert preprocessing.illumination_contrast.clahe_clip_limit == pytest.approx(2.5)
    assert preprocessing.illumination_contrast.gamma == pytest.approx(1.2)


def test_config_section_parsers_build_compatible_defects_section() -> None:
    top = {
        "defects": {
            "pixel_threshold": "0.5",
            "map_smoothing": {"method": "median", "ksize": "3", "sigma": "0.0"},
            "merge_nearby": {"enabled": True, "max_gap_px": "1"},
            "max_regions_sort_by": "score_mean",
        }
    }

    defects = _parse_defects_config(top)

    assert defects.pixel_threshold == pytest.approx(0.5)
    assert defects.map_smoothing.method == "median"
    assert defects.map_smoothing.ksize == 3
    assert defects.map_smoothing.sigma == pytest.approx(0.0)
    assert defects.merge_nearby.enabled is True
    assert defects.merge_nearby.max_gap_px == 1
    assert defects.max_regions_sort_by == "score_mean"
