import pytest

from pyimgano.workbench.config import WorkbenchConfig


def test_workbench_config_adaptation_defaults():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
        }
    )
    assert cfg.adaptation.save_maps is False
    assert cfg.adaptation.tiling.tile_size is None
    assert cfg.adaptation.postprocess is None


def test_workbench_config_adaptation_parses_sections():
    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
            "adaptation": {
                "save_maps": True,
                "tiling": {
                    "tile_size": 256,
                    "stride": 128,
                    "score_reduce": "topk_mean",
                    "score_topk": 0.2,
                    "map_reduce": "hann",
                },
                "postprocess": {
                    "normalize": True,
                    "normalize_method": "percentile",
                    "percentile_range": [5, 95],
                    "gaussian_sigma": 1.0,
                    "morph_open_ksize": 3,
                    "morph_close_ksize": 5,
                    "component_threshold": 0.5,
                    "min_component_area": 10,
                },
            },
        }
    )
    assert cfg.adaptation.save_maps is True
    assert cfg.adaptation.tiling.tile_size == 256
    assert cfg.adaptation.tiling.stride == 128
    assert cfg.adaptation.tiling.score_reduce == "topk_mean"
    assert cfg.adaptation.tiling.score_topk == 0.2
    assert cfg.adaptation.tiling.map_reduce == "hann"
    assert cfg.adaptation.postprocess is not None
    assert cfg.adaptation.postprocess.normalize_method == "percentile"
    assert cfg.adaptation.postprocess.percentile_range == (5.0, 95.0)
    assert cfg.adaptation.postprocess.gaussian_sigma == 1.0
    assert cfg.adaptation.postprocess.morph_open_ksize == 3
    assert cfg.adaptation.postprocess.morph_close_ksize == 5
    assert cfg.adaptation.postprocess.component_threshold == 0.5
    assert cfg.adaptation.postprocess.min_component_area == 10


def test_workbench_config_adaptation_invalid_tiling_raises():
    raw = {
        "dataset": {"name": "custom", "root": "/tmp/data"},
        "model": {"name": "vision_patchcore"},
        "adaptation": {"tiling": {"tile_size": 0}},
    }
    with pytest.raises(ValueError):
        WorkbenchConfig.from_dict(raw)

