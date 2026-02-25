from pathlib import Path

from pyimgano.config import load_config
from pyimgano.workbench.config import WorkbenchConfig


def test_examples_configs_load_and_parse():
    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "examples" / "configs"

    paths = [
        configs_dir / "industrial_adapt_fast.json",
        configs_dir / "industrial_adapt_defects_roi.json",
        configs_dir / "industrial_adapt_defects_fp40.json",
        configs_dir / "industrial_adapt_maps_tiling.json",
        configs_dir / "manifest_industrial_adapt_fast.json",
        configs_dir / "micro_finetune_autoencoder.json",
    ]

    for p in paths:
        raw = load_config(p)
        cfg = WorkbenchConfig.from_dict(raw)
        assert isinstance(cfg.recipe, str)
        assert cfg.dataset.name
        assert cfg.model.name
