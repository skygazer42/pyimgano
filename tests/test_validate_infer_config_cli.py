from __future__ import annotations

import json
from pathlib import Path


def test_validate_infer_config_cli_smoke(tmp_path: Path, capsys) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {
                    "name": "vision_patchcore",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "model_kwargs": {},
                },
                "defects": {
                    "enabled": True,
                    "pixel_threshold_strategy": "normal_pixel_quantile",
                    "pixel_normal_quantile": 0.999,
                    "mask_format": "png",
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "ok" in out


def test_validate_infer_config_cli_rejects_bad_mask_format(tmp_path: Path, capsys) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "defects": {"mask_format": "nope"},
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "mask_format" in err

