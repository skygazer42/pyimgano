from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyimgano.cli import main as cli_main


def _write_png(path: Path, *, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cli_manifest_embedding_plus_core_smoke(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # Hard block any torchvision weight downloads in unit tests.
    import torch.hub

    def _blocked(*_args, **_kwargs):  # noqa: ANN001, ANN201
        raise AssertionError("Torchvision weight download is forbidden in unit tests.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _blocked, raising=True)

    mdir = tmp_path / "m"
    manifest = mdir / "manifest.jsonl"

    _write_png(mdir / "train_0.png", value=120)
    _write_png(mdir / "train_1.png", value=121)
    _write_png(mdir / "good.png", value=120)
    _write_png(mdir / "bad.png", value=240)

    _write_jsonl(
        manifest,
        [
            {"image_path": "train_0.png", "category": "bottle", "split": "train"},
            {"image_path": "train_1.png", "category": "bottle", "split": "train"},
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
            {"image_path": "bad.png", "category": "bottle", "split": "test", "label": 1},
        ],
    )

    out_dir = tmp_path / "run_out"
    model_kwargs = {
        "embedding_extractor": "torchvision_backbone",
        "embedding_kwargs": {
            "backbone": "resnet18",
            "pretrained": False,
            "pool": "avg",
            "device": "cpu",
            "batch_size": 2,
            "image_size": 16,
        },
        "core_detector": "core_ecod",
        "core_kwargs": {},
    }

    code = cli_main(
        [
            "--dataset",
            "manifest",
            "--root",
            str(mdir),
            "--manifest-path",
            str(manifest),
            "--category",
            "bottle",
            "--model",
            "vision_embedding_core",
            "--model-kwargs",
            json.dumps(model_kwargs),
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "16",
            "16",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    assert (out_dir / "categories" / "bottle" / "report.json").exists()
    assert (out_dir / "categories" / "bottle" / "per_image.jsonl").exists()

