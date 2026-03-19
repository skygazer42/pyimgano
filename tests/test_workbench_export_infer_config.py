import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyimgano.models.registry import MODEL_REGISTRY


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_train_cli_export_infer_config_writes_artifact(tmp_path):
    import cv2

    from pyimgano.train_cli import main

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, x):  # noqa: ANN001
            self.fit_inputs = list(x)
            return self

        def decision_function(self, x):  # noqa: ANN001
            n = len(list(x))
            if n == 0:
                return np.asarray([], dtype=np.float32)
            return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_export_infer_config_dummy_detector",
        _DummyDetector,
        tags=("classical",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    out_dir = tmp_path / "run_out"
    cfg = {
        "recipe": "industrial-adapt",
        "seed": 123,
        "dataset": {
            "name": "custom",
            "root": str(root),
            "category": "custom",
            "resize": [16, 16],
            "input_mode": "paths",
            "limit_train": 2,
            "limit_test": 2,
        },
        "model": {
            "name": "test_export_infer_config_dummy_detector",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
        },
        "output": {
            "output_dir": str(out_dir),
            "save_run": True,
            "per_image_jsonl": False,
        },
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path), "--export-infer-config"])
    assert code == 0

    infer_cfg_path = out_dir / "artifacts" / "infer_config.json"
    assert infer_cfg_path.exists()
    payload = json.loads(infer_cfg_path.read_text(encoding="utf-8"))

    assert payload["model"]["name"] == "test_export_infer_config_dummy_detector"
    assert "threshold" in payload
    prov = payload["threshold_provenance"]
    assert prov["method"] == "quantile"
    assert np.isclose(prov["quantile"], 0.9)
    assert prov["source"] == "contamination"
    assert np.isclose(prov["contamination"], 0.1)

    defects = payload["defects"]
    assert defects["enabled"] is False
    assert defects["pixel_threshold"] is None
    assert defects["pixel_threshold_strategy"] == "normal_pixel_quantile"
    assert np.isclose(defects["pixel_normal_quantile"], 0.999)
    assert defects["mask_format"] == "png"
    assert defects["roi_xyxy_norm"] is None
    assert defects["min_area"] == 0
    assert defects["open_ksize"] == 0
    assert defects["close_ksize"] == 0
    assert defects["fill_holes"] is False
    assert defects["max_regions"] is None


def test_train_cli_export_infer_config_delegates_to_workbench_service(tmp_path, monkeypatch):
    from pyimgano.recipes.registry import RECIPE_REGISTRY
    from pyimgano.train_cli import main

    def _dummy_recipe(cfg):  # noqa: ANN001 - test stub
        run_dir = Path(str(cfg.output.output_dir))
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "threshold": 0.5}

    RECIPE_REGISTRY.register(
        "test_workbench_service_export_recipe",
        _dummy_recipe,
        overwrite=True,
    )

    import pyimgano.services.workbench_service as workbench_service

    calls: list[dict[str, object]] = []
    expected_payload = {"sentinel": True, "threshold": 0.5}

    def _fake_build_infer_config_payload(*, config, report):  # noqa: ANN001 - service seam
        calls.append({"config": config, "report": dict(report)})
        return dict(expected_payload)

    monkeypatch.setattr(
        workbench_service,
        "build_infer_config_payload",
        _fake_build_infer_config_payload,
    )

    out_dir = tmp_path / "run_out"
    cfg = {
        "recipe": "test_workbench_service_export_recipe",
        "dataset": {"name": "custom", "root": str(tmp_path), "category": "custom"},
        "model": {"name": "vision_ecod"},
        "output": {"output_dir": str(out_dir), "save_run": True},
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path), "--export-infer-config"])
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["report"]["run_dir"] == str(out_dir)

    infer_cfg_path = out_dir / "artifacts" / "infer_config.json"
    assert infer_cfg_path.exists()
    assert json.loads(infer_cfg_path.read_text(encoding="utf-8"))["sentinel"] is True


def test_workbench_runner_does_not_import_cli_module(tmp_path, monkeypatch) -> None:
    import builtins

    from pyimgano.workbench.config import WorkbenchConfig

    original_import = builtins.__import__
    imported: list[str] = []

    def _guard_import(name, *args, **kwargs):  # noqa: ANN001 - import hook
        imported.append(str(name))
        if str(name) == "pyimgano.cli" or str(name).startswith("pyimgano.cli."):
            raise AssertionError("pyimgano.workbench.runner imported pyimgano.cli")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard_import)
    sys.modules.pop("pyimgano.workbench.runner", None)

    import pyimgano.services.workbench_service as workbench_service
    import pyimgano.workbench.runner as runner

    class _DummyDetector:
        def __init__(self):
            self.threshold_ = None

        def fit(self, x):  # noqa: ANN001 - test stub
            self.fit_inputs = list(x)
            return self

        def decision_function(self, x):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(x)), dtype=np.float32)

    monkeypatch.setattr(
        workbench_service,
        "create_workbench_detector",
        lambda *, config: _DummyDetector(),
    )

    root = tmp_path / "custom"
    for rel in [
        "train/normal/train_0.png",
        "train/normal/train_1.png",
        "test/normal/good_0.png",
        "test/anomaly/bad_0.png",
    ]:
        _write_png(root / rel)

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(root),
                "category": "custom",
                "resize": [16, 16],
                "input_mode": "paths",
                "limit_train": 2,
                "limit_test": 2,
            },
            "model": {
                "name": "vision_ecod",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "output": {"save_run": False, "per_image_jsonl": False},
        }
    )

    payload = runner.run_workbench(config=cfg, recipe_name="industrial-adapt")
    assert payload["threshold"] >= 0.0
    assert "pyimgano.cli" not in imported


def test_train_cli_export_deploy_bundle_copies_infer_config_and_checkpoint(tmp_path):
    from pyimgano.recipes.registry import RECIPE_REGISTRY
    from pyimgano.train_cli import main

    def _dummy_recipe(cfg):  # noqa: ANN001 - test stub
        run_dir = Path(str(cfg.output.output_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "checkpoints" / "custom" / "model.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("ckpt", encoding="utf-8")
        return {
            "run_dir": str(run_dir),
            "dataset": str(cfg.dataset.name),
            "category": str(cfg.dataset.category),
            "model": str(cfg.model.name),
            "threshold": 0.5,
            "threshold_provenance": {"method": "fixed", "source": "test"},
            "checkpoint": {"path": "checkpoints/custom/model.pt"},
        }

    RECIPE_REGISTRY.register(
        "test_export_deploy_bundle_dummy_recipe",
        _dummy_recipe,
        overwrite=True,
    )

    root = tmp_path / "custom"
    root.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "run_out"
    cfg = {
        "recipe": "test_export_deploy_bundle_dummy_recipe",
        "seed": 123,
        "dataset": {
            "name": "custom",
            "root": str(root),
            "category": "custom",
            "resize": [16, 16],
            "input_mode": "paths",
            "limit_train": 1,
            "limit_test": 1,
        },
        "model": {
            "name": "vision_ecod",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
        },
        "output": {
            "output_dir": str(out_dir),
            "save_run": True,
            "per_image_jsonl": False,
        },
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path), "--export-deploy-bundle"])
    assert code == 0

    bundle_dir = out_dir / "deploy_bundle"
    assert bundle_dir.exists()
    assert (bundle_dir / "infer_config.json").exists()

    copied_ckpt = bundle_dir / "checkpoints" / "custom" / "model.pt"
    assert copied_ckpt.exists()
    assert copied_ckpt.read_text(encoding="utf-8") == "ckpt"


def test_train_cli_export_deploy_bundle_copies_model_checkpoint_path_artifact(tmp_path):
    from pyimgano.recipes.registry import RECIPE_REGISTRY
    from pyimgano.train_cli import main

    def _dummy_recipe(cfg):  # noqa: ANN001 - test stub
        run_dir = Path(str(cfg.output.output_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "checkpoints" / "custom" / "model.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("ckpt", encoding="utf-8")

        # Model-level artifact expected by deployment wrappers (e.g. TorchScript/ONNX backbones).
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "backbone.onnx").write_text("onnx", encoding="utf-8")

        return {
            "run_dir": str(run_dir),
            "dataset": str(cfg.dataset.name),
            "category": str(cfg.dataset.category),
            "model": str(cfg.model.name),
            "threshold": 0.5,
            "threshold_provenance": {"method": "fixed", "source": "test"},
            "checkpoint": {"path": "checkpoints/custom/model.pt"},
        }

    RECIPE_REGISTRY.register(
        "test_export_deploy_bundle_copies_model_checkpoint_artifact_recipe",
        _dummy_recipe,
        overwrite=True,
    )

    root = tmp_path / "custom"
    root.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "run_out"
    cfg = {
        "recipe": "test_export_deploy_bundle_copies_model_checkpoint_artifact_recipe",
        "seed": 123,
        "dataset": {
            "name": "custom",
            "root": str(root),
            "category": "custom",
            "resize": [16, 16],
            "input_mode": "paths",
            "limit_train": 1,
            "limit_test": 1,
        },
        "model": {
            "name": "vision_onnx_ecod",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
            # Relative to artifacts/: this used to break deploy_bundle copy logic.
            "checkpoint_path": "backbone.onnx",
        },
        "output": {
            "output_dir": str(out_dir),
            "save_run": True,
            "per_image_jsonl": False,
        },
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path), "--export-deploy-bundle"])
    assert code == 0

    bundle_dir = out_dir / "deploy_bundle"
    assert bundle_dir.exists()
    assert (bundle_dir / "infer_config.json").exists()

    copied_artifact = bundle_dir / "backbone.onnx"
    assert copied_artifact.exists()
    assert copied_artifact.read_text(encoding="utf-8") == "onnx"


def test_train_cli_export_deploy_bundle_rewrites_absolute_model_checkpoint_path(tmp_path):
    from pyimgano.recipes.registry import RECIPE_REGISTRY
    from pyimgano.train_cli import main

    abs_artifact = tmp_path / "abs_backbone.onnx"
    abs_artifact.write_text("onnx_abs", encoding="utf-8")

    def _dummy_recipe(cfg):  # noqa: ANN001 - test stub
        run_dir = Path(str(cfg.output.output_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_dir": str(run_dir),
            "dataset": str(cfg.dataset.name),
            "category": str(cfg.dataset.category),
            "model": str(cfg.model.name),
            "threshold": 0.5,
            "threshold_provenance": {"method": "fixed", "source": "test"},
        }

    RECIPE_REGISTRY.register(
        "test_export_deploy_bundle_rewrites_abs_model_ckpt_recipe",
        _dummy_recipe,
        overwrite=True,
    )

    root = tmp_path / "custom"
    root.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "run_out"
    cfg = {
        "recipe": "test_export_deploy_bundle_rewrites_abs_model_ckpt_recipe",
        "seed": 123,
        "dataset": {
            "name": "custom",
            "root": str(root),
            "category": "custom",
            "resize": [16, 16],
            "input_mode": "paths",
            "limit_train": 1,
            "limit_test": 1,
        },
        "model": {
            "name": "vision_onnx_ecod",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
            "checkpoint_path": str(abs_artifact),
        },
        "output": {
            "output_dir": str(out_dir),
            "save_run": True,
            "per_image_jsonl": False,
        },
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path), "--export-deploy-bundle"])
    assert code == 0

    bundle_dir = out_dir / "deploy_bundle"
    infer_cfg_path = bundle_dir / "infer_config.json"
    assert infer_cfg_path.exists()
    payload = json.loads(infer_cfg_path.read_text(encoding="utf-8"))
    assert payload["model"]["checkpoint_path"] == "artifacts_abs/abs_backbone.onnx"

    copied = bundle_dir / "artifacts_abs" / "abs_backbone.onnx"
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == "onnx_abs"


def test_train_cli_export_deploy_bundle_stamps_infer_config_schema_version(tmp_path):
    from pyimgano.recipes.registry import RECIPE_REGISTRY
    from pyimgano.train_cli import main

    def _dummy_recipe(cfg):  # noqa: ANN001 - test stub
        run_dir = Path(str(cfg.output.output_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_dir": str(run_dir),
            "dataset": str(cfg.dataset.name),
            "category": str(cfg.dataset.category),
            "model": str(cfg.model.name),
            "threshold": 0.5,
            "threshold_provenance": {"method": "fixed", "source": "test"},
        }

    RECIPE_REGISTRY.register(
        "test_export_deploy_bundle_schema_version_recipe",
        _dummy_recipe,
        overwrite=True,
    )

    root = tmp_path / "custom"
    root.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "run_out"
    cfg = {
        "recipe": "test_export_deploy_bundle_schema_version_recipe",
        "dataset": {"name": "custom", "root": str(root), "category": "custom", "resize": [16, 16]},
        "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False, "contamination": 0.1},
        "output": {"output_dir": str(out_dir), "save_run": True, "per_image_jsonl": False},
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path), "--export-deploy-bundle"])
    assert code == 0

    payload = json.loads((out_dir / "deploy_bundle" / "infer_config.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_validate_cli_accepts_legacy_deploy_bundle_without_schema_version(tmp_path, capsys):
    from pyimgano.validate_infer_config_cli import main

    bundle = tmp_path / "deploy_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    cfg = bundle / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "defects": {"mask_format": "png"},
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1


def test_infer_cli_accepts_legacy_deploy_bundle_with_preprocessing_and_defects(
    tmp_path, monkeypatch
):
    import pyimgano.infer_cli as infer_cli

    run_dir = tmp_path / "run"
    bundle = run_dir / "deploy_bundle"
    ckpt_dir = bundle / "checkpoints" / "custom"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    cfg = bundle / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_patchcore",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "preprocessing": {
                    "illumination_contrast": {
                        "white_balance": "gray_world",
                    }
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {"pixel_threshold": 0.5, "mask_format": "png"},
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, x):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(x)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            _ = item
            m = np.zeros((4, 4), dtype=np.float32)
            m[1:3, 1:3] = 1.0
            return m

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(cfg),
            "--input",
            str(input_dir),
            "--defects",
            "--save-masks",
            str(masks_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert det.loaded == str(ckpt_path)

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert np.isclose(record["defects"]["pixel_threshold"], 0.5)


def test_train_cli_export_deploy_bundle_requires_pixel_threshold_when_defects_enabled(
    tmp_path, capsys
):
    from pyimgano.recipes.registry import RECIPE_REGISTRY
    from pyimgano.train_cli import main

    def _dummy_recipe(cfg):  # noqa: ANN001 - test stub
        run_dir = Path(str(cfg.output.output_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_dir": str(run_dir),
            "threshold": 0.5,
            "threshold_provenance": {"method": "fixed"},
        }

    RECIPE_REGISTRY.register(
        "test_export_deploy_bundle_requires_pixel_threshold_recipe",
        _dummy_recipe,
        overwrite=True,
    )

    root = tmp_path / "custom"
    root.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "run_out"
    cfg = {
        "recipe": "test_export_deploy_bundle_requires_pixel_threshold_recipe",
        "dataset": {"name": "custom", "root": str(root), "category": "custom", "resize": [16, 16]},
        "model": {
            "name": "vision_ecod",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
        },
        "defects": {
            "enabled": True,
            "pixel_threshold": None,
            "pixel_threshold_strategy": "normal_pixel_quantile",
        },
        "output": {"output_dir": str(out_dir), "save_run": True, "per_image_jsonl": False},
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path), "--export-deploy-bundle"])
    assert code == 2
    err = capsys.readouterr().err.lower()
    assert "pixel_threshold" in err

