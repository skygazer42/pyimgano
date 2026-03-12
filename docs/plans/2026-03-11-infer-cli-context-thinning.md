# Infer CLI Config Context Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin the `--from-run` and `--infer-config` setup branches in `pyimgano.infer_cli` by moving config-backed context loading into a service module.

**Architecture:** Keep `pyimgano.infer_cli` responsible for CLI-only argument validation, ONNX session-option sweeps, detector wrapping, and output. Introduce `pyimgano.services.infer_context_service` to own run/config loading, default extraction, checkpoint path resolution, threshold recovery, and inference-default payload assembly for config-backed inference.

**Tech Stack:** Python, argparse, pytest, dataclasses, pathlib, existing inference config and workbench load utilities.

---

### Task 1: Add Config-Backed Infer Context Service

**Files:**
- Create: `pyimgano/services/infer_context_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_context_service.py`
- Test: `tests/test_infer_cli_infer_config.py`

**Step 1: Write the failing test**

Create `tests/test_infer_context_service.py` with a focused infer-config context test:

```python
from __future__ import annotations

import json

from pyimgano.services.infer_context_service import InferConfigContextRequest, prepare_infer_config_context


def test_prepare_infer_config_context_returns_threshold_and_checkpoint(tmp_path) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "model.pt").write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False, "contamination": 0.1, "preset": None, "model_kwargs": {}, "checkpoint_path": None},
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            }
        ),
        encoding="utf-8",
    )

    context = prepare_infer_config_context(InferConfigContextRequest(config_path=str(infer_cfg_path)))
    assert context.model_name == "vision_ecod"
    assert context.threshold == 0.7
    assert context.trained_checkpoint_path.endswith("model.pt")
```

Add one CLI delegation assertion in `tests/test_infer_cli_infer_config.py`:

```python
def test_infer_cli_infer_config_delegates_context_loading(tmp_path: Path, monkeypatch) -> None:
    import pyimgano.services.infer_context_service as infer_context_service

    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text("{}", encoding="utf-8")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _Det:
        threshold_ = 0.5
        def decision_function(self, X):  # noqa: ANN001
            return np.asarray([0.1 for _ in list(X)], dtype=np.float32)

    monkeypatch.setattr(
        infer_context_service,
        "prepare_infer_config_context",
        lambda _request: infer_context_service.ConfigBackedInferContext(
            model_name="vision_ecod",
            preset=None,
            device="cpu",
            contamination=0.1,
            pretrained=False,
            base_user_kwargs={},
            checkpoint_path=None,
            trained_checkpoint_path=None,
            threshold=0.5,
            defects_payload=None,
            defects_payload_source=None,
            illumination_contrast_knobs=None,
            tiling_payload=None,
            infer_config_postprocess=None,
            enable_maps_by_default=False,
            warnings=(),
        ),
    )
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_infer_context_service.py tests/test_infer_cli_infer_config.py -k "infer_context_service or delegates_context_loading" -v
```

Expected: FAIL because `pyimgano.services.infer_context_service` does not exist and `pyimgano.infer_cli` still loads infer-config context inline.

**Step 3: Write minimal implementation**

Create `pyimgano/services/infer_context_service.py` with:

- `ConfigBackedInferContext`
- `FromRunInferContextRequest`
- `InferConfigContextRequest`
- `prepare_from_run_context(...)`
- `prepare_infer_config_context(...)`

Migrate the `--from-run` and `--infer-config` context-loading parts of `pyimgano.infer_cli` to call the service, but keep ONNX sweeps, `resolve_model_options`, detector creation, checkpoint loading, and downstream inference flow in the CLI.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_infer_context_service.py tests/test_infer_cli_infer_config.py -v
```

Expected: PASS.
