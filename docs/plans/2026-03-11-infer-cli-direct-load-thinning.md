# Infer CLI Direct Load Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin the direct `--model` / `--model-preset` branch in `pyimgano.infer_cli` by moving detector setup into a reusable service module.

**Architecture:** Keep `pyimgano.infer_cli` responsible for CLI-only preprocessing such as argument parsing, ONNX session-option sweeps, and output formatting. Introduce `pyimgano.services.infer_setup_service` to own model preset alias resolution, preset kwargs composition, checkpoint requirement validation, and detector construction for the direct inference path.

**Tech Stack:** Python, argparse, pytest, dataclasses, existing model registry and model-options service.

---

### Task 1: Add a Direct Infer Setup Service

**Files:**
- Create: `pyimgano/services/infer_setup_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_setup_service.py`
- Test: `tests/test_infer_cli_smoke.py`

**Step 1: Write the failing test**

Create `tests/test_infer_setup_service.py` with:

```python
from __future__ import annotations

from pyimgano.services.infer_setup_service import DirectInferLoadRequest, load_direct_infer_detector


def test_load_direct_infer_detector_accepts_model_preset_alias() -> None:
    created: dict[str, object] = {}

    result = load_direct_infer_detector(
        DirectInferLoadRequest(requested_model="industrial-structural-ecod"),
        create_detector=lambda name, **kwargs: created.update(name=str(name), kwargs=dict(kwargs)) or object(),
    )

    assert result.model_name == "vision_feature_pipeline"
    assert created["name"] == "vision_feature_pipeline"
```

Add one CLI delegation assertion in `tests/test_infer_cli_smoke.py`:

```python
def test_infer_cli_direct_mode_delegates_detector_setup_to_service(tmp_path, monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.infer_setup_service as infer_setup_service

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    monkeypatch.setattr(
        infer_setup_service,
        "load_direct_infer_detector",
        lambda request, *, create_detector=None: infer_setup_service.DirectInferLoadResult(
            model_name="delegated-model",
            detector=_DummyDetector(),
            model_kwargs={},
        ),
    )

    rc = infer_cli.main(["--model", "vision_ecod", "--input", str(input_dir), "--save-jsonl", str(out_jsonl)])
    assert rc == 0
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_infer_setup_service.py tests/test_infer_cli_smoke.py -k "infer_setup_service or delegates_detector_setup_to_service" -v
```

Expected: FAIL because `pyimgano.services.infer_setup_service` does not exist and `pyimgano.infer_cli` still builds direct-mode detectors inline.

**Step 3: Write minimal implementation**

Create `pyimgano/services/infer_setup_service.py` with:

- `DirectInferLoadRequest`
- `DirectInferLoadResult`
- `load_direct_infer_detector(...)`

Migrate only the direct mode branch in `pyimgano.infer_cli` to call the service. Pass `infer_cli.create_model` into the service so existing tests that monkeypatch `infer_cli.create_model` keep working.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_infer_setup_service.py tests/test_infer_cli_smoke.py -v
```

Expected: PASS.
