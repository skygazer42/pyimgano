# Anomalib Checkpoint Benchmark Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano-benchmark` able to evaluate anomalib-trained checkpoints (image + pixel metrics) via a clean CLI (`--checkpoint-path`, `--model-kwargs`) while keeping anomalib optional.

**Architecture:** Extend `pyimgano/cli.py` to parse/merge model kwargs safely and to require checkpoints for anomalib-backed registry entries. Harden `VisionAnomalibCheckpoint` to normalize anomaly maps to a stable 2D `float32` contract for downstream pixel metrics.

**Tech Stack:** Python, `argparse`, `json`, `inspect`, NumPy; optional `anomalib` behind `pyimgano[anomalib]`.

---

### Task 1: Add `--model-kwargs` parser + JSON parsing helper

**Files:**
- Modify: `pyimgano/cli.py`
- Create: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Create `tests/test_cli_model_kwargs.py`:

```python
import pytest


def test_parse_model_kwargs_none_returns_empty_dict():
    from pyimgano.cli import _parse_model_kwargs

    assert _parse_model_kwargs(None) == {}


def test_parse_model_kwargs_requires_json_object():
    from pyimgano.cli import _parse_model_kwargs

    with pytest.raises(ValueError, match="JSON object"):
        _parse_model_kwargs("[1, 2, 3]")
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py`
Expected: FAIL (`ImportError` for `_parse_model_kwargs`).

**Step 3: Write minimal implementation**

In `pyimgano/cli.py`, add:
- `parser.add_argument("--model-kwargs", default=None, help="JSON dict for model constructor kwargs")`
- `_parse_model_kwargs(text)` that returns `{}` for `None`, else parses JSON and validates dict.

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "feat: add CLI model-kwargs JSON parsing"
```

---

### Task 2: Add `--checkpoint-path` + merge rules (precedence + conflict)

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Append to `tests/test_cli_model_kwargs.py`:

```python
def test_merge_checkpoint_path_sets_checkpoint_path():
    from pyimgano.cli import _merge_checkpoint_path

    out = _merge_checkpoint_path({}, checkpoint_path="/x.ckpt")
    assert out["checkpoint_path"] == "/x.ckpt"


def test_merge_checkpoint_path_detects_conflict():
    from pyimgano.cli import _merge_checkpoint_path

    with pytest.raises(ValueError, match="conflict"):
        _merge_checkpoint_path({"checkpoint_path": "/a.ckpt"}, checkpoint_path="/b.ckpt")
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py`
Expected: FAIL (`ImportError` for `_merge_checkpoint_path`).

**Step 3: Write minimal implementation**

In `pyimgano/cli.py`, implement `_merge_checkpoint_path(model_kwargs, checkpoint_path)`:
- If `checkpoint_path` is `None`: return `model_kwargs`
- If `checkpoint_path` is set and `checkpoint_path` key exists with different value: raise `ValueError` mentioning conflict
- Else set `checkpoint_path`

Add CLI flag:
- `parser.add_argument("--checkpoint-path", default=None, help="Checkpoint path for checkpoint-backed models")`

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "feat: add CLI checkpoint-path with merge validation"
```

---

### Task 3: Detect “requires checkpoint” via registry metadata

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/models/anomalib_backend.py`
- Modify: `tests/test_anomalib_backend_optional.py`

**Step 1: Write the failing test**

Update `tests/test_anomalib_backend_optional.py`:

```python
def test_anomalib_aliases_are_registered():
    from pyimgano.models import list_models

    anomalib_models = set(list_models(tags=["anomalib"]))
    assert "vision_patchcore_anomalib" in anomalib_models
    assert "vision_padim_anomalib" in anomalib_models
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_anomalib_backend_optional.py::test_anomalib_aliases_are_registered`
Expected: FAIL if tags/aliases are missing.

**Step 3: Write minimal implementation**

In `pyimgano/models/anomalib_backend.py`:
- Add `metadata={"backend": "anomalib", "requires_checkpoint": True, ...}` for all anomalib-backed registry entries.

In `pyimgano/cli.py`:
- Use `MODEL_REGISTRY.info(args.model).metadata` to determine if checkpoint is required.
- If required and no `checkpoint_path` present after merge: raise a clear `ValueError`.

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_anomalib_backend_optional.py::test_anomalib_aliases_are_registered`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py pyimgano/models/anomalib_backend.py tests/test_anomalib_backend_optional.py
git commit -m "feat: require anomalib checkpoints via registry metadata"
```

---

### Task 4: Add safe constructor kwargs validation (unknown user keys)

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Append to `tests/test_cli_model_kwargs.py`:

```python
def test_validate_user_kwargs_rejects_unknown_keys_for_strict_models():
    from pyimgano.cli import _validate_user_model_kwargs

    with pytest.raises(TypeError, match="does not accept"):
        _validate_user_model_kwargs("vision_abod", {"not_a_param": 1})
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py::test_validate_user_kwargs_rejects_unknown_keys_for_strict_models`
Expected: FAIL (`ImportError` for `_validate_user_model_kwargs`).

**Step 3: Write minimal implementation**

In `pyimgano/cli.py`, implement:
- `_get_model_signature_info(model_name)` that returns `(accepted_keys, accepts_var_kwargs)`
- `_validate_user_model_kwargs(model_name, user_kwargs)`:
  - If accepts `**kwargs`: accept all
  - Else: reject keys not in accepted_keys with a `TypeError`

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py::test_validate_user_kwargs_rejects_unknown_keys_for_strict_models`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "feat: validate user model-kwargs against constructor signature"
```

---

### Task 5: Build final kwargs (user wins; drop unsupported auto defaults)

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Append to `tests/test_cli_model_kwargs.py`:

```python
def test_build_model_kwargs_does_not_override_user_values():
    from pyimgano.cli import _build_model_kwargs

    out = _build_model_kwargs(
        "vision_patchcore",
        user_kwargs={"device": "cpu"},
        auto_kwargs={"device": "cuda", "contamination": 0.1},
    )
    assert out["device"] == "cpu"
    assert out["contamination"] == 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py::test_build_model_kwargs_does_not_override_user_values`
Expected: FAIL (helper missing).

**Step 3: Write minimal implementation**

Implement `_build_model_kwargs(model_name, user_kwargs, auto_kwargs)`:
- Validate `user_kwargs` first
- Start from `dict(user_kwargs)`
- Add each `auto_kwargs` key only if absent in user_kwargs
- If the model is strict (no `**kwargs`), drop auto keys not accepted

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py::test_build_model_kwargs_does_not_override_user_values`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "feat: build model kwargs with user precedence"
```

---

### Task 6: Wire helpers into CLI `main()` model creation

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_smoke.py`

**Step 1: Write the failing test**

Add a new smoke test to ensure a strict/classical model can run without receiving `device/pretrained`:

```python
def test_cli_can_run_with_classical_model(tmp_path, capsys):
    from pyimgano.cli import main

    # Reuse the minimal dataset structure from the existing smoke test (good only).
    root = tmp_path / "mvtec"
    cat = "bottle"
    (root / cat / "train" / "good").mkdir(parents=True)
    (root / cat / "test" / "good").mkdir(parents=True)

    import cv2
    import numpy as np
    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(root / cat / "train" / "good" / "train_0.png"), img)
    cv2.imwrite(str(root / cat / "test" / "good" / "good_0.png"), img)

    code = main(
        [
            "--dataset",
            "mvtec",
            "--root",
            str(root),
            "--category",
            cat,
            "--model",
            "vision_abod",
            "--device",
            "cpu",
            "--no-pretrained",
        ]
    )
    assert code == 0
    assert "results" in capsys.readouterr().out
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_cli_smoke.py`
Expected: FAIL until CLI stops passing unsupported kwargs.

**Step 3: Write minimal implementation**

In `pyimgano/cli.py`:
- Parse user kwargs via `_parse_model_kwargs(args.model_kwargs)`
- Merge `--checkpoint-path`
- Build final kwargs with `_build_model_kwargs` using auto defaults:
  - `device`, `contamination`, `pretrained`
- Create model via `create_model(args.model, **final_kwargs)`

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_cli_smoke.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_smoke.py
git commit -m "feat: CLI builds model kwargs safely for all models"
```

---

### Task 7: Add friendly CLI error handling (return non-zero)

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Add to `tests/test_cli_model_kwargs.py` (monkeypatch-based):

```python
def test_cli_returns_nonzero_on_checkpoint_conflict(monkeypatch):
    import pyimgano.cli as cli

    def fake_load_benchmark_split(*_args, **_kwargs):
        return object()

    def fake_evaluate_split(*_args, **_kwargs):
        return {"image_metrics": {"auroc": 0.0}}

    monkeypatch.setattr(cli, "load_benchmark_split", fake_load_benchmark_split)
    monkeypatch.setattr(cli, "evaluate_split", fake_evaluate_split)

    code = cli.main(
        [
            "--dataset",
            "mvtec",
            "--root",
            "/tmp",
            "--category",
            "bottle",
            "--model",
            "vision_patchcore",
            "--checkpoint-path",
            "/a.ckpt",
            "--model-kwargs",
            '{"checkpoint_path": "/b.ckpt"}',
        ]
    )
    assert code != 0
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py::test_cli_returns_nonzero_on_checkpoint_conflict`
Expected: FAIL (exception currently bubbles).

**Step 3: Write minimal implementation**

Wrap `main()` core logic with `try/except Exception`:
- On error: `print(f"error: {exc}", file=sys.stderr)` and return `1`.

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py::test_cli_returns_nonzero_on_checkpoint_conflict`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "fix: return non-zero on CLI runtime errors"
```

---

### Task 8: Normalize anomalib anomaly map shapes to 2D float32

**Files:**
- Modify: `pyimgano/models/anomalib_backend.py`
- Modify: `tests/test_anomalib_backend_wrapper.py`

**Step 1: Write the failing test**

Append to `tests/test_anomalib_backend_wrapper.py`:

```python
def test_anomalib_checkpoint_wrapper_normalizes_map_shapes():
    from pyimgano.models.anomalib_backend import VisionAnomalibCheckpoint

    paths = ["a.png"]
    scores = {"a.png": 0.1}
    maps = {"a.png": np.ones((1, 3, 5), dtype=np.float32)}
    inferencer = _FakeInferencerDict(scores, maps)

    model = VisionAnomalibCheckpoint(
        checkpoint_path="ignored.ckpt",
        inferencer=inferencer,
        contamination=0.1,
        device="cpu",
    )

    m = model.get_anomaly_map("a.png")
    assert m.shape == (3, 5)
    assert m.dtype == np.float32
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_anomalib_backend_wrapper.py::test_anomalib_checkpoint_wrapper_normalizes_map_shapes`
Expected: FAIL (shape is `(1, 3, 5)` today).

**Step 3: Write minimal implementation**

In `pyimgano/models/anomalib_backend.py`:
- Add `_normalize_anomaly_map(map)` helper
- Call it inside `get_anomaly_map`

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_anomalib_backend_wrapper.py::test_anomalib_checkpoint_wrapper_normalizes_map_shapes`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/anomalib_backend.py tests/test_anomalib_backend_wrapper.py
git commit -m "fix: normalize anomalib anomaly map outputs"
```

---

### Task 9: Expand anomalib alias coverage (popular models)

**Files:**
- Modify: `pyimgano/models/anomalib_backend.py`
- Modify: `tests/test_anomalib_backend_optional.py`

**Step 1: Write the failing test**

Update `tests/test_anomalib_backend_optional.py`:

```python
def test_more_anomalib_aliases_are_registered():
    from pyimgano.models import list_models

    anomalib_models = set(list_models(tags=["anomalib"]))
    assert "vision_csflow_anomalib" in anomalib_models
    assert "vision_dsr_anomalib" in anomalib_models
    assert "vision_uflow_anomalib" in anomalib_models
    assert "vision_winclip_anomalib" in anomalib_models
```

**Step 2: Run test to verify it fails**

Run: `pytest -q -o addopts='' tests/test_anomalib_backend_optional.py::test_more_anomalib_aliases_are_registered`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add alias registry entries in `pyimgano/models/anomalib_backend.py`:
- `vision_csflow_anomalib`
- `vision_dfkde_anomalib`
- `vision_dsr_anomalib`
- `vision_fre_anomalib`
- `vision_ganomaly_anomalib`
- `vision_uflow_anomalib`
- `vision_winclip_anomalib`
- `vision_supersimplenet_anomalib`

Each should set `metadata.anomalib_model` appropriately.

**Step 4: Run test to verify it passes**

Run: `pytest -q -o addopts='' tests/test_anomalib_backend_optional.py::test_more_anomalib_aliases_are_registered`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/anomalib_backend.py tests/test_anomalib_backend_optional.py
git commit -m "feat: expand anomalib checkpoint alias coverage"
```

---

### Task 10: Document anomalib checkpoint benchmarking in README

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

N/A (docs-only).

**Step 2: Implement doc update**

Add a short snippet showing:
- install extra: `pip install 'pyimgano[anomalib]'`
- run: `pyimgano-benchmark --model vision_patchcore_anomalib --checkpoint-path ... --pixel`

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add anomalib checkpoint benchmarking example"
```

---

### Task 11: Add a dedicated docs section (optional, if README gets too long)

**Files:**
- Create: `docs/ANOMALIB_CHECKPOINTS.md`
- Modify: `docs/QUICKSTART.md`

**Steps:**
- Add a concise guide: “Train with anomalib, evaluate with pyimgano”.
- Link from Quickstart.
- Commit: `git commit -m "docs: add anomalib checkpoint evaluation guide"`

---

### Task 12: Add a CLI smoke test for anomalib kwargs merging (monkeypatch)

**Files:**
- Modify: `tests/test_cli_model_kwargs.py`

**Steps:**
- Monkeypatch `pyimgano.cli.create_model` to capture kwargs.
- Ensure `--checkpoint-path` and `--model-kwargs` merge into `checkpoint_path`.
- Commit: `git commit -m "test: cover CLI anomalib checkpoint kwargs merging"`

---

### Task 13: Ensure `pyimgano-benchmark --help` documents new flags

**Files:**
- Modify: `pyimgano/cli.py`

**Steps:**
- Improve `help=` strings for `--checkpoint-path` and `--model-kwargs`.
- Commit: `git commit -m "docs: clarify benchmark CLI help for checkpoints"`

---

### Task 14: Add robust error messages for common mistakes

**Files:**
- Modify: `pyimgano/cli.py`

**Steps:**
- Clear messages for:
  - invalid JSON in `--model-kwargs`
  - non-object JSON
  - missing checkpoint for `requires_checkpoint` models
  - conflict between `--checkpoint-path` and `checkpoint_path`
- Commit: `git commit -m "fix: improve CLI error messages for model kwargs"`

---

### Task 15: Keep pixel metrics stable for anomalib anomaly maps

**Files:**
- Modify: `pyimgano/models/anomalib_backend.py`

**Steps:**
- Ensure `predict_anomaly_map()` stacks maps consistently after normalization.
- Commit: `git commit -m "fix: ensure anomalib batch anomaly maps are consistent"`

---

### Task 16: Add optional web research note to docs (links)

**Files:**
- Modify: `docs/ANOMALIB_CHECKPOINTS.md` (or README if Task 11 skipped)

**Steps:**
- Link to anomalib “supported models” docs page for reference.
- Commit: `git commit -m "docs: link to anomalib supported models reference"`

---

### Task 17: Run focused test suite

**Files:**
- N/A

**Steps:**
- Run: `pytest -q -o addopts='' tests/test_cli_model_kwargs.py tests/test_anomalib_backend_wrapper.py`
- Expected: PASS

---

### Task 18: Run full test suite

**Files:**
- N/A

**Steps:**
- Run: `pytest -q`
- Expected: PASS

---

### Task 19: Merge branch to `main`

**Files:**
- N/A

**Steps:**
- From repo root: `git merge anomalib-benchmark-coverage`
- Expected: fast-forward or clean merge

---

### Task 20: Push to `origin/main`

**Files:**
- N/A

**Steps:**
- Run: `git push origin main`
- Expected: success

