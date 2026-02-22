# Industrial Inference (numpy-first) + `ImageFormat` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a production-friendly inference API that is numpy-first with explicit `ImageFormat`, upgrade key industrial detectors to accept canonical numpy inputs, and add an inference-oriented CLI for easy system integration.

**Architecture:** Introduce `pyimgano.inputs` for strict format declaration + canonicalization to `RGB/u8/HWC`, add `pyimgano.inference` for calibration + inference output contracts, and add an array-backed dataset to enable numpy inputs for deep-training-loop detectors. Upgrade a curated set of industrial detectors (PatchCore/PaDiM/SPADE/AnomalyDINO/SoftPatch/STFPM/DRAEM) to accept numpy images natively (canonical form) and tag capabilities for discovery.

**Tech Stack:** Python, NumPy, PyTorch, torchvision, Pillow, argparse, pytest.

---

### Task 1: Add `ImageFormat` + parser (strict, no guessing)

**Files:**
- Create: `pyimgano/inputs/__init__.py`
- Create: `pyimgano/inputs/image_format.py`
- Test: `tests/test_inputs_image_format.py`

**Step 1: Write the failing test**

```python
import pytest

from pyimgano.inputs.image_format import ImageFormat, parse_image_format


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("bgr_u8_hwc", ImageFormat.BGR_U8_HWC),
        ("rgb_u8_hwc", ImageFormat.RGB_U8_HWC),
        ("rgb_f32_chw", ImageFormat.RGB_F32_CHW),
    ],
)
def test_parse_image_format(raw, expected):
    assert parse_image_format(raw) is expected


def test_parse_image_format_rejects_unknown():
    with pytest.raises(ValueError):
        parse_image_format("auto")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_inputs_image_format.py -k parse`
Expected: FAIL (module doesn’t exist).

**Step 3: Write minimal implementation**

```python
# pyimgano/inputs/image_format.py
from enum import Enum

class ImageFormat(str, Enum):
    BGR_U8_HWC = "bgr_u8_hwc"
    RGB_U8_HWC = "rgb_u8_hwc"
    RGB_F32_CHW = "rgb_f32_chw"

def parse_image_format(raw: str) -> ImageFormat:
    try:
        return ImageFormat(str(raw))
    except Exception as exc:
        raise ValueError(f"Unknown image format: {raw!r}") from exc
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_inputs_image_format.py -k parse`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/inputs/__init__.py pyimgano/inputs/image_format.py tests/test_inputs_image_format.py
git commit -m "feat: add ImageFormat and strict parser"
```

---

### Task 2: Implement `normalize_numpy_image(..., input_format=...) -> RGB/u8/HWC`

**Files:**
- Modify: `pyimgano/inputs/image_format.py`
- Test: `tests/test_inputs_image_format.py`

**Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from pyimgano.inputs.image_format import ImageFormat, normalize_numpy_image


def test_normalize_bgr_u8_hwc_to_rgb_u8_hwc_swaps_channels():
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 30  # R
    rgb = normalize_numpy_image(bgr, input_format=ImageFormat.BGR_U8_HWC)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (2, 3, 3)
    assert np.all(rgb[..., 0] == 30)
    assert np.all(rgb[..., 1] == 20)
    assert np.all(rgb[..., 2] == 10)


def test_normalize_rgb_f32_chw_to_rgb_u8_hwc_scales_and_transposes():
    chw = np.ones((3, 4, 5), dtype=np.float32) * 0.5
    out = normalize_numpy_image(chw, input_format=ImageFormat.RGB_F32_CHW)
    assert out.shape == (4, 5, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 128


@pytest.mark.parametrize("shape", [(3, 3), (1, 3, 3), (3, 3, 4)])
def test_normalize_rejects_bad_shapes(shape):
    arr = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        normalize_numpy_image(arr, input_format=ImageFormat.RGB_U8_HWC)
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest -q tests/test_inputs_image_format.py -k normalize`
Expected: FAIL.

**Step 3: Implement minimal conversion**

Implement `normalize_numpy_image` with:
- strict shape checks by declared format
- `BGR_U8_HWC`: `[..., ::-1]`
- `RGB_U8_HWC`: identity + validation
- `RGB_F32_CHW`: transpose + scale by 255 + clip + round

**Step 4: Run to verify it passes**

Run: `.venv/bin/pytest -q tests/test_inputs_image_format.py -k normalize`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/inputs/image_format.py tests/test_inputs_image_format.py
git commit -m "feat: add normalize_numpy_image to canonical RGB/u8/HWC"
```

---

### Task 3: Add `to_torch_chw_float(...)` helper for deep detectors

**Files:**
- Create: `pyimgano/inputs/torch_ops.py`
- Test: `tests/test_inputs_torch_ops.py`

**Step 1: Write failing test**

```python
import numpy as np

from pyimgano.inputs.torch_ops import to_torch_chw_float


def test_to_torch_chw_float_shapes():
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    t = to_torch_chw_float(rgb, normalize=None)
    assert tuple(t.shape) == (3, 10, 20)
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest -q tests/test_inputs_torch_ops.py`
Expected: FAIL.

**Step 3: Implement helper**

Implement:
- input must be RGB/u8/HWC
- output is torch float32 CHW in [0,1]
- optional `normalize="imagenet"` applies mean/std

**Step 4: Run to verify it passes**

Run: `.venv/bin/pytest -q tests/test_inputs_torch_ops.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/inputs/torch_ops.py tests/test_inputs_torch_ops.py
git commit -m "feat: add torch conversion helper for canonical numpy images"
```

---

### Task 4: Add `VisionArrayDataset` for numpy-backed deep workflows

**Files:**
- Create: `pyimgano/datasets/array.py`
- Test: `tests/test_datasets_array.py`

**Step 1: Write failing test**

```python
import numpy as np

from pyimgano.datasets.array import VisionArrayDataset


def test_array_dataset_returns_tensor_pair():
    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    ds = VisionArrayDataset(images=imgs)
    x, y = ds[0]
    assert x.shape == y.shape
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest -q tests/test_datasets_array.py`
Expected: FAIL.

**Step 3: Implement minimal dataset**

Contract:
- input images are **canonical** `RGB/u8/HWC`
- dataset converts to torch tensor (CHW float in [0,1])
- returns `(image, image)` like `VisionImageDataset`

**Step 4: Run to verify it passes**

Run: `.venv/bin/pytest -q tests/test_datasets_array.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/datasets/array.py tests/test_datasets_array.py
git commit -m "feat: add VisionArrayDataset for numpy image inputs"
```

---

### Task 5: Teach `BaseVisionDeepDetector` to accept numpy lists (via `VisionArrayDataset`)

**Files:**
- Modify: `pyimgano/models/baseCv.py`
- Test: `tests/test_basecv_numpy_inputs.py`

**Step 1: Write failing test**

```python
import numpy as np
import pytest

import pyimgano.models.baseCv as baseCv


class DummyDeep(baseCv.BaseVisionDeepDetector):
    def build_model(self):
        self.model = object()
        return self.model

    def training_forward(self, batch):
        return 0.0

    def evaluating_forward(self, batch):
        import numpy as np
        x, _ = batch
        return np.zeros((x.shape[0],), dtype=np.float32)


def test_basecv_accepts_numpy_list(monkeypatch):
    imgs = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
    det = DummyDeep(epoch_num=1, batch_size=1, verbose=0, device="cpu")
    det.fit(imgs)
    scores = det.decision_function(imgs)
    assert scores.shape == (2,)
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest -q tests/test_basecv_numpy_inputs.py`
Expected: FAIL until `fit/decision_function` route to array dataset.

**Step 3: Implement routing**

In `BaseVisionDeepDetector.fit/decision_function`:
- if `X` is a list/tuple and first element is `np.ndarray`, use `VisionArrayDataset`
- else keep existing path behavior using `VisionImageDataset`

**Step 4: Run to verify it passes**

Run: `.venv/bin/pytest -q tests/test_basecv_numpy_inputs.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/baseCv.py tests/test_basecv_numpy_inputs.py
git commit -m "feat: allow BaseVisionDeepDetector to consume numpy arrays"
```

---

### Task 6: Add `pyimgano.inference` API (calibrate + infer + optional maps)

**Files:**
- Create: `pyimgano/inference/__init__.py`
- Create: `pyimgano/inference/api.py`
- Test: `tests/test_inference_api.py`

**Step 1: Write failing tests**

```python
import numpy as np

from pyimgano.inference.api import calibrate_threshold, infer
from pyimgano.inputs.image_format import ImageFormat


class ScoreOnly:
    def __init__(self):
        self.threshold_ = 0.5
    def decision_function(self, X):
        return np.asarray([0.1, 0.9], dtype=np.float32)


def test_infer_returns_scores_and_labels():
    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    out = infer(ScoreOnly(), imgs, input_format=ImageFormat.RGB_U8_HWC)
    assert out[0].score == 0.1
    assert out[1].label in (0, 1)
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest -q tests/test_inference_api.py`
Expected: FAIL.

**Step 3: Implement API**

Implement:
- `InferenceResult` dataclass: `score: float`, `label: int|None`, `anomaly_map: np.ndarray|None`
- `calibrate_threshold(...)` (quantile-based)
- `infer(...)`:
  - normalizes numpy input using `normalize_numpy_image`
  - calls detector `decision_function`
  - optionally calls `predict_anomaly_map` / `get_anomaly_map`

**Step 4: Run to verify it passes**

Run: `.venv/bin/pytest -q tests/test_inference_api.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/inference/__init__.py pyimgano/inference/api.py tests/test_inference_api.py
git commit -m "feat: add industrial inference API (numpy-first)"
```

---

### Task 7: Upgrade `vision_padim` to accept canonical numpy images

**Files:**
- Modify: `pyimgano/models/padim.py`
- Test: `tests/test_padim_numpy_inputs.py`

**Step 1: Write failing test**

```python
import numpy as np

from pyimgano.models import create_model


def test_padim_decision_function_accepts_numpy_images(monkeypatch):
    det = create_model("vision_padim", pretrained=False, device="cpu")
    imgs = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)
    scores = det.decision_function(imgs)
    assert scores.shape == (2,)
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest -q tests/test_padim_numpy_inputs.py`
Expected: FAIL (currently expects paths).

**Step 3: Implement**

Allow `fit/decision_function/get_anomaly_map/predict_anomaly_map` to accept iterable of:
- `str` paths (existing)
- `np.ndarray` canonical images

**Step 4: Run test**

Run: `.venv/bin/pytest -q tests/test_padim_numpy_inputs.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/padim.py tests/test_padim_numpy_inputs.py
git commit -m "feat: add numpy input support for vision_padim"
```

---

### Task 8: Upgrade `vision_patchcore` to accept canonical numpy images

**Files:**
- Modify: `pyimgano/models/patchcore.py`
- Test: `tests/test_patchcore_numpy_inputs.py`

**Steps:**
- Add tests analogous to Task 7 (fit + decision_function on numpy list).
- Implement `_extract_patch_features` to accept `str|np.ndarray` (canonical).
- Keep path support unchanged.

Run: `.venv/bin/pytest -q tests/test_patchcore_numpy_inputs.py`
Commit: `git commit -m "feat: add numpy input support for vision_patchcore"`

---

### Task 9: Upgrade `vision_stfpm` to accept canonical numpy images

**Files:**
- Modify: `pyimgano/models/stfpm.py`
- Test: `tests/test_stfpm_numpy_inputs.py`

**Steps:**
- Add a minimal test that:
  - constructs model with tiny epochs
  - runs `fit` and `decision_function` on 2 canonical images
- Implement image loading path to accept `str|np.ndarray`.

Run: `.venv/bin/pytest -q tests/test_stfpm_numpy_inputs.py`
Commit: `git commit -m "feat: add numpy input support for vision_stfpm"`

---

### Task 10: Upgrade `vision_draem` to accept canonical numpy images

**Files:**
- Modify: `pyimgano/models/draem.py`
- Test: `tests/test_draem_numpy_inputs.py`

**Steps:**
- Add minimal fit/score smoke test on 2 canonical images.
- Refactor image load helper(s) to accept `str|np.ndarray`.

Run: `.venv/bin/pytest -q tests/test_draem_numpy_inputs.py`
Commit: `git commit -m "feat: add numpy input support for vision_draem"`

---

### Task 11: Upgrade `vision_anomalydino` embedder to accept numpy images

**Files:**
- Modify: `pyimgano/models/anomalydino.py`
- Test: `tests/test_anomalydino_numpy_inputs.py`

**Steps:**
- Extend embedder interface to accept `str|np.ndarray` (canonical).
- Update default `TorchHubDinoV2Embedder` loader to accept numpy (convert to PIL).
- Add a unit test using an injected fake embedder to avoid downloading weights.

Run: `.venv/bin/pytest -q tests/test_anomalydino_numpy_inputs.py`
Commit: `git commit -m "feat: allow vision_anomalydino to embed numpy images"`

---

### Task 12: Upgrade `vision_softpatch` to accept numpy images (via updated embedder)

**Files:**
- Modify: `pyimgano/models/softpatch.py`
- Test: `tests/test_softpatch_numpy_inputs.py`

**Steps:**
- Change `_embed` to accept `image: str|np.ndarray`.
- Ensure `fit/decision_function/get_anomaly_map` accept numpy inputs.
- Keep injected-embedder tests fast; no torch required.

Run: `.venv/bin/pytest -q tests/test_softpatch_numpy_inputs.py`
Commit: `git commit -m "feat: add numpy input support for vision_softpatch"`

---

### Task 13: Add `vision_score_ensemble` (popular production pattern)

**Files:**
- Create: `pyimgano/models/score_ensemble.py`
- Modify: `pyimgano/models/__init__.py` (auto-import list)
- Test: `tests/test_score_ensemble.py`

**Steps:**
- Implement a wrapper that combines scores from multiple detectors.
- Default combine: mean of rank-normalized scores.
- Add tests with dummy detectors.

Run: `.venv/bin/pytest -q tests/test_score_ensemble.py`
Commit: `git commit -m "feat: add vision_score_ensemble wrapper detector"`

---

### Task 14: Add `pyimgano-infer` CLI (JSONL + optional map export)

**Files:**
- Create: `pyimgano/infer_cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_infer_cli_smoke.py`

**Steps:**
- CLI args: `--model`, `--preset`, `--checkpoint-path`, `--train-dir`, `--input`, `--save-jsonl`, `--save-maps`
- Uses `pyimgano.inference.api` for calibration + inference.
- Smoke test monkeypatching `create_model` + writing dummy images to tempdir.

Run: `.venv/bin/pytest -q tests/test_infer_cli_smoke.py`
Commit: `git commit -m "feat: add pyimgano-infer CLI for industrial scoring"`

---

### Task 15: Tag capabilities for discovery (`numpy`, `pixel_map`)

**Files:**
- Modify: `pyimgano/models/*.py` (for upgraded models)
- Modify: `tests/test_cli_discovery.py`

**Steps:**
- Add registry tags:
  - `numpy` for detectors that accept numpy images
  - `pixel_map` for detectors that implement `get_anomaly_map` / `predict_anomaly_map`
- Add a discovery test:
  - `main(["--list-models","--tags","numpy"])` includes `vision_spade` (already numpy) and upgraded models

Run: `.venv/bin/pytest -q tests/test_cli_discovery.py -k tags`
Commit: `git commit -m "feat: tag numpy/pixel_map capabilities for model discovery"`

---

### Task 16: Add JSON serialization helpers for inference results

**Files:**
- Modify: `pyimgano/inference/api.py`
- Test: `tests/test_inference_api.py`

**Steps:**
- Ensure JSON output is stable:
  - numpy scalars become Python scalars
  - anomaly maps can be saved separately; JSON stores metadata only (path/shape) by default

Run: `.venv/bin/pytest -q tests/test_inference_api.py`
Commit: `git commit -m "feat: stabilize inference result JSON serialization"`

---

### Task 17: Add optional postprocess hook for anomaly maps in inference API

**Files:**
- Modify: `pyimgano/inference/api.py`
- Test: `tests/test_inference_api.py`

**Steps:**
- Accept `postprocess: AnomalyMapPostprocess|None`
- Apply postprocess only to maps, never to image-level scores

Run: `.venv/bin/pytest -q tests/test_inference_api.py -k postprocess`
Commit: `git commit -m "feat: support optional anomaly-map postprocess in inference"`

---

### Task 18: Add docs page for industrial inference (numpy-first)

**Files:**
- Create: `docs/INDUSTRIAL_INFERENCE.md`
- Modify: `README.md`

**Steps:**
- Document:
  - `ImageFormat` and why explicit
  - calibration + infer workflow
  - detector capability tags (`numpy`, `pixel_map`)
- Add CLI example for `pyimgano-infer`.

Commit: `git commit -m "docs: add industrial inference guide (numpy-first)"`

---

### Task 19: Add runnable examples

**Files:**
- Create: `examples/industrial_infer_numpy.py`
- Create: `examples/industrial_score_ensemble.py`

**Steps:**
- Keep examples dependency-light; no weight downloads required by default.
- Show how to pass `input_format=...` and how to request maps.

Commit: `git commit -m "examples: add industrial inference + ensemble examples"`

---

### Task 20: Update selection guide with “industrial embedding” recommendations

**Files:**
- Modify: `docs/ALGORITHM_SELECTION_GUIDE.md`

**Steps:**
- Add a short section:
  - If you already have numpy frames -> use `pyimgano.inference`
  - Start with `vision_patchcore` / `vision_softpatch` / `vision_anomalydino`
  - Use `vision_score_ensemble` for robustness

Commit: `git commit -m "docs: recommend numpy-first inference path for industrial embedding"`

