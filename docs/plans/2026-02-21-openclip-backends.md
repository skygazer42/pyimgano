# OpenCLIP Backend + CLIP-based Detectors Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pip-friendly OpenCLIP backend (`pyimgano[clip]`) and ship two CLIP-based detectors (`vision_openclip_patchknn`, `vision_openclip_promptscore`) with image-level scores + pixel-level anomaly maps, aligned to `pyimgano`’s unified API.

**Architecture:** Keep `pyimgano` core lightweight. Gate OpenCLIP imports behind optional deps. Implement detectors so tests can run without OpenCLIP by injecting fake embedders / text features. Reuse `vision_anomalydino` for PatchKNN behavior.

**Tech Stack:** Python, NumPy, OpenCV (for map upsampling), PyTorch; optional `open_clip_torch` (imported as `open_clip`).

---

### Task 1: Add `clip` optional extra (OpenCLIP)

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

**Step 1: Write a failing smoke test**

Create `tests/test_openclip_optional.py` asserting missing OpenCLIP yields a clean `ImportError` hint.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_openclip_optional.py -v`  
Expected: FAIL (file/module missing).

**Step 3: Implement minimal packaging change**

Add:

```toml
[project.optional-dependencies]
clip = [
  "open_clip_torch>=2.0.0",
]
```

And mention in `README.md`:

```bash
pip install "pyimgano[clip]"
```

**Step 4: Re-run test to verify it still fails**

Expected: still FAIL until Task 2/3 create the wrapper.

**Step 5: Commit**

```bash
git add pyproject.toml README.md tests/test_openclip_optional.py
git commit -m "build: add openclip optional extra"
```

---

### Task 2: Add `openclip_backend` module skeleton (no hard deps)

**Files:**
- Create: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_optional.py`

**Step 1: Write failing test**

In `tests/test_openclip_optional.py`:

```python
import pytest
from pyimgano import models


def test_openclip_models_registered_without_openclip_installed():
    assert "vision_openclip_patchknn" in models.list_models()
    assert "vision_openclip_promptscore" in models.list_models()


def test_openclip_promptscore_requires_openclip_if_no_injection():
    with pytest.raises(ImportError):
        models.create_model("vision_openclip_promptscore")
```

**Step 2: Run test**

Run: `pytest tests/test_openclip_optional.py -v`  
Expected: FAIL (models not registered / module missing).

**Step 3: Implement minimal code**

Create `pyimgano/models/openclip_backend.py` with:

- lazy `require("open_clip", extra="clip", purpose="OpenCLIP detectors")`
- placeholder classes registered in the registry
- constructors accept injection so tests can avoid open_clip

**Step 4: Run test**

Expected: PASS for registration + import behavior.

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_openclip_optional.py
git commit -m "feat: add openclip backend skeleton"
```

---

### Task 3: Auto-import OpenCLIP models in `pyimgano.models`

**Files:**
- Modify: `pyimgano/models/__init__.py`
- Test: `tests/test_models_import_optional.py`

**Step 1: Write failing test**

Extend `tests/test_models_import_optional.py` to assert OpenCLIP models are present in registry output.

**Step 2: Run test**

Run: `pytest tests/test_models_import_optional.py -v`  
Expected: FAIL.

**Step 3: Implement**

Add `"openclip_backend"` to the `_auto_import([...])` list.

**Step 4: Re-run test**

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/__init__.py tests/test_models_import_optional.py
git commit -m "feat: auto-import openclip backend models"
```

---

### Task 4: Implement `vision_openclip_patchknn` as an AnomalyDINO adapter (injectable)

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_patchknn.py`

**Step 1: Write failing test**

Create `tests/test_openclip_patchknn.py` using a fake embedder:

```python
import numpy as np
from pyimgano import models


class FakeEmbedder:
    def embed(self, image_path: str):
        # 4 patches, dim=2, deterministic by path
        base = 0.0 if "normal" in image_path else 10.0
        patches = np.array([[base, 0.0], [base, 1.0], [base, 2.0], [base, 3.0]], dtype=np.float32)
        return patches, (2, 2), (8, 8)


def test_openclip_patchknn_scores_higher_for_anomaly():
    detector = models.create_model("vision_openclip_patchknn", embedder=FakeEmbedder(), contamination=0.5)
    detector.fit(["normal_1.png", "normal_2.png"])
    scores = detector.decision_function(["normal_x.png", "anomaly_x.png"])
    assert scores[1] > scores[0]
```

**Step 2: Run red**

Run: `pytest tests/test_openclip_patchknn.py -v`  
Expected: FAIL (model not implemented or missing API).

**Step 3: Implement minimal adapter**

Make `vision_openclip_patchknn`:
- accept `embedder=...` injection
- otherwise default to an OpenCLIP patch embedder (added later)
- reuse AnomalyDINO scoring/thresholding

**Step 4: Run green**

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_openclip_patchknn.py
git commit -m "feat: add vision_openclip_patchknn (embedder-injectable)"
```

---

### Task 5: Add numpy helper for prompt-based patch scoring

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_promptscore_core.py`

**Step 1: Write failing test**

Create `tests/test_openclip_promptscore_core.py`:

```python
import numpy as np
from pyimgano.models.openclip_backend import _prompt_patch_scores


def test_prompt_patch_scores_diff_mode():
    patches = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text_normal = np.array([1.0, 0.0], dtype=np.float32)
    text_anom = np.array([0.0, 1.0], dtype=np.float32)
    scores = _prompt_patch_scores(patches, text_normal=text_normal, text_anomaly=text_anom, mode="diff")
    assert scores.shape == (2,)
    assert scores[1] > scores[0]
```

**Step 2: Run red**

Run: `pytest tests/test_openclip_promptscore_core.py -v`  
Expected: FAIL.

**Step 3: Implement helper**

Implement `_prompt_patch_scores(patch_embeddings, text_normal, text_anomaly, mode="diff"|"ratio")`
using cosine similarity on normalized vectors.

**Step 4: Run green**

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_openclip_promptscore_core.py
git commit -m "feat: add openclip promptscore core helpers"
```

---

### Task 6: Implement `vision_openclip_promptscore` with injectable embedder + text features

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_promptscore.py`

**Step 1: Write failing test**

Create `tests/test_openclip_promptscore.py`:

```python
import numpy as np
from pyimgano import models


class FakeEmbedder:
    def embed(self, image_path: str):
        base = 0.0 if "normal" in image_path else 5.0
        patches = np.array([[base, 0.0], [base, 1.0], [base, 2.0], [base, 3.0]], dtype=np.float32)
        return patches, (2, 2), (8, 8)


def test_openclip_promptscore_fit_predict_and_map():
    detector = models.create_model(
        "vision_openclip_promptscore",
        embedder=FakeEmbedder(),
        text_features_normal=np.array([1.0, 0.0], dtype=np.float32),
        text_features_anomaly=np.array([0.0, 1.0], dtype=np.float32),
        contamination=0.5,
    )
    detector.fit(["normal_1.png", "normal_2.png"])
    scores = detector.decision_function(["normal_x.png", "anomaly_x.png"])
    assert scores[1] > scores[0]

    amap = detector.get_anomaly_map("anomaly_x.png")
    assert amap.shape == (8, 8)
    assert np.isfinite(amap).all()
```

**Step 2: Run red**

Run: `pytest tests/test_openclip_promptscore.py -v`  
Expected: FAIL.

**Step 3: Implement minimal promptscore detector**

Implement:
- `fit()` calibrates `threshold_` from train scores
- `decision_function()` uses patch scoring helper + `topk_mean` aggregation (reuse AnomalyDINO helper)
- `get_anomaly_map()` reshapes to grid and upsamples to `(H, W)` (OpenCV; fallback allowed)

**Step 4: Run green**

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_openclip_promptscore.py
git commit -m "feat: add vision_openclip_promptscore (injectable)"
```

---

### Task 7: Add OpenCLIP lazy loader (model + preprocess)

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_optional.py`

**Step 1: Write failing test**

Add a test asserting constructing promptscore **without injection** raises ImportError when `open_clip` missing.

**Step 2: Run red**

**Step 3: Implement**

Add a small internal helper:
- `_require_open_clip()`
- `_load_openclip_model_and_preprocess(model_name, pretrained, device)`

**Step 4: Run green**

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_openclip_optional.py
git commit -m "feat: add openclip lazy loader"
```

---

### Task 8: Implement OpenCLIP patch-token extraction (best-effort)

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_patch_tokens_optional.py`

**Step 1: Add a skip-if-missing integration test**

Create `tests/test_openclip_patch_tokens_optional.py`:

- `pytest.importorskip("open_clip")`
- load one ViT model
- embed one tiny generated image
- assert:
  - embeddings shape is (num_patches, dim)
  - grid shape matches patch count

**Step 2: Run test**

Expected: SKIP if OpenCLIP not installed.

**Step 3: Implement extraction logic**

Implement `OpenCLIPViTPatchEmbedder.embed()`:
- load image with PIL
- apply OpenCLIP preprocess
- forward model in a way that returns patch tokens if possible
- infer patch grid shape; validate `grid_h * grid_w == num_patches`

**Step 4: Run tests**

Run: `pytest tests/test_openclip_patch_tokens_optional.py -v` (with OpenCLIP installed)  
Expected: PASS (in environments where OpenCLIP available).

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_openclip_patch_tokens_optional.py
git commit -m "feat: add OpenCLIP ViT patch embedder"
```

---

### Task 9: Wire default embedder for `vision_openclip_patchknn`

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_optional.py`

**Step 1: Write failing test**

Ensure `models.create_model("vision_openclip_patchknn")` raises clean ImportError if OpenCLIP missing.

**Step 2: Run red**

**Step 3: Implement**

If `embedder is None`, set default embedder = `OpenCLIPViTPatchEmbedder(...)` (lazy).

**Step 4: Run green**

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_openclip_optional.py
git commit -m "feat: default openclip embedder for patchknn"
```

---

### Task 10: Add text prompt formatting + caching for promptscore

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_openclip_promptscore.py`

**Step 1: Write failing test**

Add a test that `set_class_name()` updates prompts and does not recompute features when class unchanged.

**Step 2: Run red**

**Step 3: Implement**

- `set_class_name(class_name)`
- internal cache key: `(class_name, prompts)`
- store `text_features_normal/anomaly`

**Step 4: Run green**

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_openclip_promptscore.py
git commit -m "feat: cache text prompts for openclip promptscore"
```

---

### Task 11: Ensure anomaly-map shapes are consistent + stackable

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_backend_hardening.py`

**Step 1: Write failing test**

Add a test similar to anomalib hardening:
- `predict_anomaly_map()` raises if returned maps have inconsistent shapes.

**Step 2: Run red**

**Step 3: Implement**

Validate and raise clean `ValueError` if shapes mismatch.

**Step 4: Run green**

**Step 5: Commit**

```bash
git add pyimgano/models/openclip_backend.py tests/test_backend_hardening.py
git commit -m "fix: harden openclip anomaly-map stacking"
```

---

### Task 12: Add example script for MVTec/VisA pipeline

**Files:**
- Create: `examples/openclip_mvtec_visa.py`

**Step 1: Write a smoke test (optional)**

If examples are not tested, skip.

**Step 2: Implement example**

Demonstrate:
- choosing detector `"vision_openclip_promptscore"` or `"vision_openclip_patchknn"`
- fitting on normal paths
- evaluating via `pyimgano.pipelines.mvtec_visa.evaluate_split`

**Step 3: Commit**

```bash
git add examples/openclip_mvtec_visa.py
git commit -m "docs: add OpenCLIP pipeline example"
```

---

### Task 13: Update docs for OpenCLIP usage

**Files:**
- Modify: `README.md`
- Modify: `docs/DEEP_LEARNING_MODELS.md`

**Step 1: Update docs**

Add:
- install: `pip install "pyimgano[clip]"`
- usage examples for both detectors
- reminder: use `decision_function` for continuous scores

**Step 2: Commit**

```bash
git add README.md docs/DEEP_LEARNING_MODELS.md
git commit -m "docs: document openclip backend detectors"
```

---

### Task 14: Add registry tags + metadata for discoverability

**Files:**
- Modify: `pyimgano/models/openclip_backend.py`

**Step 1: Implement**

Ensure both models include tags like:
- `("vision", "deep", "clip", "openclip", ...)`

**Step 2: Commit**

```bash
git add pyimgano/models/openclip_backend.py
git commit -m "chore: tag openclip models in registry"
```

---

### Task 15: Add `clip` to `pyimgano[all]` bundle

**Files:**
- Modify: `pyproject.toml`

**Step 1: Implement**

Add `clip` to `all = [...]` so `pip install "pyimgano[all]"` includes OpenCLIP.

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: include clip extra in all bundle"
```

---

### Task 16: Add fast import test to ensure OpenCLIP is truly optional

**Files:**
- Create: `tests/test_openclip_import_optional.py`

**Step 1: Write test**

Test that:
- importing `pyimgano.models.openclip_backend` does not import `open_clip` eagerly

**Step 2: Commit**

```bash
git add tests/test_openclip_import_optional.py
git commit -m "test: ensure openclip backend is lazily imported"
```

---

### Task 17: Add minimal “weights cache” documentation

**Files:**
- Modify: `README.md`

**Step 1: Document**

Explain where OpenCLIP weights are cached by default (torch cache), and that users can pre-download
to avoid repeated downloads.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document openclip weight caching"
```

---

### Task 18: Run fast unit tests (no OpenCLIP required)

**Files:**
- N/A

**Step 1: Run**

Run: `pytest -m "not slow" -v`

**Step 2: Commit**

No commit; this is verification.

---

### Task 19: Compile check

**Files:**
- N/A

**Step 1: Run**

Run: `python -m compileall pyimgano tests examples docs`

---

### Task 20: Merge and push

**Files:**
- N/A

**Step 1: Merge**

```bash
git checkout main
git pull --ff-only
git merge --ff-only openclip-backends
git push
```

